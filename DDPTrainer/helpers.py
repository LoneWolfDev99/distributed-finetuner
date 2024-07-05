import base64
import csv
import json
import logging
import os
import pathlib
import subprocess as sp
import sys
import time
from datetime import datetime

import pandas as pd
import pyarrow.parquet as parquet
import wandb
from datasets import load_dataset
from e2enetworks.cloud import tir
from e2enetworks.cloud.tir.minio_service import MinioService
from tensorboard.backend.event_processing import event_accumulator
from transformers import TrainerCallback

ARROW = 'arrow'
CSV = 'csv'
JSON = 'json'
PARQUET = 'parquet'
ALLOWED_FILE_TYPES = [ARROW, CSV, JSON, PARQUET]
DATASET_DOWNLOAD_PATH = '/mnt/workspace/custom_dataset/'
LAST_RUN_INFO_PATH = '/mnt/workspace/last_run.json'
LOCAL_MODEL_PATH = '/mnt/workspace/local_model/'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_dataset(dataset_id, dataset_path='') -> str:
    datasets_client = tir.Datasets()
    datasets_client.download_model(dataset_id, local_path=DATASET_DOWNLOAD_PATH, prefix=dataset_path)
    return f"{DATASET_DOWNLOAD_PATH}{dataset_path}" if dataset_path else DATASET_DOWNLOAD_PATH


def get_dataset_format(path: str):
    # function to return the file extension
    file_extension = pathlib.Path(path).suffix
    print(f"Dataset file {path} extension {file_extension}")
    return "json" if file_extension[1:] in ["json", "jsonl"] else file_extension[1:]


def prepare_prompt(example, columns, prompt_template):
    if len(columns) > 0:
        output_text = prompt_template
        for c in columns:
            output_text = output_text.replace(
                '[{}]'.format(c), example[c])
        example["training_text"] = output_text
    return example


def check_file_type(file_path) -> tuple[bool, str]:
    logger.info(f"file name is {file_path}")
    _, file_extension = os.path.splitext(file_path)
    if file_extension.startswith("."):
        file_extension = file_extension.lstrip('.').lower()
    if file_extension in ALLOWED_FILE_TYPES:
        return True, file_extension
    return False, file_extension


def get_allowed_files(dataset_folder) -> str:
    files = []
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        is_valid_file, file_extension = check_file_type(file_path)
        if os.path.isfile(file_path) and is_valid_file:
            files.append(file_path)
    if not files:
        logger.error(f"ERROR_UNSUPPORTED_FILES_GIVEN, ALLOWED_TYPES={ALLOWED_FILE_TYPES}")
        return ""
    logger.info(f"FILES ARE -> {files}")
    return files[0]


def retry_decorator(func_object):
    def wrapper(*args, **kwargs):
        for retry_count in range(3):
            try:
                func_object(*args, **kwargs)
                break
            except Exception as e:
                logger.error(f"OOPS_AN_ERROR_OCCURRED | {e}")
                if retry_count == 2:
                    raise e
                time.sleep(10)
                continue
    return wrapper


def main_process_decorator(func_object):
    def wrapper(*func_args, **kwargs):
        state = func_args[2]
        if not state.is_world_process_zero:
            return
        func_object(*func_args, **kwargs)
    return wrapper


def set_run_value(key_name: str, value: str | int) -> bool:
    if os.path.exists(LAST_RUN_INFO_PATH):
        with open(LAST_RUN_INFO_PATH) as run_file:
            run_info_dict = json.loads(run_file.read())
    else:
        run_info_dict = {}
    run_info_dict[key_name] = value
    with open(LAST_RUN_INFO_PATH, 'w') as run_file:
        run_file.write(json.dumps(run_info_dict))
        logger.info(f"VALUE_SET | key_name={key_name} -> value={value}")


def get_run_value(key_name):
    if not os.path.exists(LAST_RUN_INFO_PATH):
        return None
    with open(LAST_RUN_INFO_PATH) as run_file:
        run_info_dict = json.loads(run_file.read())
    return run_info_dict.get(key_name)


@retry_decorator
def push_model(model_path: str, info: dict = {}):
    model_repo_client = tir.Models()
    job_id = os.getenv("E2E_TIR_FINETUNE_JOB_ID")
    timestamp = datetime.now().strftime("%s")
    model_id = get_run_value('model_id')
    if not model_id:
        model_repo = model_repo_client.create(f"mistral7binst-{job_id}-{timestamp}", model_type="custom", job_id=job_id, score=info)
        model_id = model_repo.id
        set_run_value('model_id', model_id)
    else:
        model_repo_client._update_repo(model_id, score=info)
    model_repo_client.push_model(model_path=model_path, prefix='', model_id=model_id)


def download_folder_from_repo(model_id: int, model_path: str):
    model_repo_client = tir.Models()
    model_repo_client.download_model(model_id, local_path=LOCAL_MODEL_PATH, prefix=model_path)


def load_custom_dataset(dataset_path):
    file_type = get_dataset_format(dataset_path)
    try:
        if file_type == CSV:
            return load_dataset(CSV, data_files=dataset_path, on_bad_lines="warn", split="train")
        else:
            return load_dataset(file_type, data_files=dataset_path, split="train")
    except Exception as e:
        logger.error(f"ERROR_IN_LOADING_DATASET={e}")
        sys.exit(e)


def decode_base64(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string)
    decoded_text = decoded_bytes.decode('utf-8')
    return decoded_text


def gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def json_loader(file_path):
    file = open(file_path, 'r')
    json_data = json.load(file)
    if type(json_data) != list:
        raise Exception("Invalid json data, expected list/tabular")
    for row in json_data:
        yield row


def csv_loader(file_path):
    file = open(file_path, 'r')
    csv_reader = csv.reader(file)
    """ here csv_reader is generator object """
    return csv_reader


def parquet_loader(file_path):
    ''' yield id, value'''
    parquet_file = parquet.ParquetFile(file_path)
    for rows in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(rows)
        row_group = row_group.to_pandas()
        for row in row_group.iterrows():
            yield row


def arrow_loader(file_path):
    ''' yield id, value'''
    arrow_file_contents = pd.read_feather(file_path)
    for row in arrow_file_contents.iterrows():
        yield row


def initialize_wandb(script_args, last_checkpoint=None):
    try:
        if last_checkpoint is not None:
            run = resume_previous_run(script_args)
        else:
            run = wandb.init(
                name=script_args.wandb_run_name,
                project=script_args.wandb_project)
    except Exception as e:
        logger.warning(f"WANDB: Failed to create run: {e}")
        return
    logger.info(f"WANDB: Run is created with name: {run.name}, project: {script_args.wandb_project}")


def resume_previous_run(script_args):
    wandb_api = wandb.Api(overrides={"project": script_args.wandb_project})
    for run in wandb_api.runs(path=script_args.wandb_project):
        logger.info(f"PRIOR RUN: {run} {run.name} {run.id} {run.state}")
        if run.state in ["crashed", "failed"] and run.name == script_args.wandb_run_name:
            logger.info(f"CHECKPOINT: Resuming {run.id}")
            return wandb.init(
                id=run.id,
                project=script_args.wandb_project,
                resume="must",
                name=run.name,
            )
    return None


def make_finetuning_metric_json(output_dir):
    all_metrics = {}
    ea = event_accumulator.EventAccumulator(f"{output_dir}tensorboard_logs/")
    ea.Reload()
    logger.info(f"PREPARING_METRIC_JSON | Tags={ea.Tags()}")
    all_metric_json_path = f"{output_dir}tensorboard_logs/all_finetuning_metric.json"
    for key_name in ea.Tags()['scalars']:
        update_metric_dict(ea, all_metrics, key_name)
    make_metric_json_file(all_metric_json_path, all_metrics)


def make_metric_json_file(all_metric_json_path: str, all_metrics: dict):
    with open(all_metric_json_path, 'w') as file:
        file.write(json.dumps(all_metrics))


def update_metric_dict(ea, all_metric_json, key_name):
    try:
        truncated_df = df = pd.DataFrame(ea.Scalars(key_name))
        if len(df) >= 2000:
            truncated_df = df[0::20]
        all_metric_json[key_name] = truncated_df.to_dict()
    except Exception as e:
        logger.error(f"MAKE_FINETUNING_METRIC_JSON | KEY_NAME={key_name} | ERROR={e}")


class TrainingCallback(TrainerCallback):

    @main_process_decorator
    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            make_finetuning_metric_json(args.output_dir)
            push_model(args.output_dir, {})
        except Exception as e:
            logger.error(f"EXPORTER_CALLBACK_FAILED | ERROR={e}")

    @main_process_decorator
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            print(f"Saving checkpoint-{state.global_step}")
            control.should_save = True
