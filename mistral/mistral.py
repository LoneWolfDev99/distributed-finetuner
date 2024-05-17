import logging
import math
import os
import random
import sys
import time
import base64
import wandb
import torch
import transformers
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import re
import pathlib
import subprocess as sp
from datasets import load_dataset
from e2enetworks.cloud import tir
from e2enetworks.cloud.tir.minio_service import MinioService
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    HfArgumentParser, TrainingArguments)
from trl import SFTTrainer

logger = logging.getLogger(__name__)

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    output_dir: Optional[str] = field(default=None, metadata={"help": "Out directory to store model"})

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="mlabonne/guanaco-llama2-1k", metadata={"help": "the dataset name"}
    )
    dataset_type: Optional[str] = field(default="huggingface", metadata={"help": "the dataset source. Options: huggingface or eos-bucket"})
    dataset_bucket: Optional[str] = field(default="", metadata={"help": "the bucket when dataset type is eos bucket"})
    dataset_path: Optional[str] = field(default="", metadata={"help": "the bucket path when dataset type is eos bucket"})
    dataset_accesskey: Optional[str] = field(default="", metadata={"help": "the bucket access key when dataset type is eos bucket"})
    dataset_secretkey: Optional[str] = field(default="", metadata={"help": "the bucket secret key when dataset type is eos bucket"})

    log_level: Optional[str] = field(default="info", metadata={"help": "log level"})
    dataset_split: Optional[float] = field(default=0, metadata={"help": "training split ratio"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2.5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=500, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=300, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    run_name: Optional[str] = field(default=None, metadata={"help": "TIR finetuning job name"})
    auto_find_batch_size: Optional[str] = field(default=False, metadata={"help": "Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (pip install accelerate)"})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "wandb project"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Run name for wandb project"})
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    prompt_template_base64: Optional[str] = field(default=None, metadata={"help": "prompt template in base64"})
    resume: Optional[str] = field(default=True, metadata={"help": "resume from last checkpoint"})


def gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def download_dataset(script_args) -> str:
    try:
        dataset_download_path = '/home/jovyan/custom_dataset/'
        minio_service = MinioService(
            access_key=script_args.dataset_accesskey,
            secret_key=script_args.dataset_secretkey)
        minio_service.download_directory_recursive(
            bucket_name=script_args.dataset_bucket,
            local_path=dataset_download_path,
            prefix=script_args.dataset_path)
        logger.info(f"Dataset downloaded to {dataset_download_path}{script_args.dataset_path}")
        return f"{dataset_download_path}{script_args.dataset_path}" if script_args.dataset_path else dataset_download_path
    except Exception as e:
        logger.error(e)
        raise Exception(f"dataset_error -> {e}")


def retry_push_model(func_object):
    def wrapper(*args, **kwargs):
        for i in range(3):
            try:
                return func_object(*args, **kwargs)
            except Exception as e:
                logger.error(f"ERROR_DURING_PUSH_MODEL | {e}")
                time.sleep(10)
                continue
    return wrapper


def get_dataset_format(path: str):
    # function to return the file extension
    file_extension = pathlib.Path(path).suffix
    print(f"Dataset file {path} extension {file_extension}")
    return "json" if file_extension[1:] in ["json", "jsonl"] else file_extension[1:]


@retry_push_model
def push_model(model_path: str, info: dict = {}):
    model_repo_client = tir.Models()
    job_id = os.getenv("E2E_TIR_FINETUNE_JOB_ID")
    timestamp = datetime.now().strftime("%s")
    model_repo = model_repo_client.create(f"mistral7b-{job_id}-{timestamp}", model_type="custom", job_id=job_id, score=info)
    model_id = model_repo.id
    model_repo_client.push_model(
        model_path=model_path, prefix='',
        model_id=model_id)
    return True


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


def main():
    parser = HfArgumentParser(ScriptArguments)
    output = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    script_args = output[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    gpufree = gpu_memory()
    logger.info(f"starting the job. gpu memory free -  {gpufree}")
    logger.info(f"Script parameters {script_args}")

    # initiate tir
    tir.init()

    # Step 2: Load the dataset
    if script_args.dataset_type == "eos-bucket":
        dataset_path = download_dataset(script_args)

        dataset_type = get_dataset_format(dataset_path)
        logger.info(f"loading dataset from {dataset_path}")
        train_dataset = load_dataset(dataset_type, data_files=[dataset_path], split="train")

    else:
        logger.info(f"loading dataset {script_args.dataset_name} from huggingface")
        train_dataset = load_dataset(script_args.dataset_name, split="train")

    if script_args.prompt_template_base64:
        prompt_template = str(base64.b64decode(script_args.prompt_template_base64))
        logger.info(f"adding text column to dataset {prompt_template}")
        columns = re.findall(r'\[(.*?)\]', prompt_template)
        logger.info(f"found {len(columns)} columns in prompt template. replacing them")
        if len(columns) == 0:
            raise Exception("invalid prompt template")

        def prepare_prompt(example):
            if len(columns) > 0:
                output_text = prompt_template
                for c in columns:
                    output_text = output_text.replace('[{}]'.format(c), example[c])
                example["text"] = output_text
            return example
        train_dataset = train_dataset.map(prepare_prompt)
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(f"Sample {index} after adding text to dataset: {train_dataset[index]['text']}.")

    eval_dataset = None

    if script_args.dataset_split < 1:
        logger.info("splitting dataset")
        dataset = train_dataset.train_test_split(test_size=script_args.dataset_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    if script_args.max_train_samples > 0:
        max_train_samples = min(
            len(train_dataset),
            script_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        #
        # train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if eval_dataset and script_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Discover if we have any checkpoints to resume from.
    if script_args.resume:
        try:
            output_dir_list = os.listdir(script_args.output_dir)
            checkpoints = sorted(
                output_dir_list,
                key=lambda x: int(x.split("checkpoint-")[1]) if len(x.split("checkpoint-")) > 1 else 0,
                reverse=True)
            if len(checkpoints) > 0:
                last_checkpoint = checkpoints[0]
            else:
                logger.info("no checkpoint not found. training will start from step 0")
        except FileNotFoundError:
            logger.info("failed to find last_checkpoint: output directory does not exist")
            last_checkpoint = None
        except Exception as e:
            logger.info(f"failed to find last_checkpoint: {str(e)}")
            last_checkpoint = None
            raise Exception("failed to check if last checkpoint exists")
    else:
        last_checkpoint = None

    logger.info(f"LAST CHECKPOINT: {last_checkpoint}")

    # Weights & Biases integration
    if script_args.wandb_project and os.environ.get('WANDB_API_KEY'):
        script_args.log_with = 'wandb'
        initialize_wandb(script_args, last_checkpoint)
    else:
        script_args.log_with = None
        logger.warning("WANDB: WANDB_API_KEY not found, disabling wandb.")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        trust_remote_code=script_args.trust_remote_code,
        token=script_args.use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample, cutoff_len=512, add_eos_token=True):
        if script_args.dataset_text_field:
            prompt = sample[script_args.dataset_text_field]
        else:
            prompt = sample['text']
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result

    tokenized_train_dataset = train_dataset.map(tokenize)
    tokenized_eval_dataset = eval_dataset.map(tokenize) if eval_dataset else eval_dataset

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        log_level=script_args.log_level,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        run_name=script_args.run_name,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        gradient_checkpointing=script_args.gradient_checkpointing,
        auto_find_batch_size=script_args.auto_find_batch_size,
        # TODO: uncomment that on the next release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            lora_dropout=0.05,  # TODO: This should be available in the UI.
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    logger.info(f"initiating trainer. gpu memory free-  {gpu_memory()}")
    logger.info(f"device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own
        # DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=tokenized_train_dataset,
        dataset_text_field=script_args.dataset_text_field if script_args.dataset_text_field else 'text',
        eval_dataset=tokenized_eval_dataset,
        peft_config=peft_config,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    # Silence the warnings. Re-enable for inference!
    model.config.use_cache = False

    if last_checkpoint is not None:
        train_result = trainer.train(str(os.path.join(script_args.output_dir, last_checkpoint)))
    else:
        train_result = trainer.train()

    # Step 6: Save the model
    final_path = os.path.join(script_args.output_dir, "final")
    trainer.save_model(final_path)
    metrics = train_result.metrics
    max_train_samples = (
        script_args.max_train_samples if script_args.max_train_samples > 0 else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset:
        metrics = trainer.evaluate()
        max_eval_samples = script_args.max_eval_samples if script_args.max_eval_samples > 0 else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info(f"eval metrics {metrics}")

    if not push_model(script_args.output_dir, metrics):
        raise Exception("failed to push model")


if __name__ == "__main__":
    main()
