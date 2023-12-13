import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import transformers
import wandb
from datasets import load_dataset
from e2enetworks.cloud import tir
from e2enetworks.cloud.tir.minio_service import MinioService
from peft import LoraConfig
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          HfArgumentParser, TrainingArguments)
from trl import SFTTrainer, is_xpu_available

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
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
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
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
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
    run_name: Optional[str] = field(default=None, metadata={"help": "run name for wandb"})
    auto_find_batch_size: Optional[str] = field(default=False, metadata={"help": "Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (pip install accelerate)"})
    wandb_key: Optional[str] = field(default=None, metadata={"help": "wandb key"})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "wandb project"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def download_dataset(script_args) -> str:
    try:
        dataset_download_path = 'home/jovyan/custom_dataset/'
        minio_service = MinioService(access_key=script_args.dataset_accesskey,
                                     secret_key=script_args.dataset_secretkey)
        minio_service.download_directory_recursive(bucket_name=script_args.dataset_bucket,
                                                   local_path=dataset_download_path,
                                                   prefix=script_args.dataset_path)
        logger.info("Dataset download success")
        return dataset_download_path
    except Exception as e:
        logger.error(e)
        raise Exception(f"dataset_error -> {e}")


def retry_push_model(func_object):
    def wrapper(*args, **kwargs):
        for i in range(3):
            try:
                func_object(*args, **kwargs)
                break
            except Exception as e:
                logger.error(f"ERROR_DURING_PUSH_MODEL | {e}")
                time.sleep(10)
                continue
    return wrapper


@retry_push_model
def push_model(model_path: str, info: dict = {}):
    model_repo_client = tir.Models()
    job_id = os.getenv("E2E_TIR_FINETUNE_JOB_ID")
    model_repo = model_repo_client.create(f"llama2-custom-{uuid4()}", model_type="custom", job_id=job_id, score=info)
    model_id = model_repo.id
    model_repo_client.push_model(model_path=model_path, prefix='', model_id=model_id)


def main():

    parser = HfArgumentParser(ScriptArguments)
    output = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    script_args = output[0]
    print("script_args:", output)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    logger.info(f"Script parameters {script_args}")

    # initiate tir
    tir.init()

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        trust_remote_code=script_args.trust_remote_code,
        use_auth_token=script_args.use_auth_token,
    )

    # download dataset
    dataset_path = download_dataset(script_args) if script_args.dataset_type == "eos-bucket" else script_args.dataset_name

    # Step 2: Load the dataset
    train_dataset = load_dataset(dataset_path, split="train")
    eval_dataset = None

    # todo - replace with dataset_split
    # print("dataset_split:", script_args.dataset_split)
    if script_args.dataset_split < 1:
        dataset = train_dataset.train_test_split(test_size=script_args.dataset_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    if script_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), script_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        #
        # train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if eval_dataset and script_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

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
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        gradient_checkpointing=script_args.gradient_checkpointing,
        run_name=script_args.run_name,
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
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_dataset,
        dataset_text_field=script_args.dataset_text_field,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if script_args.wandb_key:
        wandb.init(name=script_args.run_name,project=script_args.wandb_project)

    train_result = trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)
    metrics = train_result.metrics

    max_train_samples = (
        script_args.max_train_samples if script_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset:
        metrics = trainer.evaluate()
        max_eval_samples = script_args.max_eval_samples if script_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print(metrics)
    push_model(script_args.output_dir, metrics)


if __name__ == "__main__":
    main()
