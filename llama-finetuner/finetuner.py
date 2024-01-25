import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
import wandb
from datasets import load_dataset
from e2enetworks.cloud import tir
from peft import LoraConfig
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          HfArgumentParser, TrainingArguments)
from trl import SFTTrainer, is_xpu_available

from helpers import (decode_base64, download_dataset, get_dataset_format,
                     gpu_memory, load_custom_dataset, push_model)

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
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=500, metadata={"help": "Number of updates steps before two checkpoint saves"}
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
        train_dataset = load_custom_dataset(dataset_path)
    else:
        logger.info(f"loading dataset {script_args.dataset_name} from huggingface")
        train_dataset = load_dataset(script_args.dataset_name, split="train")

    if script_args.prompt_template_base64:
        prompt_template = decode_base64(script_args.prompt_template_base64)
        logger.info(f"adding training_text column to dataset {prompt_template}")
        columns = re.findall(r'\[(.*?)\]', prompt_template)
        logger.info(f"found {len(columns)} columns in prompt template. replacing them")
        if len(columns) == 0:
            raise Exception("invalid prompt template")
        def prepare_prompt(example):
            if len(columns) > 0:
                output_text = prompt_template
                for c in columns:
                    output_text = output_text.replace('[{}]'.format(c), example[c])
                example["training_text"] = output_text
            return example
        train_dataset = train_dataset.map(prepare_prompt)
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(f"Sample {index} after adding training_text to dataset: {train_dataset[index]['training_text']}.")

    eval_dataset = None

    if script_args.dataset_split < 1:
        logger.info("splitting dataset")
        dataset = train_dataset.train_test_split(test_size=script_args.dataset_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    if script_args.max_train_samples > 0:
        max_train_samples = min(len(train_dataset), script_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        #
        # train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if eval_dataset and script_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Discover if we have any checkpoints to resume from.
    if script_args.resume and os.path.exists(script_args.output_dir):
        try:
            output_dir_list = os.listdir(script_args.output_dir)
            checkpoints = sorted(output_dir_list, key=lambda x: int(x.split("checkpoint-")[1]) if len(x.split("checkpoint-")) > 1 else 0, reverse=True)
            if len(checkpoints) > 0:
                last_checkpoint = checkpoints[0]
            else:
                logger.info("no checkpoint not found. training will start from step 0")
        except Exception as e:
            logger.error(f"failed to find last_checkpoint: {str(e)}")
            last_checkpoint = None
            raise Exception("failed to check if last checkpoint exists")
    else:
        last_checkpoint = None

    logger.info(f"LAST CHECKPOINT: {last_checkpoint}")

    if not script_args.wandb_key:
        logger.warning("WANDB_API_KEY: WANDB_API_KEY not found, disabling wandb.")
        os.environ["WANDB_DISABLED"] = "True"

    report_to = script_args.log_with

    if not report_to and script_args.wandb_key:
        # todo: check if main process when distributed is implemented
        report_to = ["wandb"]
        if last_checkpoint is not None:
            wandb_api = wandb.Api(overrides={"project": script_args.wandb_project})
            for run in wandb_api.runs(path=script_args.wandb_project):
                logger.info(f"PRIOR RUN: {run} {run.name} {run.id} {run.state}")
                if run.state in ["crashed", "failed"] and run.name == script_args.run_name:
                    logger.info(f"CHECKPOINT: Resuming {run.id}")
                    run = wandb.init(
                        id=run.id,
                        project=script_args.wandb_project,
                        resume="must",
                        name=run.name,
                    )
                    break
        else:
            wandb.init(name=script_args.run_name,project=script_args.wandb_project)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        trust_remote_code=script_args.trust_remote_code,
        token=script_args.use_auth_token,
    )

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

    logger.info(f"initiating trainer. gpu memory free-  {gpu_memory()}")

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_dataset,
        dataset_text_field=script_args.dataset_text_field if script_args.dataset_text_field else 'training_text',
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

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

    push_model(script_args.output_dir, metrics)


if __name__ == "__main__":
    main()