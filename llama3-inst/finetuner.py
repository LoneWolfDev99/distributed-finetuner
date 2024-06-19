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
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments)
from trl import SFTTrainer

from helpers import (LOCAL_MODEL_PATH, ExporterCallback, decode_base64,
                     download_dataset, download_folder_from_repo,
                     get_dataset_format, gpu_memory, initialize_wandb,
                     load_custom_dataset, make_finetuning_metric_json,
                     push_model)

logger = logging.getLogger(__name__)

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    output_dir: Optional[str] = field(default=None, metadata={"help": "Out directory to store model"})

    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the model name"})
    source_model_repo_id: Optional[int] = field(default=0, metadata={"help": "source_model_repo_id"})
    source_model_path: Optional[str] = field(default="", metadata={"help": "source_model_path"})
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
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb/tensorboard/etc' to log"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "Lora attention dimension (the “rank”)"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "The alpha parameter for Lora scaling"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "The dropout probability for Lora layers"})
    lora_bias: Optional[str] = field(default="none", metadata={"help": "the corresponding biases will be updated during training"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "This flag is used to enable 4-bit quantization"})
    bnb_4bit_compute_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "This sets the computational type"})
    bnb_4bit_quant_type: Optional[str] = field(default="fp4", metadata={"help": "This sets the quantization data type in the bnb.nn.Linear4Bit layers"})
    bnb_4bit_use_double_quant: Optional[bool] = field(default=False, metadata={"help": "Quantization constants from the first quantization are quantized again"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
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

    if eval_dataset and script_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Discover if we have any checkpoints to resume from.
    if script_args.resume and os.path.exists(script_args.output_dir):
        try:
            output_dir_list = os.listdir(script_args.output_dir)
            checkpoints_dir_list = [directory for directory in output_dir_list if ('checkpoint-' in directory)]
            checkpoints = sorted(checkpoints_dir_list, key=lambda x: int(x.split("checkpoint-")[1]) if len(x.split("checkpoint-")) > 1 else 0, reverse=True)
            if len(checkpoints) > 0:
                last_checkpoint = checkpoints[0]
            else:
                last_checkpoint = None
                logger.info("no checkpoint not found. training will start from step 0")
        except Exception as e:
            logger.error(f"failed to find last_checkpoint: {str(e)}")
            last_checkpoint = None
            raise Exception("failed to check if last checkpoint exists")
    else:
        last_checkpoint = None

    logger.info(f"LAST CHECKPOINT: {last_checkpoint}")

    # Weights & Biases integration
    if script_args.wandb_project and os.environ.get('WANDB_API_KEY'):
        script_args.log_with = ["tensorboard", "wandb"]
        initialize_wandb(script_args, last_checkpoint)
    else:
        script_args.log_with = ["tensorboard"]
        logger.warning("WANDB: WANDB_API_KEY not found, disabling wandb.")

    # Bitsandbytes configuration
    if script_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=script_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=script_args.bnb_4bit_compute_dtype,
        )
        logger.info(
            f"\nBitsandbytes quantization is enabled with the following configuration:\n"
            f"  -- load_in_4bit: {script_args.load_in_4bit}\n"
            f"  -- bnb_4bit_compute_dtype: {script_args.bnb_4bit_compute_dtype}\n"
            f"  -- bnb_4bit_quant_type: {script_args.bnb_4bit_quant_type}\n"
            f"  -- bnb_4bit_use_double_quant: {script_args.bnb_4bit_use_double_quant}\n"
        )
    else:
        bnb_config = None
        logger.info("Bitsandbytes quantization is disabled.")

    # Loading Model and Tokenizer
    if script_args.source_model_repo_id:
        download_folder_from_repo(script_args.source_model_repo_id, script_args.source_model_path)
        download_folder_from_repo(script_args.source_model_repo_id, 'base_model/')
        base_model_path = f"{LOCAL_MODEL_PATH}base_model/"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        logger.info(f"Loaded base model : {base_model}")
        model = PeftModel.from_pretrained(base_model, f"{LOCAL_MODEL_PATH}{script_args.source_model_path}")
        model = model.merge_and_unload()
        logger.info(f"Loaded merged model : {model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                                  padding_side="right")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=script_args.trust_remote_code,
            token=script_args.use_auth_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name,
            padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

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
        logging_dir=f"{script_args.output_dir}tensorboard_logs/"  # [TensorBoard] log directory
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
            bias=script_args.lora_bias,
            lora_dropout=script_args.lora_dropout,
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
        peft_config=peft_config,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ExporterCallback]
    )

    if not os.path.exists(str(os.path.join(script_args.output_dir, 'base_model/'))):
        model.save_pretrained(str(os.path.join(script_args.output_dir, 'base_model/')))
        tokenizer.save_pretrained(str(os.path.join(script_args.output_dir, 'base_model/')))
        logger.info("Saved initial model to base_model/")

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

    make_finetuning_metric_json(script_args.output_dir)
    push_model(script_args.output_dir, metrics)


if __name__ == "__main__":
    main()
