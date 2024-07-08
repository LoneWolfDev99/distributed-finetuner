#!/bin/bash

while [ $# -gt 0 ]; do
  case "$1" in
    --model_name=*)
      model_name="${1#*=}"
      ;;
    --dataset_name=*)
      dataset_name="${1#*=}"
      ;;
    --dataset_text_field=*)
      dataset_text_field="${1#*=}"
      ;;
    --dataset_split=*)
      dataset_split="${1#*=}"
      ;;
    --dataset_type=*)
      dataset_type="${1#*=}"
      ;;
    --dataset_id=*)
      dataset_id="${1#*=}"
      ;;
    --dataset_path=*)
      dataset_path="${1#*=}"
      ;;
    --output_dir=*)
      output_dir="${1#*=}"
      ;;
    --num_train_epochs=*)
      num_train_epochs="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --gradient_accumulation_steps=*)
      gradient_accumulation_steps="${1#*=}"
      ;;
    --use_peft=*)
      use_peft="${1#*=}"
      ;;
    --peft_lora_r=*)
      peft_lora_r="${1#*=}"
      ;;
    --peft_lora_alpha=*)
      peft_lora_alpha="${1#*=}"
      ;;
    --lora_dropout=*)
      lora_dropout="${1#*=}"
      ;;
    --lora_bias=*)
      lora_bias="${1#*=}"
      ;;
    --load_in_4bit=*)
      load_in_4bit="${1#*=}"
      ;;
    --bnb_4bit_compute_dtype=*)
      bnb_4bit_compute_dtype="${1#*=}"
      ;;
    --bnb_4bit_quant_type=*)
      bnb_4bit_quant_type="${1#*=}"
      ;;
    --bnb_4bit_use_double_quant=*)
      bnb_4bit_use_double_quant="${1#*=}"
      ;;
    --max_steps=*)
      max_steps="${1#*=}"
      ;;
    --save_steps=*)
      save_steps="${1#*=}"
      ;;
    --save_total_limit=*)
      save_total_limit="${1#*=}"
      ;;
    --max_train_samples=*)
      max_train_samples="${1#*=}"
      ;;
    --max_eval_samples=*)
      max_eval_samples="${1#*=}"
      ;;
    --prompt_template_base64=*)
      prompt_template_base64="${1#*=}"
      ;;
    --resume=*)
      resume="${1#*=}"
      ;;
    --wandb_project=*)
      wandb_project="${1#*=}"
      ;;
    --wandb_run_name=*)
      wandb_run_name="${1#*=}"
      ;;
    --seq_length=*)
      seq_length="${1#*=}"
      ;;
    --run_name=*)
      run_name="${1#*=}"
      ;;
    --source_model_repo_id=*)
      source_model_repo_id="${1#*=}"
      ;;
    --source_model_path=*)
      source_model_path="${1#*=}"
      ;;
    --trainer_gpus=*)
      trainer_gpus="${1#*=}"
      ;;
    *)
      echo "Invalid argument: $1" >&2
      exit 1
  esac
  shift
done

echo "***Recieved Training parameters***"
echo ""
echo "model_name: $model_name"
echo "dataset_name: $dataset_name"
echo "dataset_text_field: $dataset_text_field"
echo "dataset_split: $dataset_split"
echo "dataset_type: $dataset_type"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"
echo "num_train_epochs: $num_train_epochs"
echo "batch_size: $batch_size"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"
echo "use_peft: $use_peft"
echo "peft_lora_r: $peft_lora_r"
echo "peft_lora_alpha: $peft_lora_alpha"
echo "lora_dropout: $lora_dropout"
echo "lora_bias: $lora_bias"
echo "load_in_4bit: $load_in_4bit"
echo "bnb_4bit_compute_dtype: $bnb_4bit_compute_dtype"
echo "bnb_4bit_quant_type: $bnb_4bit_quant_type"
echo "bnb_4bit_use_double_quant: $bnb_4bit_use_double_quant"
echo "max_steps: $max_steps"
echo "save_steps: $save_steps"
echo "save_total_limit: $save_total_limit"
echo "max_train_samples: $max_train_samples"
echo "max_eval_samples: $max_eval_samples"
echo "prompt_template_base64: $prompt_template_base64"
echo "resume: $resume"
echo "wandb_project: $wandb_project"
echo "wandb_run_name: $wandb_run_name"
echo "seq_length: $seq_length"
echo "run_name: $run_name"
echo "source_model_repo_id: $source_model_repo_id"
echo "source_model_path: $source_model_path"
echo "trainer_gpus: $trainer_gpus"


export dataset_type=$dataset_type
export dataset_id=$dataset_id
export dataset_path=$dataset_path
export source_model_repo_id=$source_model_repo_id
export source_model_path=$source_model_path
python3 initializer.py || { echo "initializer failed to execute"; exit 1; }


export model_name=$model_name
torchrun --nproc_per_node=$trainer_gpus finetuner.py --model_name=$model_name --dataset_name=$dataset_name --dataset_text_field=$dataset_text_field --dataset_split=$dataset_split --dataset_type=$dataset_type --dataset_path=$dataset_path --output_dir=$output_dir --num_train_epochs=$num_train_epochs --batch_size=$batch_size --gradient_accumulation_steps=$gradient_accumulation_steps --use_peft=true --peft_lora_r=$peft_lora_r --peft_lora_alpha=$peft_lora_alpha --save_total_limit=$save_total_limit --max_train_samples=$max_train_samples --max_eval_samples=$max_eval_samples --prompt_template_base64=$prompt_template_base64 --resume=true --save_steps=$save_steps || { echo "finetuner failed to execute"; exit 1; }
