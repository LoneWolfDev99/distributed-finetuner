#!/bin/which python3
from typing import List
from torch import Tensor
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from utils import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

val = validation

parser = DashParser(description="Simple Model Evaluator")

parser.add_argument(
    "--model",
    type=str,
    help="The model to evaluate against (directory, or HuggingFace ID)",
    required=True,
)
parser.add_argument(
    "--trust-remote-code",
    action=FuzzyBoolAction,
    help="Whether to trust remote code coming with the model",
    default=False,
)
parser.add_argument("--tokenizer", type=str, help="The tokenizer to use")
parser.add_argument(
    "--eot", type=str, help="EOT token to use", default=""
)  # default is model-dependent
parser.add_argument(
    "--pad", type=str, help="Pad token to use", default=""
)  # default is model-dependent
parser.add_argument(
    "--cache", type=str, help="HuggingFace cache location", default="/tmp"
)
parser.add_argument(
    "--fp16",
    action=FuzzyBoolAction,
    help="Force evaluation in fp16",
    default=False,
)
parser.add_argument("--prompt", type=str, help="Prompt to use")
parser.add_argument(
    "--prompt-file",
    type=val.optional_extant_file,
    help="File containing prompts",
)
parser.add_argument(
    "--prompt-tokens",
    type=val.non_negative(int),
    help="Number of tokens to generate",
    default=200,
)
parser.add_argument(
    "--seed",
    # Range restrictions are imposed by numpy.random.seed
    type=val.at_most_32_bit(val.non_negative(int)),
    help="Random seed value",
    default=None,
)
parser.add_argument(
    "--prompt-samples",
    type=val.non_negative(int),
    help="Number of samples to generate",
    default=1,
)
parser.add_argument(
    "--top-k",
    type=val.non_negative(int),
    help="Top K to use for sampling",
    default=16,
)
parser.add_argument(
    "--top-p",
    type=val.at_most_1(val.non_negative(float)),
    help="Top P to use for sampling",
    default=0.95,
)
parser.add_argument(
    "--temperature",
    type=val.positive(float),
    help="Temperature to use for sampling",
    default=1.0,
)
parser.add_argument(
    "--repetition-penalty",
    type=val.positive(float),
    help="Repetition penalty to use for sampling",
    default=1.1,
)

args = parser.parse_args()
if args.tokenizer is None:
    args.tokenizer = args.model

if args.prompt and args.prompt_file:
    parser.error("Cannot specify both a prompt and a prompt file")

if not args.prompt and not args.prompt_file:
    parser.error("Please specify either a prompt or a prompt file")

if args.prompt_file:
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = (line.rstrip("\n").replace("\\n", "\n") for line in f)
            prompts = list(filter(None, prompts))
        if not prompts:
            parser.error(f"Provided prompt file was blank: {args.prompt_file}")
    except OSError:
        parser.error(
            f"Provided prompt file could not be read: {args.prompt_file}"
        )
else:
    prompts = [args.prompt.strip()]


print(MemoryUsage.now())

start = time.time()


tokens_to_add = {}
if args.eot:
    tokens_to_add["eos_token"] = args.eot
if args.pad:
    tokens_to_add["pad_token"] = args.pad

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer,
    **tokens_to_add,
    cache_dir=args.cache,
    padding_side="left",
    trust_remote_code=args.trust_remote_code,
    init_device=str(device),
)

tokens_to_add.clear()
if "eos_token" not in tokenizer.special_tokens_map:
    tokens_to_add["eos_token"] = "<|endoftext|>"
if "pad_token" not in tokenizer.special_tokens_map:
    tokens_to_add["pad_token"] = "<|endoftext|>"
if tokens_to_add:
    tokenizer.add_special_tokens(tokens_to_add)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    # Can be a HuggingFace ID or directory.
    cache_dir=args.cache,
    use_cache=True
)

with no_init():
    if args.fp16:
        model = model.half().to(device)
    else:
        model = model.to(device)

model.resize_token_embeddings(len(tokenizer))
model.eval()


duration = time.time() - start
print(f"Loaded model in {duration:.2f}s")
torch.cuda.memory.empty_cache()
print(MemoryUsage.now())


def evaluate(
    prompt,
    generate_tokens: int = 200,
    num_samples: int = 1,
    eval_model: PreTrainedModel = model,
    eval_tokenizer: PreTrainedTokenizer = tokenizer,
    top_k: int = args.top_k,
    top_p: float = args.top_p,
    temperature: float = args.temperature,
    repetition_penalty: float = args.repetition_penalty
) -> List[str]:
    """
    Evaluate the model on the given prompt, and return the output text.
    """
    output_texts: List[str] = []
    input_tokens: Tensor = (
        torch.LongTensor(eval_tokenizer.encode(prompt)).unsqueeze(0).to(device)
    )
    if eval_tokenizer.pad_token_id != eval_tokenizer.eos_token_id:
        attention_mask: Tensor = input_tokens != eval_tokenizer.pad_token_id
    else:
        # If padding is indistinguishable from end-of-sentence,
        # we can't tell which to mask, so mask nothing.
        attention_mask = torch.ones_like(input_tokens, dtype=torch.bool)
    max_length = input_tokens.shape[1] + generate_tokens

    generated_tokens = eval_model.generate(
        input_tokens,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        pad_token_id=eval_tokenizer.pad_token_id,
        num_return_sequences=num_samples,
        bad_words_ids=[[eval_tokenizer.eos_token_id]]
    )

    for token in generated_tokens:
        output_text = eval_tokenizer.decode(token, skip_special_tokens=False)
        output_texts.append(output_text)

    return output_texts


if args.seed is not None:
    torch.cuda.manual_seed(args.seed)

for prompt in prompts:
    print("=============================")
    print("PROMPT:", prompt)
    print("UTILIZATION:", MemoryUsage.now())
    for response in evaluate(prompt, args.prompt_tokens, args.prompt_samples):
        print("-----------------------------")
        print("RESPONSE:", response)