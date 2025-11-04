#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into the base model and save a standalone Hugging Face model.

Usage:
  python3 merge_lora_adapter.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-dir /path/to/save_dir/global_step_XXXX \
    --output-dir /path/to/merged_model \
    [--trust-remote-code]

Notes:
- --adapter-dir should point to a checkpoint folder produced by fsdp_sft_trainer when LoRA was enabled.
- The script loads the base model in 16-bit bfloat by default and merges PEFT weights into it.
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="HF name or local path of the base model")
    p.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter checkpoint directory (global_step_*)")
    p.add_argument("--output-dir", required=True, help="Where to save the merged full model")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass to trust remote code when loading models")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    print(f"Loading LoRA adapter from: {args.adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging LoRA weights into base...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
