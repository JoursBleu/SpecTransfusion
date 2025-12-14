# val.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import random

class SpecValidationDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.spec_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<|spec_{i}|>" for i in range(1, args.spec_depth + 1)]
        )
        self.dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=True,
            trust_remote_code=True
        ).shuffle(seed=42, buffer_size=10000)
        self.iter = iter(self.dataset)

    def __len__(self):
        return 1000  # evaluate on 1000 samples

    def __getitem__(self, idx):
        while True:
            try:
                example = next(self.iter)
                text = example.get("text", "").strip()
                if not text:
                    continue
                tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                min_len = self.args.max_context + self.args.spec_depth + 10
                if len(tokens) < min_len:
                    continue
                context = tokens[:self.args.max_context]
                input_ids = context + self.spec_token_ids
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long)
                }
            except StopIteration:
                self.iter = iter(self.dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to draft model checkpoint (e.g., ./new-model/checkpoint_latest). "
                             "If not provided, uses --output_dir/checkpoint_latest")
    parser.add_argument("--output_dir", type=str, default="./qwen3-spec")
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_context", type=int, default=2048)
    parser.add_argument("--spec_depth", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

@torch.no_grad()
def speculative_decode(base_model, draft_model, input_ids, spec_depth, tokenizer, device):
    """
    Simulate speculative decoding with ground-truth future for acceptance rate calculation.
    """
    input_ids = input_ids.to(device)  # [C]

    # Step 1: Base model predicts K tokens
    base_tokens = base_model.generate(
        input_ids[:, :-spec_depth],
        do_sample=False,
        max_new_tokens=spec_depth,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )  # [1, C + K]

    # Step 2: Target model predicts K tokens
    target_tokens = draft_model.generate(
        input_ids[:, :-spec_depth],
        do_sample=False,
        max_new_tokens=spec_depth,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )  # [1, C + K]

    # Step 2: Draft model predicts K tokens
    logits = draft_model(input_ids=input_ids).logits
    spec_tokens = logits.argmax(dim=-1)

    aligned_base = (target_tokens[:, -spec_depth:] == base_tokens[:, -spec_depth:]).sum().item()
    accepted = (target_tokens[:, -spec_depth:] == spec_tokens[:, -spec_depth-1:-1]).sum().item()
    print("base", base_tokens[:, -spec_depth:])
    print("base text:", tokenizer.decode(base_tokens[0, -spec_depth:]))
    print("target", target_tokens[:, -spec_depth:])
    print("target text:", tokenizer.decode(target_tokens[0, -spec_depth:]))
    print("draft", spec_tokens[:, -spec_depth-1:])
    print("draft text:", tokenizer.decode(spec_tokens[0, -spec_depth-1:]))
    
    print("aligned_base", aligned_base)
    print("accepted", accepted)

    return accepted, aligned_base

def main():
    args = parse_args()
    device = args.device

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_path, trust_remote_code=True)

    # === Load base model (frozen) ===
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # === Load draft model ===
    draft_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.use_lora:
        draft_model = PeftModel.from_pretrained(draft_base, args.draft_model_path)
    else:
        state_dict = torch.load(os.path.join(args.draft_model_path, "pytorch_model.bin"), map_location="cpu")
        draft_base.load_state_dict(state_dict)
        draft_model = draft_base
    draft_model = draft_model

    # Load base model's trainable params (e.g., spec_embed_tokens)
    print("load trainable params")
    trainable_path = os.path.join(args.draft_model_path, "trainable_base_params.bin")
    if os.path.exists(trainable_path):
        trainable_state = torch.load(trainable_path, map_location="cpu")
        draft_model.load_state_dict(trainable_state, strict=False)  # only load matching keys
    draft_model.to(device).eval()


    # === Load dataset ===
    dataset = SpecValidationDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_accepted = 0
    total_aligned = 0
    total_drafts = 0
    count = 0

    print(f"Evaluating {args.num_samples} samples...")
    for batch in dataloader:
        if count >= args.num_samples:
            break

        accepted, aligned_base = speculative_decode(
            base_model, draft_model, batch["input_ids"], args.spec_depth, tokenizer, device
        )
        total_accepted += accepted
        total_drafts += args.spec_depth
        total_aligned += aligned_base
        count += 1

        print(f"Processed {count} samples, current acceptance rate: {total_accepted / total_drafts:.2%}, current aligned rate: {total_aligned / total_drafts:.2%}")

    acceptance_rate = total_accepted / total_drafts
    avg_accepted_per_step = total_accepted / count

    print("\n" + "="*50)
    print(f"Speculative Decoding Evaluation Results")
    print(f"Spec depth: {args.spec_depth}")
    print(f"Samples: {count}")
    print(f"Average accepted tokens per draft: {avg_accepted_per_step:.2f} / {args.spec_depth}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    print(f"Average aligned tokens: {total_aligned / count:.3}")
    print("="*50)

if __name__ == "__main__":
    main()