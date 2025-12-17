# train.py
import os
import argparse
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig

class DynamicSpeculativeDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=True,
            trust_remote_code=True
        ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        self.iter = iter(self.dataset)

    def __len__(self):
        return int(1e12)

    def __getitem__(self, idx):
        while True:
            try:
                example = next(self.iter)
                if "conversations" in example:
                    convs = example["conversations"]
                    if len(convs) < 2:
                        continue
                    user_msg = convs[0]
                    asst_msg = convs[1]
                    if not (user_msg.get("from", "").lower() in ("human", "user") and
                            asst_msg.get("from", "").lower() in ("gpt", "assistant")):
                        continue
                    user_content = user_msg["value"].strip()
                    asst_content = asst_msg["value"].strip()
                    if not user_content or not asst_content:
                        continue
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": asst_content}
                    ]
                else:
                    continue

                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if not text.strip():
                    continue

                tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(tokens) < self.args.spec_depth + 10:
                    continue
                if len(tokens) > 4096:
                    continue

                max_start = len(tokens) - self.args.spec_depth
                start = random.randint(0, max_start)
                context = tokens[:start]
                future = tokens[start:start + self.args.spec_depth]

                if len(context) == 0 or len(future) != self.args.spec_depth:
                    continue

                return {
                    "context_ids": torch.tensor(context, dtype=torch.long),
                    "future_ids": torch.tensor(future, dtype=torch.long),
                }

            except StopIteration:
                self.iter = iter(self.dataset.shuffle(buffer_size=self.args.shuffle_buffer))
            except Exception:
                print("DataLoader Exception", Exception)
                continue

def collate_fn(batch, tokenizer):
    context_ids = [item["context_ids"] for item in batch]
    future_ids = [item["future_ids"] for item in batch]
    context_ids = torch.nn.utils.rnn.pad_sequence(
        context_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    future_ids = torch.stack(future_ids, dim=0)
    return {"context_ids": context_ids, "future_ids": future_ids}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--spec_depth", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--shuffle_buffer", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./llada/llada-spec")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        writer = None

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Base Model (Teacher, frozen) ===
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(local_rank).eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    # === Checkpoint path ===
    latest_ckpt = os.path.join(args.output_dir, "latest")
    step_offset = 0

    if os.path.exists(latest_ckpt) and os.path.islink(latest_ckpt):
        # Resolve checkpoint path
        ckpt_path = os.path.join(args.output_dir, os.readlink(latest_ckpt))
        if os.path.isdir(ckpt_path):
            if rank == 0:
                print(f"Resuming from checkpoint: {ckpt_path}")

            llada_model = LLaDAModelLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(local_rank)

            # Load optimizer, scheduler, step
            optimizer_state_path = os.path.join(ckpt_path, "optimizer.pt")
            scheduler_state_path = os.path.join(ckpt_path, "scheduler.pt")
            trainer_state_path = os.path.join(ckpt_path, "trainer_state.pt")

            if os.path.exists(trainer_state_path):
                trainer_state = torch.load(trainer_state_path, map_location="cpu")
                step_offset = trainer_state["step"]

            # Create optimizer & scheduler (needed before loading state)
            optimizer = torch.optim.AdamW(
                llada_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

            if os.path.exists(optimizer_state_path):
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=f"cuda:{local_rank}"))
            if os.path.exists(scheduler_state_path):
                scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))

            if rank == 0:
                print(f"Resumed from step {step_offset}")

        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        llada_config = LLaDAConfig.from_pretrained(
            'llada/config.json',  # or a local path with config.json
            trust_remote_code=True
        )
        # Override spec depth if needed (optional)
        llada_config.spec_depth = args.spec_depth

        llada_model = LLaDAModelLM(llada_config, init_params=True,)
        llada_model = llada_model.to(local_rank).to(torch.bfloat16)

        # === Optimizer ===
        optimizer = torch.optim.AdamW(llada_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
        num_warmup_steps = int(args.num_steps*0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=args.num_steps,
        )

    if rank == 0:
        total = sum(p.numel() for p in llada_model.parameters())
        trainable = sum(p.numel() for p in llada_model.parameters() if p.requires_grad)
        print(f"LLaDA Total params: {total}, Trainable: {trainable}")

    llada_model = DDP(llada_model, device_ids=[local_rank], find_unused_parameters=False)

    # === Dataset & DataLoader ===
    train_dataset = DynamicSpeculativeDataset(tokenizer, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    # === Training Loop ===
    step = step_offset
    train_iter = iter(train_loader)
    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        context_ids = batch["context_ids"].to(local_rank)
        future_ids = batch["future_ids"].to(local_rank)

        # Get clean future hidden states from base_model
        with torch.no_grad():
            base_out = base_model(
                input_ids=context_ids,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = base_out.past_key_values

            # Get target hidden states (future tokens)
            base_full_out = base_model(input_ids=future_ids, past_key_values=past_key_values, output_hidden_states=True)
            future_hidden_clean = base_full_out.hidden_states[-1][:, -args.spec_depth:, :]  # [B, L, D]

        # Forward LLaDA model on context only
        input_ids = torch.zeros(future_ids.shape, dtype=torch.long, device=future_ids.device)
        llada_out = llada_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        llada_future_pred = llada_out.hidden_states[-1]  # [B, L, D]

        # Compute MSE loss in hidden space
        loss = torch.nn.functional.mse_loss(llada_future_pred, future_hidden_clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llada_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if rank == 0 and step % args.logging_steps == 0:
            print(f"Step {step}, LR: {scheduler.get_last_lr()[0]:.2e}, Loss: {loss.item():.6f}")
            writer.add_scalar("Loss", loss.item(), step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

        if rank == 0 and step > 0 and step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(save_path, exist_ok=True)
            # Save LLaDA model in HF format
            llada_model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            torch.save({"step": step}, os.path.join(save_path, "trainer_state.pt"))

            # Soft link
            latest = os.path.join(args.output_dir, "latest")
            if os.path.exists(latest) or os.path.islink(latest):
                os.unlink(latest)
            os.symlink(os.path.relpath(save_path, args.output_dir), latest)

        step += 1

    if rank == 0:
        writer.add_scalar("Loss", loss.item(), step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
        writer.close()
        final_path = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(final_path, exist_ok=True)
        llada_model.module.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        torch.save(optimizer.state_dict(), os.path.join(final_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(final_path, "scheduler.pt"))
        torch.save({"step": step}, os.path.join(final_path, "trainer_state.pt"))

        latest = os.path.join(args.output_dir, "latest")
        if os.path.exists(latest) or os.path.islink(latest):
            os.unlink(latest)
        os.symlink(os.path.relpath(final_path, args.output_dir), latest)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()