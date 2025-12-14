# train.py
import os
import argparse
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import re

class DynamicSpeculativeDataset(Dataset):
    def __init__(self, tokenizer, args, skip_steps=0):
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
        ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        self.iter = iter(self.dataset)

        # # Skip examples if resuming (approximate, not exact due to streaming)
        # if skip_steps > 0:
            # total_skip = skip_steps * args.batch_size
            # for _ in range(total_skip):
                # try:
                    # next(self.iter)
                # except StopIteration:
                    # self.iter = iter(self.dataset.shuffle(buffer_size=args.shuffle_buffer))

    def __len__(self):
        return int(1e12)  # infinite

    def __getitem__(self, idx):
        while True:
            try:
                example = next(self.iter)
                if "conversations" in example:
                    conversations = example["conversations"]
                    # 至少需要一轮问答（user + assistant）
                    if len(conversations) < 2:
                        continue
                    # 只取第一轮
                    first_turn = conversations[0]
                    second_turn = conversations[1]
                    
                    # 验证角色顺序：user -> assistant
                    if not (first_turn.get("from", "").lower() in ("human", "user") and
                            second_turn.get("from", "").lower() in ("gpt", "assistant")):
                        continue  # skip malformed
                    
                    messages = [
                        {"role": "user", "content": first_turn["value"].strip()},
                        {"role": "assistant", "content": second_turn["value"].strip()}
                    ]
                    messages_user = [
                        {"role": "user", "content": first_turn["value"].strip()},
                    ]
                    
                    if not messages[0]["content"] or not messages[1]["content"]:
                        continue
                    
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    text_user = self.tokenizer.apply_chat_template(
                        messages_user,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    continue

                if not text.strip():
                    continue

                tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                tokens_user = self.tokenizer(text_user, add_special_tokens=False)["input_ids"][:-3]
                len_user = len(tokens_user)
                if len(tokens) - len_user < self.args.spec_depth:
                    continue
                start = random.randint(0, len(tokens) - len_user - self.args.spec_depth) + len_user
                context = tokens[:start]
                input_ids = context + self.spec_token_ids
                base_input_ids = tokens[:start+self.args.spec_depth]
                if len(input_ids) > 2048:
                    continue
                return {
                    "base_input_ids": torch.tensor(base_input_ids, dtype=torch.long),
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                }

            except StopIteration:
                self.iter = iter(self.dataset.shuffle(buffer_size=self.args.shuffle_buffer))
            except Exception:
                continue


def collate_fn(batch, tokenizer):
    base_input_ids = [item["base_input_ids"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    base_input_ids = torch.nn.utils.rnn.pad_sequence(
        base_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return {"base_input_ids": base_input_ids, "input_ids": input_ids}

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--spec_depth", type=int, default=4)
    # Data
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_context", type=int, default=2048)
    parser.add_argument("--shuffle_buffer", type=int, default=500000)
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./qwen3-spec")
    parser.add_argument("--resume_from_checkpoint", type=str, default="checkpoint_latest",
                        help="Path to checkpoint directory to resume from (e.g., ./qwen3-spec/step_2000). "
                             "If not provided, will auto-resume from latest checkpoint in output_dir.")
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)  # for torchrun
    parser.add_argument("--warmup_vloss_weight", type=float, default=None, help="Weight for hidden state loss")
    parser.add_argument("--warmup_ploss_weight", type=float, default=None, help="Weight for probability loss")
    parser.add_argument("--vloss_weight", type=float, default=0.01, help="Weight for hidden state loss")
    parser.add_argument("--ploss_weight", type=float, default=1.0, help="Weight for probability loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature")
    parser.add_argument("--lora_layer_ratio", type=float, default=0.25,
                        help="Ratio of layers (from the end) to apply LoRA. "
                             "e.g., 0.5 for last 50%% of layers. Default: 0.5")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[-1, -2, -3, -4],
                        help="Which hidden layers to use for hidden state loss (default: last 4 layers)")
    return parser.parse_args()

def compute_loss(targets, target_logits, predicts, predict_logits, hidden_layers, spec_depth, temperature=1.0):
    """
    Compute distillation loss for speculative decoding draft model.
    
    Args:
        targets (tuple of torch.Tensor): Teacher hidden states from all layers.
            Each tensor has shape [B, L, D].
        target_logits (torch.Tensor): Teacher logits, shape [B, L, V].
        predicts (tuple of torch.Tensor): Student hidden states from all layers.
            Each tensor has shape [B, L, D].
        predict_logits (torch.Tensor): Student logits, shape [B, L, V].
        hidden_layers (list of int): Indices of layers to use for hidden state alignment
            (e.g., [-1, -2] for last two layers).
        temperature (float): Temperature for distillation. Default: 1.0.
    
    Returns:
        vloss (torch.Tensor): Scalar, average SmoothL1 loss over selected hidden layers.
        ploss (torch.Tensor): Scalar, KL divergence loss over full sequence logits.
    """
    # 1. Probability alignment via KL divergence
    # target_probs = torch.softmax(target_logits / temperature, dim=-1)
    # predict_logprobs = torch.log_softmax(predict_logits / temperature, dim=-1)
    # ploss = F.kl_div(
        # predict_logprobs,
        # target_probs,
        # reduction='sum'
    # ) * (temperature ** 2) / (target_probs.shape[0] * target_probs.shape[1])
    ploss = F.smooth_l1_loss(predict_logits, target_logits, reduction='mean')

    # 2. Hidden state alignment via SmoothL1Loss
    selected_targets = [targets[i][:,-spec_depth:,:] for i in hidden_layers]
    selected_predicts = [predicts[i][:,-spec_depth:,:] for i in hidden_layers]
    vloss = 0.0
    for predict,target in zip(selected_predicts,selected_targets):
        vloss += F.smooth_l1_loss(predict, target, reduction='mean')
    vloss = vloss / len(selected_predicts)  # average over layers

    return vloss, ploss

def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    resume_ckpt = os.path.join(args.output_dir, args.resume_from_checkpoint)
    if os.path.isdir(resume_ckpt):
        tokenizer = AutoTokenizer.from_pretrained(resume_ckpt, trust_remote_code=True)
        print("load base")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()
        print("load st")
        st_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        print("use lora")
        if args.use_lora:
            model = PeftModel.from_pretrained(st_model, resume_ckpt, is_trainable=True)
        else:
            # Full fine-tune: not recommended, but for completeness
            state_dict = torch.load(os.path.join(resume_ckpt, "pytorch_model.bin"), map_location="cpu")
            st_model.load_state_dict(state_dict)
            model = st_model

        # Load base model's trainable params (e.g., spec_embed_tokens)
        print("load trainable params")
        trainable_path = os.path.join(resume_ckpt, "trainable_base_params.bin")
        if os.path.exists(trainable_path):
            trainable_state = torch.load(trainable_path, map_location="cpu")
            model.load_state_dict(trainable_state, strict=False)  # only load matching keys

        # Load training state
        trainer_state_path = os.path.join(resume_ckpt, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, map_location="cpu")
            step_offset = trainer_state["step"]
            if rank == 0:
                print(f"Resuming from step {step_offset}")
        else:
            step_offset = 0
            if rank == 0:
                print("Warning: checkpoint found but no trainer_state.pt, starting from step 0")
    else:
        # === Tokenizer & Model ===
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()
        print("load st")
        st_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        # === Add special tokens ===
        print("=== Add special tokens ===")
        spec_tokens = [f"<|spec_{i}|>" for i in range(1, args.spec_depth + 1)]
        num_added = tokenizer.add_tokens(spec_tokens, special_tokens=True)
        # model.resize_token_embeddings(len(tokenizer))
        spec_embed_tokens = st_model.model.embed_tokens.weight[tokenizer.eos_token_id].unsqueeze(0).repeat(args.spec_depth, 1)
        st_model.model.spec_embed_tokens = torch.nn.Parameter(spec_embed_tokens.clone())

        # === LoRA ===
        print("=== LoRA ===")
        if args.use_lora:
            # 获取总层数
            num_layers = st_model.config.num_hidden_layers

            # 计算 LoRA 作用的层数
            num_lora_layers = max(1, int(num_layers * args.lora_layer_ratio + 0.5))  # 四舍五入
            start_layer = num_layers - num_lora_layers
            lora_layer_indices = list(range(start_layer, num_layers))

            target_modules = []
            for i in lora_layer_indices:
                target_modules.extend([
                    f"model.layers.{i}.self_attn.q_proj",
                    f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj",
                    f"model.layers.{i}.self_attn.o_proj",
                    f"model.layers.{i}.mlp.gate_proj",
                    f"model.layers.{i}.mlp.up_proj",
                    f"model.layers.{i}.mlp.down_proj",
                ])

            print(f"Applying LoRA to layers: {lora_layer_indices} (total {len(lora_layer_indices)} layers)")

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(st_model, lora_config)
        step_offset = 0

    assert(base_model.config.spec_token == args.spec_depth)
    model.train()
    model.model.model.spec_embed_tokens.requires_grad_(True)

    if rank == 0:
        model.print_trainable_parameters()
    model = model.to(local_rank)
    base_model = base_model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # === Dataset & DataLoader ===
    print("=== Dataset & DataLoader ===")
    train_dataset = DynamicSpeculativeDataset(tokenizer, args, skip_steps=step_offset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    # === Optimizer ===
    print("=== Optimizer ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    num_warmup_steps = int(args.num_steps*0.03)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_steps,
    )

    # If resuming, load optimizer state
    if args.resume_from_checkpoint:
        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=f"cuda:{local_rank}"))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(local_rank)
        # Load scheduler
        sched_path = os.path.join(resume_ckpt, "scheduler.pt")
        if os.path.exists(sched_path):
            sched_state_dict = torch.load(sched_path)  # No map_location needed!
            scheduler.load_state_dict(sched_state_dict)
        else:
            print(f"Warning: scheduler.pt not found in {resume_ckpt}. Scheduler starts from scratch.")

    # === Training Loop ===
    print("=== Training Loop ===")
    step = step_offset
    train_iter = iter(train_loader)
    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        base_input_ids = batch["base_input_ids"].to(local_rank)
        input_ids = batch["input_ids"].to(local_rank)
        with torch.no_grad():
            base_outputs = base_model(
                input_ids=base_input_ids,
                output_hidden_states=True,
                # use_cache=True
            )
            # base_logits = base_outputs.logits[:, -args.spec_depth:]  # [B, L, V]
            # base_hidden = base_outputs.hidden_states[-1][:, -args.spec_depth:]  # [B, L, D]，或选某一层
            # past_key_values = base_outputs.past_key_values
            # past_key_values.crop(base_input_ids.shape[1] - args.spec_depth)  # [B, L, D]，或选某一层
            base_logits = base_outputs.logits  # [B, L, V]
            base_hidden = base_outputs.hidden_states  # [B, L, D]，或选某一层

        optimizer.zero_grad()
        st_outputs = model(
            # input_ids=input_ids[:, -args.spec_depth:],
            input_ids=input_ids,
            output_hidden_states=True,
            # past_key_values=past_key_values,
            # use_cache=True
        )
        st_logits = st_outputs.logits  # [B, L, V]
        st_hidden = st_outputs.hidden_states
        vloss, ploss = compute_loss(base_hidden, base_logits, st_hidden, st_logits, args.hidden_layers, args.spec_depth, temperature=args.temperature)
        vloss_weight = args.vloss_weight
        ploss_weight = args.ploss_weight
        if step < num_warmup_steps:
            if args.warmup_vloss_weight is not None:
                vloss_weight = args.warmup_vloss_weight
            if args.warmup_ploss_weight is not None:
                ploss_weight = args.warmup_ploss_weight
        loss = vloss * vloss_weight + ploss * ploss_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if rank == 0 and step % args.logging_steps == 0:
            print(f"Step {step}, LR: {scheduler.get_last_lr()[0]:.2e}, Loss: {loss.item():.4f}, VLoss: {vloss.item():.4f}, PLoss: {ploss.item():.4f}")
            writer.add_scalar("Loss/Total", loss.item(), step)
            writer.add_scalar("Loss/VLoss", vloss.item(), step)
            writer.add_scalar("Loss/PLoss", ploss.item(), step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

        if rank == 0 and step > 0 and step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(save_path, exist_ok=True)

            # 1. 保存 LoRA 适配器（标准方式）
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # 2. 保存 base model 中可训练的参数（如 spec_embed_tokens）
            trainable_state_dict = {}
            for name, param in model.module.named_parameters():
                if param.requires_grad and "lora" not in name:
                    # 排除 LoRA（已由 save_pretrained 处理），只存 base 中可训练部分
                    trainable_state_dict[name] = param.data.cpu()

            if trainable_state_dict:
                torch.save(trainable_state_dict, os.path.join(save_path, "trainable_base_params.bin"))

            # Save optimizer and step
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            torch.save({"step": step}, os.path.join(save_path, "trainer_state.pt"))

            # Create/update soft link to latest checkpoint
            latest_link = os.path.join(args.output_dir, "checkpoint_latest")
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.unlink(latest_link)
            os.symlink(os.path.relpath(save_path, args.output_dir), latest_link)

        step += 1

    # Final save
    if rank == 0:
        writer.close()
        final_path = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(final_path, exist_ok=True)
        model.module.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        # 2. 保存 base model 中可训练的参数（如 spec_embed_tokens）
        trainable_state_dict = {}
        for name, param in model.module.named_parameters():
            if param.requires_grad and "lora" not in name:
                # 排除 LoRA（已由 save_pretrained 处理），只存 base 中可训练部分
                trainable_state_dict[name] = param.data.cpu()

        if trainable_state_dict:
            torch.save(trainable_state_dict, os.path.join(final_path, "trainable_base_params.bin"))

        torch.save(optimizer.state_dict(), os.path.join(final_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(final_path, "scheduler.pt"))
        torch.save({"step": step}, os.path.join(final_path, "trainer_state.pt"))

        # Update latest soft link
        latest_link = os.path.join(args.output_dir, "checkpoint_latest")
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(os.path.relpath(final_path, args.output_dir), latest_link)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()


