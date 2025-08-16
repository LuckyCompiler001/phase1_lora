import os, json, math, argparse
from typing import Dict
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def build_prompt_causal(row: Dict):
    # trained on data/alpaca_causal.jsonl
    return f"{row['prompt'].rstrip()}\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--data_path", default="data/alpaca_causal.jsonl")
    ap.add_argument("--output_dir", default="models/lora_llama2")
    ap.add_argument("--use_qlora", action="store_true", help="4-bit QLoRA; otherwise 8-bit LoRA")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--cutoff", type=int, default=0, help="if 0, compute from stats.json (p95+64, clip to [256, 1024])")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- seq length from yesterday's stats
    cutoff = args.cutoff
    if cutoff <= 0 and os.path.exists("data/stats.json"):
        with open("data/stats.json", "r", encoding="utf-8") as f:
            st = json.load(f)
        cutoff = int(min(1024, max(256, st["tokens_p95"] + 64)))
    if cutoff <= 0:
        cutoff = 512
    print(f"[train] max_seq_length = {cutoff}")

    # ---- tokenizer / model load (8-bit or 4-bit)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"device_map": "auto"}
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- dataset (jsonl)
    ds = load_dataset("json", data_files=args.data_path, split="train")

    def map_fn(example):
        # causal jsonl has: {"prompt": "...", "response": "..."}
        prompt = build_prompt_causal(example)
        full = prompt + example["response"].strip() + tok.eos_token
        out = tok(
            full,
            truncation=True,
            max_length=cutoff,
            padding="max_length",
        )
        # labels = input_ids (standard LM). If you want to ignore loss on padding:
        labels = out["input_ids"].copy()
        # pad tokens label = -100 (ignored by loss)
        pad_id = tok.pad_token_id
        out["labels"] = [(lid if iid != pad_id else -100) for iid, lid in zip(out["input_ids"], labels)]
        return out

    ds = ds.map(map_fn, remove_columns=ds.column_names, batched=False)

    # ---- trainer
    wc = int(max(1, math.floor(len(ds) * args.warmup_ratio)))
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,  # -1 to disable
        learning_rate=args.lr,
        warmup_steps=wc if args.max_steps < 0 else 0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=torch.cuda.is_available(),  # Windows+CUDA ok; silently ignored if not
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=train_args, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))

    # quick gen smoke test
    prompt = "Explain what LoRA does in one sentence."
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80)
    print("\n=== SAMPLE ===")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
