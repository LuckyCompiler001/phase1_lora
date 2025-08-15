import argparse, json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer

def to_chat_item(inst, inp, out):
    # System can be extended later; keep minimal now.
    msgs = []
    if inp and len(inp.strip()) > 0:
        user = f"{inst.strip()}\n\nInput:\n{inp.strip()}"
    else:
        user = inst.strip()
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": out.strip()})
    return {"messages": msgs}

def to_causal_item(inst, inp, out):
    if inp and len(inp.strip()) > 0:
        prompt = f"Instruction: {inst.strip()}\n\nInput: {inp.strip()}\n\nOutput:"
    else:
        prompt = f"Instruction: {inst.strip()}\n\nOutput:"
    return {"prompt": prompt, "response": out.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf",
                    help="Tokenizer source to measure token lengths")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca", help="HF dataset repo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="optional cap for quick runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=2048, help="warn if prompt+resp exceed this")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)

    # Optional cap for faster iteration
    if args.limit and args.limit > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))
        print(f"Capped to {len(ds)} samples")

    # Prepare outputs
    sft_path = os.path.join(args.out_dir, "alpaca_sft.jsonl")
    causal_path = os.path.join(args.out_dir, "alpaca_causal.jsonl")
    stats_path = os.path.join(args.out_dir, "stats.json")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    eos = tok.eos_token or ""

    total = 0
    lengths = []
    long_ids = []

    with open(sft_path, "w", encoding="utf-8") as f_sft, \
         open(causal_path, "w", encoding="utf-8") as f_causal:

        for i, ex in enumerate(ds):
            inst = ex.get("instruction", "")
            inp  = ex.get("input", "")
            out  = ex.get("output", "")

            # Skip obviously empty/garbage rows
            if not inst or not out:
                continue

            chat_item = to_chat_item(inst, inp, out)
            causal_item = to_causal_item(inst, inp, out)

            # Token length check (approx for llama tokenizer)
            # Build one sequence like training would: prompt + response + eos
            prompt_text = chat_item["messages"][0]["content"]
            full = prompt_text + "\n\nAssistant:\n" + out + eos
            ids = tok(full, add_special_tokens=False).input_ids
            lengths.append(len(ids))
            if len(ids) > args.max_tokens:
                long_ids.append(i)

            f_sft.write(json.dumps({"id": i, **chat_item}, ensure_ascii=False) + "\n")
            f_causal.write(json.dumps({"id": i, **causal_item}, ensure_ascii=False) + "\n")
            total += 1

    # Basic stats
    def pct(a, p):
        if not a: return 0
        a = sorted(a)
        k = int(round((p/100.0)*(len(a)-1)))
        return a[max(0, min(k, len(a)-1))]

    stats = {
        "samples": total,
        "tokens_p50": pct(lengths, 50),
        "tokens_p90": pct(lengths, 90),
        "tokens_p95": pct(lengths, 95),
        "max_tokens": max(lengths) if lengths else 0,
        "over_limit_count": len(long_ids),
        "over_limit_ids_head": long_ids[:10],
        "max_allowed": args.max_tokens,
        "tokenizer": args.model
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote:\n  {sft_path}\n  {causal_path}\n  {stats_path}")
    print(f"Samples: {total} | p50={stats['tokens_p50']} p90={stats['tokens_p90']} p95={stats['tokens_p95']} max={stats['max_tokens']}")
    if stats["over_limit_count"] > 0:
        print(f"⚠️  {stats['over_limit_count']} samples exceed {args.max_tokens} tokens (prompt+resp). Consider truncation or smaller max_length later.")

if __name__ == "__main__":
    main()
