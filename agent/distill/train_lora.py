"""LoRA fine-tune a small Qwen model to write the agent's answers like the teacher did.

    agent\\distill\\.venv\\Scripts\\python agent\\distill\\train_lora.py --teacher gpt-5.4-mini --name park-writer-1.5b

Training examples are the teacher's grounded answers on `train` questions only, as chat turns: the
writer's system prompt, the writer prompt, and the teacher's answer. The loss covers the answer tokens
alone. LoRA adapters go on every attention and MLP projection; the base model stays frozen in bf16.

Saves agent/distill/models/<name>/adapter (the LoRA weights), merged/ (base + adapter, for Ollama to
import) and training.json (settings, data counts, loss per epoch, minutes).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from park_agent.harness import WRITER_SYSTEM  # noqa: E402


def examples(teacher):
    questions = {r["id"]: r for r in map(json.loads, (HERE / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines())}
    answers = [json.loads(line) for line in (HERE / "data" / f"teacher-{teacher.replace(':', '-')}.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    keep = [a for a in answers if a["split"] == "train" and a["grounded"] and a["answer"].strip()]
    return [(questions[a["id"]]["writer_prompt"], a["answer"].strip()) for a in keep], len(answers)


def encode(tokenizer, prompt, answer, max_len):
    """Token ids for the whole chat, with labels -100 everywhere but the answer."""
    messages = [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}]
    head = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = head + answer + tokenizer.eos_token
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(head_ids), len(ids)) + ids[len(head_ids):]
    return ids, labels


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs, n_answers = examples(args.teacher)
    print(f"{len(pairs)} training examples (of {n_answers} teacher answers)")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    data = [encode(tokenizer, p, a, args.max_len) for p, a in pairs]
    print(f"tokens per example: median {sorted(len(i) for i, _ in data)[len(data) // 2]}, max {max(len(i) for i, _ in data)}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(data) / args.accumulate)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / steps))))

    started, history = time.time(), []
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(data)
        total, count = 0.0, 0
        for i, (ids, labels) in enumerate(data, 1):
            batch = torch.tensor([ids], device="cuda")
            loss = model(input_ids=batch, labels=torch.tensor([labels], device="cuda")).loss
            (loss / args.accumulate).backward()
            total, count = total + loss.item(), count + 1
            if i % args.accumulate == 0 or i == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        history.append(round(total / count, 4))
        print(f"epoch {epoch + 1}/{args.epochs}: loss {history[-1]} ({(time.time() - started) / 60:.1f} min)")

    out = HERE / "models" / args.name
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out / "merged")
    (out / "training.json").write_text(json.dumps({
        "base": args.base, "teacher": args.teacher, "examples": len(pairs), "epochs": args.epochs, "rank": args.rank,
        "lr": args.lr, "accumulate": args.accumulate, "seed": args.seed, "loss_by_epoch": history,
        "minutes": round((time.time() - started) / 60, 1)}, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
