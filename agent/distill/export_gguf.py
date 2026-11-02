"""Write a merged Qwen2 checkpoint as a GGUF file and register it with Ollama.

    agent\\distill\\.venv\\Scripts\\python agent\\distill\\export_gguf.py park-writer-1.5b

Ollama 0.40 imports safetensors only for some architectures (not Qwen2) and no longer takes LoRA
adapters, so the merged model is written as GGUF with the `gguf` library (the format's own writer).
Only Qwen2 is handled: tensor names come from gguf's map for that architecture, weight matrices are
stored as F16 and 1-D tensors (norms, biases) as F32, and the tokenizer is the byte-level BPE vocab
with its merges, special tokens and chat template. It stays F16 (3.1 GB for 1.5B): Ollama only
quantizes safetensors imports.

Check the export before trusting it: `compare` mode greedy-decodes a few prompts in transformers and
through Ollama and prints both; they should match.
"""

import json
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent


def write_gguf(merged, path):
    config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen2ForCausalLM"], config["architectures"]
    writer = gguf.GGUFWriter(path, "qwen2")
    writer.add_name(merged.parent.name)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # Tokenizer: every id up to the model's vocab size; ids the tokenizer doesn't use are padding.
    tokenizer = AutoTokenizer.from_pretrained(merged)
    by_id = {i: t for t, i in tokenizer.get_vocab().items()}
    added = tokenizer.added_tokens_decoder
    tokens, types = [], []
    for i in range(config["vocab_size"]):
        if i not in by_id:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
        elif i in added:
            tokens.append(by_id[i])
            types.append(gguf.TokenType.CONTROL if added[i].special else gguf.TokenType.USER_DEFINED)
        else:
            tokens.append(by_id[i])
            types.append(gguf.TokenType.NORMAL)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    gguf.SpecialVocab(merged, load_merges=True, n_vocab=len(tokens)).add_to_gguf(writer)

    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, config["num_hidden_layers"])
    for shard in sorted(merged.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                name = names.get_name(key, try_suffixes=(".weight", ".bias"))
                if name is None:
                    raise KeyError(f"no GGUF name for {key}")
                t = f.get_tensor(key).to(torch.float32)
                writer.add_tensor(name, (t.numpy() if t.ndim == 1 else t.to(torch.float16).numpy()).astype(
                    np.float32 if t.ndim == 1 else np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def register(name, folder):
    # Long writer prompts run to ~3,000 tokens, past Ollama's default context; and no repeat penalty,
    # which training didn't have.
    (folder / "Modelfile").write_text(f"FROM ./{name}.gguf\nPARAMETER temperature 0.2\nPARAMETER num_ctx 8192\n"
                                      "PARAMETER repeat_penalty 1.0\n", encoding="ascii")
    subprocess.run(["ollama", "create", name, "-f", str(folder / "Modelfile")], cwd=folder, check=True)


def compare(name, folder, n=3):
    """Greedy answers from transformers and from Ollama for the first few held-out prompts."""
    import openai
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(HERE.parent))
    from park_agent.harness import WRITER_SYSTEM

    rows = [json.loads(line) for line in (HERE / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == "heldout"][:n]
    tok = AutoTokenizer.from_pretrained(folder / "merged")
    model = AutoModelForCausalLM.from_pretrained(folder / "merged", torch_dtype=torch.bfloat16).to("cuda").eval()
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="local")
    for r in rows:
        messages = [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": r["writer_prompt"]}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=200, do_sample=False, repetition_penalty=1.0)   # Qwen's generation_config adds 1.05
        hf = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ol = client.chat.completions.create(model=name, messages=messages, temperature=0, max_tokens=200)
        print(f"--- {r['question']}\n[transformers] {hf}\n[ollama]       {ol.choices[0].message.content}\n")


if __name__ == "__main__":
    name = sys.argv[1]
    folder = HERE / "models" / name
    if len(sys.argv) > 2 and sys.argv[2] == "compare":
        compare(name, folder)
    else:
        write_gguf(folder / "merged", folder / f"{name}.gguf")
        register(name, folder)
