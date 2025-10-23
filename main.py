import time
import json
import csv
from pathlib import Path
from typing import List, Dict
import sys
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
except Exception as e:
    print("ERROR: 'transformers' library is not installed or could not be imported:", e)
    print("Install it with: pip install --upgrade transformers accelerate")
    sys.exit(1)

# === Configuration ===
MODEL_LIST = [
    # Change these to the models you have license/access to
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-2-7b-chat-hf",
    "tiiuae/falcon-7b-instruct",
    # optional: "google/flan-t5-xl"
]
OUTPUT_DIR = Path("hf_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test queries
TEST_QUERIES = [
    {"lang": "en", "type": "product_info", "prompt": "User: I'm looking at product SKU 12345 (Bluetooth headphones). What is the battery life, and does it support aptX?"},
    {"lang": "es", "type": "return", "prompt": "Usuario: Quiero devolver unos auriculares que compré hace 25 días. ¿Cuál es su política de devoluciones y cómo inicio el proceso?"},
    {"lang": "fr", "type": "troubleshoot", "prompt": "Utilisateur: Mon chargeur ne fonctionne pas avec le modèle Z5. L'appareil ne s'allume pas. Que puis-je vérifier?"},
    {"lang": "de", "type": "compare", "prompt": "Benutzer: Ich möchte zwei Headset-Modelle vergleichen: A (battery 24h, noise-cancel) und B (battery 36h, open-back). Bitte gib mir eine Empfehlung für häufiges Reisen und Begründung."},
    {"lang": "jp", "type": "multilingual", "prompt": "ユーザー: 商品が届きましたが箱が破損しています。交換は可能ですか？どのように写真を送れば良いですか？"}
]

# Generation settings
GEN_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,   # deterministic
    "temperature": 0.0,
    # "top_p": 0.95
}

# Device helper
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def load_model_and_tokenizer(model_id: str):
    """Load appropriate model/tokenizer for a model_id.
       Handles seq2seq vs causal models heuristically for well-known families.
    """
    print(f"\nLoading {model_id} ...")
    # Some models (flan-t5) are seq2seq, others are causal
    seq2seq_prefixes = ["google/flan-t5", "t5-"]
    is_seq2seq = any(model_id.startswith(p) for p in seq2seq_prefixes)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tokenizer, model, is_seq2seq

def generate_response(model, tokenizer, prompt: str, is_seq2seq: bool):
    """Generate response and measure elapsed time (tokenize + inference)."""
    t0 = time.perf_counter()
    if is_seq2seq:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **GEN_KWARGS)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **GEN_KWARGS)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    t1 = time.perf_counter()
    latency = t1 - t0
    return text, latency

def evaluate_models(models: List[str], queries: List[Dict]):
    results = []
    for model_id in models:
        try:
            tokenizer, model, is_seq2seq = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"ERROR loading {model_id}: {e}")
            continue

        model_res = {"model": model_id, "runs": []}
        for q in queries:
            prompt = q["prompt"]
            try:
                resp_text, latency = generate_response(model, tokenizer, prompt, is_seq2seq)
            except Exception as e:
                resp_text = f"ERROR during generation: {e}"
                latency = None
            run = {
                "prompt": prompt,
                "lang": q["lang"],
                "type": q["type"],
                "response": resp_text,
                "latency_secs": latency
            }
            print(f"Model {model_id} | {q['lang']} | latency {latency:.3f}s")
            model_res["runs"].append(run)
        results.append(model_res)
        # optionally free memory
        del model
        torch.cuda.empty_cache()
    # Save results
    out_json = OUTPUT_DIR / "hf_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved results to", out_json)
    return results

if __name__ == "__main__":
    results = evaluate_models(MODEL_LIST, TEST_QUERIES)
