# === Configuration ===
import torch
import time
import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

MODEL_LIST = [
    # Change these to the models you have license/access to
    "HuggingFaceH4/zephyr-7b-beta",  # still large but faster
    #"tiiuae/falcon-7b-instruct",  # also popular
    #"gpt2"  # lightweight, just for testing
    #"google/flan-t5-base"

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
    """Robust load: fallback tokenizer, handle CPU vs GPU and missing accelerate."""
    print(f"\nLoading {model_id} ...")
    seq2seq_prefixes = ["google/flan-t5", "t5-"]
    is_seq2seq = any(model_id.startswith(p) for p in seq2seq_prefixes)

    # try fast tokenizer first, then fallback to slow
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=False)
    except Exception as e:
        print(f"Tokenizer fast load failed ({e}), retrying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=False)

    # decide load kwargs based on device & accelerate availability
    try:
        import accelerate  # type: ignore
        have_accelerate = True
    except Exception:
        have_accelerate = False

    if device == "cuda" and have_accelerate:
        # GPU + accelerate: use device_map auto and mixed dtype for perf
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        map_args = {"device_map": "auto", "torch_dtype": dtype, "local_files_only": False}
    else:
        # CPU or no accelerate: avoid device_map and use float32
        map_args = {"torch_dtype": torch.float32, "local_files_only": False}

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **map_args)
    else:
        # trust_remote_code for some repos; keep it only if needed
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **map_args)

    # ensure model on correct device when not using device_map
    if device == "cpu":
        model.to("cpu")
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
            latency_str = f"{latency:.3f}s" if latency is not None else "None"
            print(f"Model {model_id} | {q['lang']} | latency {latency_str}")
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

# local import to avoid importing torchvision at module import time

