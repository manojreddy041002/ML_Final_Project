# mnist_vision_llm.py  (Ollama-only, no API keys)
# Classifies MNIST digits using a LOCAL vision LLM via Ollama (e.g., llava).
# Produces mnist_vision_llm.json so you can compare vs your CNN.

import os, time, json, traceback
import ollama
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# ---- Config (override via environment variables if you like) ----
MODEL     = os.getenv("OLLAMA_VISION_MODEL", "llava")   # e.g., "llava", "llava:13b"
SUBSET_N  = int(os.getenv("MNIST_LLM_SAMPLES", "10"))   # small, fast demo
PNG_DIR   = "mnist_png"
OUT_JSON  = "mnist_vision_llm.json"

def ensure_pngs(n):
    """Export n test MNIST images as PNGs and return (paths, labels)."""
    os.makedirs(PNG_DIR, exist_ok=True)
    tfm = transforms.ToTensor()
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    n = min(n, len(test))
    paths, y_true = [], []
    for i in range(n):
        img, y = test[i]
        path = os.path.join(PNG_DIR, f"{i}.png")
        if not os.path.exists(path):
            torchvision.utils.save_image(img, path)
        paths.append(path)
        y_true.append(int(y))
    return paths, y_true

def classify_one_image(path):
    """
    Ask the local Ollama vision model (e.g., llava) to classify the digit.
    Returns the *raw* string reply (we'll parse a digit out of it).
    """
    prompt = "Classify this handwritten digit (0-9). Return only the digit."
    resp = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [path],
        }],
        options={"temperature": 0}
    )
    return resp["message"]["content"].strip()

def main():
    # 1) Prepare images & labels
    paths, y_true = ensure_pngs(SUBSET_N)

    # 2) Inference loop
    y_pred = []
    t0 = time.time()

    for i, p in enumerate(paths):
        try:
            raw = classify_one_image(p)
            # Pick the first digit we see in the response; fallback to 0
            digit = next((int(ch) for ch in raw if ch.isdigit()), 0)
            y_pred.append(digit)
            print(f"[{i+1:02d}/{len(paths)}] {os.path.basename(p)}  ->  pred={digit}  gt={y_true[i]}  raw='{raw}'")
        except Exception as e:
            print(f"[warn] inference failed on {p}: {e}")
            traceback.print_exc()
            y_pred.append(0)

    latency = time.time() - t0
    acc = accuracy_score(y_true, y_pred)

    # 3) Persist results in the same schema your compare script expects
    result = {
        "model": f"Ollama:{MODEL}",
        "accuracy": float(acc),
        "n": len(paths),
        "input_tokens": None,          # Ollama doesn't expose tokens
        "output_tokens": None,         # Ollama doesn't expose tokens
        "estimated_cost_usd": 0.0,     # local inference is free
        "latency_sec": latency
    }

    print("\n=== Summary ===")
    print(json.dumps(result, indent=2))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")

if __name__ == "__main__":
    main()
