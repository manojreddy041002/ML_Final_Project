'''
import json
cnn = json.load(open("mnist_cnn.json"))
llm = json.load(open("mnist_vision_llm.json"))

print("CNN accuracy:", cnn["accuracy"])
print("Vision-LLM accuracy:", llm["accuracy"])
print("Vision-LLM cost (USD):", llm["estimated_cost_usd"])
'''

import json, os
from pathlib import Path
from tabulate import tabulate  # pip install tabulate
ROOT = Path(__file__).resolve().parents[1]

def load_json(path):
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

cnn = load_json(ROOT / "mnist_cnn.json")
llm = load_json(ROOT / "mnist_vision_llm.json")

if not cnn or not llm:
    exit()

table = [
    ["CNN", cnn.get("accuracy"), cnn.get("latency_sec"), cnn.get("params"), 0.0],
    [llm.get("model"), llm.get("accuracy"), llm.get("latency_sec"), "—", llm.get("estimated_cost_usd")],
]

headers = ["Model", "Accuracy", "Latency (s)", "Params", "Cost ($)"]
print(tabulate(table, headers=headers, floatfmt=".4f"))
