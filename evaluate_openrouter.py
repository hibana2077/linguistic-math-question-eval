import os
import json
import re
import argparse
from typing import List, Dict
import requests
from tqdm import tqdm

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1] if matches else None


def query_openrouter(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {"model": model, "messages": messages}
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def evaluate(dataset_path: str, model: str, max_samples: int | None = None, mock: bool = False) -> float:
    data = load_dataset(dataset_path)
    if max_samples:
        data = data[:max_samples]
    key = os.environ.get("OPENROUTER_API_KEY")
    if not mock and not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    correct = 0
    for sample in tqdm(data, desc="evaluating"):
        question = sample["question"]
        expected = sample["answer"].strip()
        if mock:
            reply = expected
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves math word problems."},
                {"role": "user", "content": question},
            ]
            reply = query_openrouter(messages, model, key)
        predicted = parse_answer(reply)
        if predicted == expected:
            correct += 1
    return correct / len(data) if data else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenRouter model on GSM8K-style dataset")
    parser.add_argument("dataset", help="Path to dataset JSON file")
    parser.add_argument("--model", default="mistralai/mistral-7b-instruct", help="OpenRouter model name")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--mock", action="store_true", help="Skip API calls and assume perfect model")
    args = parser.parse_args()

    acc = evaluate(args.dataset, args.model, args.max_samples, args.mock)
    print(f"Accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
