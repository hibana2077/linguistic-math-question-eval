# linguistic-math-question-eval

This repository contains a simple evaluation script for testing the mathematical reasoning ability of models hosted on [OpenRouter](https://openrouter.ai) with question sets similar to GSM8K. Two example datasets are provided:

- `data/sample_gsm8k_en.json` – English problems
- `data/sample_gsm8k_zh.json` – Chinese translations of the same problems

## Requirements

- Python 3.11+
- `requests`
- `tqdm`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt`:

```
requests
tqdm
```

## Usage

1. Obtain an API key from OpenRouter and set the environment variable `OPENROUTER_API_KEY`.
2. Run the evaluation script specifying the dataset file:

```bash
python evaluate_openrouter.py data/sample_gsm8k_en.json --model mistralai/mistral-7b-instruct
```

Use the `--mock` flag to skip API calls and get a quick test run. The script outputs the accuracy of the model on the given dataset.

## Testing

Run the unit tests with:

```bash
python -m unittest -v
```
