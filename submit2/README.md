# Submit2 - Trained on Pseudo-Headlines

## Key Difference from submit/

| Version | Training Data | Test Data |
|---------|--------------|-----------|
| submit/ | Real headlines (scraped) | URL pseudo-headlines |
| submit2/ | URL pseudo-headlines | URL pseudo-headlines |

Training and test data distributions now match.

## Backend Evaluation Flow

```
URL → preprocess.py → pseudo-headline → model → prediction
```

## Usage

```bash
python create_model_pt.py
python eval_local.py
```

## Files

| File | Description |
|------|-------------|
| preprocess.py | URL to pseudo-headline conversion |
| model.py | NewsClassifier (nn.Module compatible) |
| create_model_pt.py | Train on pseudo-headlines from URLs |
| eval_local.py | Local testing |
| model.pt | Generated model weights |
