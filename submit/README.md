# Submission Files - News Headline Classifier

## Backend Rules

- No HTTP requests allowed during evaluation
- Convert raw URLs to pseudo-headlines using string processing only
- Must work completely offline

## Evaluation Flow

```
URL → preprocess.py → pseudo-headline → model.py → "FoxNews" or "NBC"
```

## Required Files

| File | Description |
|------|-------------|
| preprocess.py | prepare_data(path) - URL to pseudo-headline |
| model.py | NewsClassifier with predict(texts) |
| model.pt | TF-IDF + MultinomialNB weights |

## Usage

```bash
python create_model_pt.py      # Generate model.pt
python eval_local.py           # Test locally
python preprocess.py           # Test URL conversion
```

## URL to Pseudo-headline Examples

| URL | Output |
|-----|--------|
| foxnews.com/politics/trump-announces-policy | trump announces policy |
| nbcnews.com/news/senate-passes-bill-rcna123 | senate passes bill |

## Model Info

- Training: Real headlines from Fox/NBC
- Evaluation: Pseudo-headlines from URLs
- Algorithm: TF-IDF + MultinomialNB
