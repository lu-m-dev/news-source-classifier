# DistilBERT

News source classification pipeline that converts URLs to pseudo-headlines, fine-tunes a DistilBERT classifier, and produces a PyTorch checkpoint for evaluation.

## Implementation
- **Model:** DistilBERT (`transformers.DistilBertForSequenceClassification`) for binary source classification.
- **Preprocess:** `submit_DistilBERT/preprocess.py` — `prepare_data(path)` converts URL CSVs into pseudo-headlines and labels (no HTTP requests).
- **Training:** `submit_DistilBERT/create_model_pt.py` — fine-tunes DistilBERT, supports stratified cross-validation (`--cv_folds`) and saves the best weights and tokenizer.
- **Runtime:** `submit_DistilBERT/model.py` — `get_model()` / `predict(model, texts)` loads `submit_DistilBERT/model.pt` and the tokenizer in `submit_DistilBERT/tokenizer/` for inference.

## Performance
- Reported result: training on 90% / testing on 10% of provided samples produced **85.53%** accuracy on the hold-out test set.

## Quick Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Fine-tune
Commands below use placeholders — replace the placeholders with your paths or values.

Cross-validated fine-tuning:

```bash
python submit_DistilBERT/create_model_pt.py \
  --csv <CSV_PATH>               # default: url_only_data.csv
  --epochs <EPOCHS>              # default: 8
  --batch_size <BATCH_SIZE>      # default: 32
  --lr <LEARNING_RATE>           # default: 2e-5
  --cv_folds <FOLDS>             # default: 1 (set to 5 for 5-fold CV)
```

Example (5-fold CV):
```bash
python submit_DistilBERT/create_model_pt.py --csv train_urls.csv --epochs 8 --batch_size 32 --lr 2e-5 --cv_folds 5
```

Outputs:
- model weights: [submit_DistilBERT/model.pt](submit_DistilBERT/model.pt)
- tokenizer: [submit_DistilBERT/tokenizer/](submit_DistilBERT/tokenizer/)

## Evaluate
### Local evaluation (recommended)
Use the bundled `eval_local.py` to run a full local workflow that:
- splits the input CSV into train/test (default 90/10),
- trains with stratified cross-validation and saves the best `submit_DistilBERT/model.pt`,
- loads `submit_DistilBERT/model.py` and evaluates accuracy on the held-out test set.

```bash
python submit_DistilBERT/eval_local.py \
  --csv <CSV_PATH>           # default: url_only_data.csv
  --test-frac <F>            # default: 0.1 (10% test)
  --cv_folds <FOLDS>         # default: 5
  --epochs <EPOCHS>          # default: 8
  --batch_size <BATCH_SIZE>  # default: 32
  --lr <LEARNING_RATE>       # default: 2e-5
```

Example:

```bash
python submit_DistilBERT/eval_local.py --csv url_only_data.csv --test-frac 0.1 --cv_folds 5 --epochs 8 --batch_size 32 --lr 2e-5
```

The script prints the train/test split sizes, trains (may take a while), and prints final test `Accuracy`.