"""
Local evaluation script for submit_llm.

Workflow:
- Load data via `submit_llm/preprocess.py::prepare_data`
- Split into train/test (`--test-frac`, default 0.1)
- Write temporary train CSV and call `submit_llm.create_model_pt.main` with `--cv_folds` to train and save best `model.pt`
- Instantiate model via `submit_llm.model.get_model()` and evaluate on test set

Example:
  python submit_llm/local_eval.py --csv url_only_data.csv --test-frac 0.1 --cv_folds 5 --epochs 3
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib.util
import sys

base_dir = os.path.dirname(__file__)
preprocess_path = os.path.join(base_dir, 'preprocess.py')
trainer_path = os.path.join(base_dir, 'create_model_pt.py')
model_path = os.path.join(base_dir, 'model.py')

spec = importlib.util.spec_from_file_location('submit_llm.preprocess', preprocess_path)
preproc = importlib.util.module_from_spec(spec)
sys.modules['submit_llm.preprocess'] = preproc
spec.loader.exec_module(preproc)

spec = importlib.util.spec_from_file_location('submit_llm.create_model_pt', trainer_path)
trainer = importlib.util.module_from_spec(spec)
sys.modules['submit_llm.create_model_pt'] = trainer
spec.loader.exec_module(trainer)

spec = importlib.util.spec_from_file_location('submit_llm.model', model_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules['submit_llm.model'] = model_module
spec.loader.exec_module(model_module)


def accuracy(preds, targets):
    if len(preds) == 0:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if str(p) == str(t))
    return correct / len(targets)


def main(csv_path: str, test_frac: float = 0.1, cv_folds: int = 5, epochs: int = 8, batch_size: int = 32, lr: float = 2e-5, keep_temp: bool = False):
    print(f'Loading data from: {csv_path}')
    X, y = preproc.prepare_data(csv_path)
    if not X:
        raise RuntimeError('No examples found from preprocess.prepare_data')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)
    print(f'Total: {len(X)}  Train: {len(X_train)}  Test: {len(X_test)}')

    tmp_train = os.path.join(os.path.dirname(__file__), 'train_temp.csv')
    df_train = pd.DataFrame({
        'headline': X_train,
        'source': y_train,
    })
    df_train['label'] = df_train['source'].apply(lambda s: 0 if s == 'FoxNews' else 1 if s == 'NBC' else None)
    df_train.to_csv(tmp_train, index=False, encoding='utf-8-sig')
    print(f'Wrote temporary train CSV: {tmp_train} ({len(df_train)} rows)')
    print('Starting training (this may take a while)...')
    trainer.main(tmp_train, epochs=epochs, batch_size=batch_size, lr=lr, cv_folds=cv_folds)

    model = model_module.get_model()
    if not getattr(model, '_loaded', False):
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        if os.path.exists(model_path):
            try:
                model._load_from_pt(model_path)
            except Exception:
                pass

    print('Evaluating on test set...')
    preds = model.predict(X_test)
    acc = accuracy(preds, y_test)
    print(f'Test examples: {len(y_test)}')
    print(f'Accuracy: {acc:.6f}')

    if not keep_temp:
        try:
            os.remove(tmp_train)
        except Exception:
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), '..', 'url_only_data.csv'))
    p.add_argument('--test-frac', type=float, default=0.1)
    p.add_argument('--cv_folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--keep-temp', action='store_true')
    args = p.parse_args()

    main(args.csv, test_frac=args.test_frac, cv_folds=args.cv_folds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, keep_temp=args.keep_temp)
