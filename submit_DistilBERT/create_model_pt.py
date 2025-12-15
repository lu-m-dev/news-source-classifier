"""
Fine-tune DistilBERT on pseudo-headlines and save model.pt + tokenizer

This script uses the `prepare_data` from `preprocess.py` and trains a
DistilBERT sequence classification head to distinguish FoxNews vs NBC.

After training it saves:
- model.pt containing: model_state_dict, config, classes
- tokenizer files under `submit_llm/tokenizer/`

Run: python create_model_pt.py
"""

import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import argparse
from typing import List
from sklearn.model_selection import StratifiedKFold

import importlib.util
import sys
import tempfile
import shutil

preprocess_path = os.path.join(os.path.dirname(__file__), 'preprocess.py')
spec = importlib.util.spec_from_file_location('submit_llm.preprocess', preprocess_path)
preprocess = importlib.util.module_from_spec(spec)
sys.modules['submit_llm.preprocess'] = preprocess
spec.loader.exec_module(preprocess)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def label_map_from_labels(labels: List[str]):
    classes = sorted(list(set(labels)))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return classes, label2id, id2label


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def main(csv_path: str, epochs: int = 2, batch_size: int = 16, lr: float = 2e-5, cv_folds: int = 1):
    print('Loading data via preprocess.prepare_data...')
    X, y = preprocess.prepare_data(csv_path)
    if not X:
        raise RuntimeError('No training examples found')

    classes, label2id, id2label = label_map_from_labels(y)
    y_ids = [label2id[label] for label in y]

    print(f'Examples: {len(X)} Classes: {classes}')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.num_labels = len(classes)
    config.id2label = {str(k): v for k, v in id2label.items()}
    config.label2id = {v: k for k, v in id2label.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = 'CUDA'
        print(f'Using device: CUDA ({gpu_name})')
    else:
        print('Using device: CPU')

    best_acc = -1.0
    best_state = None

    if cv_folds is None or cv_folds <= 1:
        enc = tokenizer(X, truncation=True, padding=True, return_tensors='pt')
        dataset = TextDataset(enc, y_ids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
        model.to(device)

        optim = AdamW(model.parameters(), lr=lr)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in loader:
                optim.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optim.step()
                scheduler.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)
            print(f'Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}')

        best_state = model.cpu().state_dict()
        best_acc = None

    else:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_ids), start=1):
            print(f'--- Fold {fold}/{cv_folds} ---')
            X_train = [X[i] for i in train_idx]
            y_train = [y_ids[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y_ids[i] for i in val_idx]

            enc_train = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
            enc_val = tokenizer(X_val, truncation=True, padding=True, return_tensors='pt')

            train_ds = TextDataset(enc_train, y_train)
            val_ds = TextDataset(enc_val, y_val)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
            model.to(device)

            optim = AdamW(model.parameters(), lr=lr)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

            for epoch in range(epochs):
                running_loss = 0.0
                model.train()
                for batch in train_loader:
                    optim.zero_grad()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optim.step()
                    scheduler.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(train_loader)
                print(f'Fold {fold} Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}')

            val_acc = evaluate(model, val_loader, device)
            print(f'Fold {fold} val_acc={val_acc:.4f}')

            if val_acc > best_acc:
                print(f'New best model found on fold {fold} (acc={val_acc:.4f})')
                best_acc = val_acc
                best_state = model.cpu().state_dict()

    out_dir = os.path.dirname(__file__)

    tmp_tokenizer_dir = None
    tokenizer_files = {}
    try:
        tmp_tokenizer_dir = tempfile.mkdtemp(prefix='tokenizer_')
        tokenizer.save_pretrained(tmp_tokenizer_dir)
        for fname in os.listdir(tmp_tokenizer_dir):
            fpath = os.path.join(tmp_tokenizer_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, 'rb') as fh:
                    tokenizer_files[fname] = fh.read()
    finally:
        if tmp_tokenizer_dir is not None:
            try:
                shutil.rmtree(tmp_tokenizer_dir)
            except Exception:
                pass

    model_path = os.path.join(out_dir, 'model.pt')
    save_data = {
        'state_dict': best_state,
        'model_state_dict': best_state,
        'config': config.to_dict(),
        'classes': classes,
        'best_cv_acc': best_acc,
        'tokenizer_files': tokenizer_files,
    }
    torch.save(save_data, model_path)
    print(f'Saved model.pt to {model_path} (best_cv_acc={best_acc})')
    print('Tokenizer embedded into checkpoint (no tokenizer/ folder left on disk)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), '..', 'url_only_data.csv'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--cv_folds', type=int, default=1, help='Number of stratified CV folds (1 = no CV)')
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    main(csv_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, cv_folds=args.cv_folds)
