"""
DistilBERT model loader + prediction for submission (submit_llm)
Provides `get_model()` and `predict(model, texts)` to match evaluation interface.
The implementation loads `model.pt` and a locally saved tokenizer folder.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Optional

from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast
import tempfile
import shutil


class NewsLLMClassifier(nn.Module):
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        self.register_buffer('_dummy', torch.zeros(1))
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.classes = ['FoxNews', 'NBC']

        if weights_path is None or weights_path == '__no_weights__.pth':
            self._auto_load()
        else:
            self._load_from_pt(weights_path)

    def _auto_load(self):
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        if os.path.exists(model_path):
            self._load_from_pt(model_path)

    def _load_from_pt(self, path: str):
        try:
            data = torch.load(path, map_location='cpu')
            if 'tokenizer_files' in data and isinstance(data['tokenizer_files'], dict):
                tmpdir = tempfile.mkdtemp(prefix='tokenizer_')
                try:
                    for fname, content in data['tokenizer_files'].items():
                        with open(os.path.join(tmpdir, fname), 'wb') as fh:
                            fh.write(content)
                    self.tokenizer = DistilBertTokenizerFast.from_pretrained(tmpdir)
                finally:
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass
            else:
                self.tokenizer = None

            if 'config' in data:
                config = DistilBertConfig.from_dict(data['config'])
                model = DistilBertForSequenceClassification(config)
                state = None
                if 'model_state_dict' in data:
                    state = data['model_state_dict']
                elif 'state_dict' in data:
                    state = data['state_dict']
                if state is not None:
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        try:
                            norm = {k.replace('module.', ''): v for k, v in state.items()}
                            model.load_state_dict(norm, strict=False)
                        except Exception:
                            pass
                self.model = model

            if 'classes' in data:
                self.classes = data['classes']

            self._loaded = True
            try:
                dev = next(self.model.parameters()).device
                if dev.type == 'cuda':
                    try:
                        gpu_name = torch.cuda.get_device_name(dev.index if hasattr(dev, 'index') else 0)
                    except Exception:
                        gpu_name = 'CUDA'
                    print(f'Model loaded on device: CUDA ({gpu_name})')
                else:
                    print('Model loaded on device: CPU')
            except Exception:
                if torch.cuda.is_available():
                    print('Model loaded; CUDA available')
                else:
                    print('Model loaded; using CPU')
        except Exception as e:
            print(f"Warning: could not load LLM model from {path}: {e}")

    def predict(self, texts: List[str]) -> List[str]:
        if not self._loaded:
            self._auto_load()

        if not texts:
            return []

        if self.model is None:
            raise RuntimeError('Model not loaded')

        if self.tokenizer is None:
            raise RuntimeError('Tokenizer not found in checkpoint. Recreate model.pt with embedded tokenizer files.')

        enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        enc = {k: v.to(device) for k, v in enc.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        return [self.classes[int(p)] for p in preds]

    def forward(self, texts: List[str]) -> List[str]:
        return self.predict(texts)


def get_model() -> NewsLLMClassifier:
    return NewsLLMClassifier()


def predict(model: NewsLLMClassifier, texts: List[str]) -> List[str]:
    return model.predict(texts)


if __name__ == '__main__':
    print('Testing LLM model loader...')
    m = get_model()
    print('Loaded:', m._loaded)
    if m._loaded:
        print('Classes:', m.classes)
        print('Sample prediction:')
        print(predict(m, ['trump announces new immigration policy']))
