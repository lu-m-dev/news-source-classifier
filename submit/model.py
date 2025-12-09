"""
Model Module for News Headline Classifier
==============================================================================
Compatible with official eval_project_b.py evaluation script.

The model uses:
- TF-IDF vectorizer with character n-grams (robust to URL-extracted text)
- Multinomial Naive Bayes classifier
==============================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional


class NewsClassifier(nn.Module):
    """
    News headline classifier that wraps scikit-learn models.
    Inherits from nn.Module for compatibility with official evaluation script.
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize classifier.
        
        Args:
            weights_path: Path to model weights (optional, will auto-load if None)
        """
        super().__init__()
        
        # Dummy parameter for PyTorch compatibility
        self.register_buffer('_dummy', torch.zeros(1))
        
        # Internal state
        self.vectorizer = None
        self.classifier = None
        self.classes = ['FoxNews', 'NBC']
        self._loaded = False
        
        # Auto-load if weights_path not specified or is placeholder
        if weights_path is None or weights_path == "__no_weights__.pth":
            self._auto_load()
    
    def _auto_load(self):
        """Auto-load model from model.pt in same directory."""
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        if os.path.exists(model_path):
            self._load_from_pt(model_path)
    
    def _load_from_pt(self, path: str):
        """Load vectorizer and classifier from .pt file."""
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                if 'vectorizer' in data:
                    self.vectorizer = data['vectorizer']
                if 'classifier' in data:
                    self.classifier = data['classifier']
                if 'classes' in data:
                    self.classes = data['classes']
                self._loaded = True
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
    
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state dict for PyTorch compatibility."""
        base_state = super().state_dict(*args, **kwargs)
        # Add marker that model is loaded
        base_state['_model_loaded'] = torch.tensor([1.0 if self._loaded else 0.0])
        return base_state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dict - also triggers auto-load of actual model."""
        # Handle PyTorch's state_dict loading
        filtered = {k: v for k, v in state_dict.items() 
                    if k in ['_dummy', '_model_loaded']}
        if filtered:
            super().load_state_dict(filtered, strict=False)
        
        # If not already loaded, auto-load
        if not self._loaded:
            self._auto_load()
        
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts: List of text strings (pseudo-headlines)
            
        Returns:
            List of predicted labels ('FoxNews' or 'NBC')
        """
        if not self._loaded or self.vectorizer is None or self.classifier is None:
            self._auto_load()
        
        if not texts:
            return []
        
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.classifier.predict(X)
        
        return list(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._loaded:
            self._auto_load()
        
        if not texts:
            return np.array([])
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def forward(self, texts: List[str]) -> List[str]:
        """Forward pass - same as predict for compatibility."""
        return self.predict(texts)


def get_model() -> NewsClassifier:
    """
    Load and return the trained model.
    
    Returns:
        NewsClassifier instance (already loaded)
    """
    model = NewsClassifier()
    return model


def predict(model: NewsClassifier, texts: List[str]) -> List[str]:
    """
    Predict labels for texts using the model.
    
    Args:
        model: NewsClassifier instance
        texts: List of text strings (pseudo-headlines extracted from URLs)
        
    Returns:
        List of predicted labels ('FoxNews' or 'NBC')
    """
    return model.predict(texts)


# For testing
if __name__ == '__main__':
    print("Testing NewsClassifier...")
    
    # Load model
    model = get_model()
    print(f"Model loaded: {model._loaded}")
    print(f"Classes: {model.classes}")
    
    # Test predictions
    test_texts = [
        "trump announces new immigration policy",
        "senate passes climate bill",
        "biden administration unveils plan",
        "supreme court ruling affects healthcare",
    ]
    
    predictions = predict(model, test_texts)
    probas = model.predict_proba(test_texts)
    
    print("\nPredictions:")
    for text, pred, proba in zip(test_texts, predictions, probas):
        conf = max(proba)
        print(f"  '{text}' -> {pred} ({conf:.3f})")
    
    # Test state_dict compatibility
    print("\nState dict keys:", list(model.state_dict().keys()))
