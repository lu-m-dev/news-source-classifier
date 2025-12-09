"""
Create model.pt for Hugging Face Submission
==============================================================================
This script trains the model and saves it for submission.

IMPORTANT: The model is trained on REAL headlines, but must generalize to
pseudo-headlines extracted from URLs during evaluation.

Key design decisions:
1. Use character n-grams (char_wb) to handle URL-based pseudo-headlines
2. Use simpler vocabulary to be robust to word variations
3. MultinomialNB is robust and generalizes well
==============================================================================
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re


def clean_text(text):
    """Clean text for training/inference."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_training_data():
    """Load training data from our collected dataset."""
    # Try multiple possible data file locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'news_train.json'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'news_data_processed.json'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'news_data_clean_v2.json'),
    ]
    
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    
    if not data_path:
        raise FileNotFoundError(f"Training data not found. Tried: {possible_paths}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    headlines = []
    labels = []
    
    for item in data:
        headline = clean_text(item.get('headline', ''))
        source = item.get('source', '')
        
        if headline and source:
            headlines.append(headline)
            labels.append(source)
    
    return headlines, labels


def main():
    print("=" * 60)
    print("CREATING model.pt FOR SUBMISSION")
    print("=" * 60)
    
    print("\n[1/5] Loading training data...")
    X, y = load_training_data()
    print(f"   Total samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    print("\n[2/5] Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        analyzer='char_wb',
        min_df=2,
        sublinear_tf=True,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    print("\n[3/5] Training MultinomialNB classifier...")
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train_tfidf, y_train)
    
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n[4/5] Saving model.pt...")
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'classes': list(classifier.classes_),
        'accuracy': accuracy,
        # Add dummy tensor for PyTorch compatibility with official eval script
        '_dummy': torch.zeros(1),
        '_model_loaded': torch.tensor([1.0]),
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    torch.save(model_data, model_path)
    print(f"   Saved to: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    print("\n[5/5] Verifying model.pt...")
    loaded = torch.load(model_path, weights_only=False)
    assert 'vectorizer' in loaded
    assert 'classifier' in loaded
    print("   Loaded successfully!")
    
    print("\n[BONUS] Testing with pseudo-headline examples...")
    pseudo_examples = [
        "trump announces new immigration policy",
        "senate passes climate bill",
        "biden administration unveils plan",
        "supreme court ruling affects",
    ]
    
    for example in pseudo_examples:
        example_tfidf = loaded['vectorizer'].transform([example])
        pred = loaded['classifier'].predict(example_tfidf)[0]
        proba = loaded['classifier'].predict_proba(example_tfidf)[0]
        print(f"   '{example}' -> {pred} (confidence: {max(proba):.3f})")
    
    print("\n" + "=" * 60)
    print("SUCCESS! model.pt created")
    print("=" * 60)


if __name__ == '__main__':
    main()
