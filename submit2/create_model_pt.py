"""
Create model.pt for Hugging Face Submission - Version 2
==============================================================================
KEY DIFFERENCE: Train on PSEUDO-HEADLINES extracted from URLs!

This matches the backend evaluation process:
- Backend gives URLs
- preprocess.py converts to pseudo-headlines
- Model predicts on pseudo-headlines

So we should TRAIN on pseudo-headlines too!
==============================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse, unquote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def url_to_pseudo_headline(url: str) -> str:
    """
    Convert a URL to a pseudo-headline WITHOUT making HTTP requests.
    MUST be identical to preprocess.py version!
    """
    if not url or not isinstance(url, str):
        return ""
    
    try:
        url = unquote(url)
        parsed = urlparse(url)
        path = parsed.path
        path = path.strip('/')
        segments = path.split('/')
        
        slug = ""
        for seg in reversed(segments):
            if seg and len(seg) > 5 and not seg.isdigit():
                slug = seg
                break
        
        if not slug:
            for seg in reversed(segments):
                if seg:
                    slug = seg
                    break
        
        # Clean the slug
        slug = re.sub(r'[-.]*(rcna|ncna|n)\d+$', '', slug, flags=re.I)
        slug = re.sub(r'\.(print|html|amp|php)$', '', slug, flags=re.I)
        slug = re.sub(r'[-_]\d+$', '', slug)
        
        headline = re.sub(r'[-_]+', ' ', slug)
        headline = re.sub(r'\s+', ' ', headline).strip()
        headline = headline.lower()
        
        return headline
        
    except Exception:
        return ""


def identify_source_from_url(url: str) -> str:
    """Identify news source from URL domain."""
    if not url:
        return ""
    url_lower = url.lower()
    if 'foxnews.com' in url_lower:
        return 'FoxNews'
    elif 'nbcnews.com' in url_lower or 'msnbc.com' in url_lower:
        return 'NBC'
    return ""


def clean_text(text: str) -> str:
    """Clean text for training/inference."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_training_data_from_urls():
    """
    Load training data by converting URLs to pseudo-headlines.
    This is the KEY CHANGE from v1!
    """
    # Find URL data file
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'url_only_data.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'url_only_data.csv'),
    ]
    
    url_path = None
    for p in possible_paths:
        if os.path.exists(p):
            url_path = p
            break
    
    if not url_path:
        raise FileNotFoundError(f"URL data not found. Tried: {possible_paths}")
    
    print(f"   Loading URLs from: {url_path}")
    
    # Read CSV
    df = pd.read_csv(url_path, encoding='utf-8-sig')
    
    # Find URL column
    url_col = None
    for col in ['url', 'URL', 'link']:
        if col in df.columns:
            url_col = col
            break
    
    if url_col is None:
        url_col = df.columns[0]
    
    headlines = []
    labels = []
    
    for idx, row in df.iterrows():
        url = str(row[url_col])
        
        # Convert URL to pseudo-headline (same as preprocess.py)
        pseudo_headline = url_to_pseudo_headline(url)
        pseudo_headline = clean_text(pseudo_headline)
        
        # Get label from URL domain
        label = identify_source_from_url(url)
        
        # Skip invalid entries
        if len(pseudo_headline) < 5 or not label:
            continue
        
        headlines.append(pseudo_headline)
        labels.append(label)
    
    print(f"   Extracted {len(headlines)} pseudo-headlines from URLs")
    
    return headlines, labels


def main():
    print("=" * 60)
    print("CREATING model.pt FOR SUBMISSION - VERSION 2")
    print("Training on PSEUDO-HEADLINES (matches backend evaluation)")
    print("=" * 60)
    
    # Load data from URLs -> pseudo-headlines
    print("\n[1/5] Loading training data from URLs...")
    X, y = load_training_data_from_urls()
    
    # Count by class
    fox_count = sum(1 for l in y if l == 'FoxNews')
    nbc_count = sum(1 for l in y if l == 'NBC')
    print(f"   Total: {len(X)} (FoxNews: {fox_count}, NBC: {nbc_count})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Show some examples
    print("\n   Sample pseudo-headlines:")
    for i in range(min(3, len(X_train))):
        print(f"   [{y_train[i]}] {X_train[i][:60]}...")
    
    # Create TF-IDF vectorizer
    # Using word n-grams since pseudo-headlines are word-based
    print("\n[2/5] Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        analyzer='word',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train multiple classifiers and pick the best
    print("\n[3/5] Training classifiers...")
    
    classifiers = {
        'MultinomialNB': MultinomialNB(alpha=0.1),
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
        'LinearSVC': LinearSVC(max_iter=2000, C=1.0),
    }
    
    best_clf = None
    best_acc = 0
    best_name = ""
    
    for name, clf in classifiers.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print(f"   {name}: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_name = name
    
    print(f"\n   Best: {best_name} with accuracy {best_acc:.4f}")
    
    # Final evaluation
    y_pred = best_clf.predict(X_test_tfidf)
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\n[4/5] Saving model.pt...")
    model_data = {
        'vectorizer': vectorizer,
        'classifier': best_clf,
        'classes': ['FoxNews', 'NBC'],
        'accuracy': best_acc,
        'model_name': best_name,
        # PyTorch compatibility
        '_dummy': torch.zeros(1),
        '_model_loaded': torch.tensor([1.0]),
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    torch.save(model_data, model_path)
    print(f"   Saved to: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    # Verify
    print("\n[5/5] Verifying model.pt...")
    loaded = torch.load(model_path, weights_only=False)
    assert 'vectorizer' in loaded
    assert 'classifier' in loaded
    print("   Loaded successfully!")
    
    # Test predictions
    print("\n[BONUS] Testing predictions...")
    test_examples = [
        "trump announces new immigration policy",
        "senate passes climate bill",
        "biden administration unveils plan",
        "harris campaign rally draws crowd",
    ]
    
    for example in test_examples:
        example_tfidf = loaded['vectorizer'].transform([example])
        pred = loaded['classifier'].predict(example_tfidf)[0]
        print(f"   '{example}' -> {pred}")
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! model.pt created (trained on pseudo-headlines)")
    print(f"Best model: {best_name}, Accuracy: {best_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

