"""
Local Evaluation Script
==============================================================================
This script mimics the backend evaluation process:
1. Load raw URLs from CSV
2. Convert URLs to pseudo-headlines (NO HTTP requests!)
3. Use model to predict labels
4. Calculate accuracy

Usage:
    python eval_local.py                    # Use default test data
    python eval_local.py path/to/urls.csv   # Use custom CSV
==============================================================================
"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Import our modules
from preprocess import prepare_data, url_to_pseudo_headline, clean_text
from model import get_model, predict


def evaluate_with_headlines(test_csv_path: str):
    """
    Evaluate using a CSV that already has headlines (our collected data).
    This tests the model's accuracy on real headlines.
    """
    print("=" * 60)
    print("LOCAL EVALUATION (with headlines)")
    print("=" * 60)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    df = pd.read_csv(test_csv_path)
    
    # Get headlines and labels
    if 'headline' in df.columns:
        headlines = df['headline'].apply(clean_text).tolist()
    elif 'text' in df.columns:
        headlines = df['text'].apply(clean_text).tolist()
    else:
        raise ValueError("No headline column found in CSV")
    
    if 'source' in df.columns:
        labels = df['source'].tolist()
    elif 'label' in df.columns:
        labels = df['label'].tolist()
    else:
        raise ValueError("No label column found in CSV")
    
    print(f"   Samples: {len(headlines)}")
    
    # Load model
    print("\n[2/3] Loading model...")
    model = get_model()
    print(f"   Model classes: {model.classes}")
    
    # Predict
    print("\n[3/3] Running predictions...")
    predictions = predict(model, headlines)
    
    # Calculate accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n   Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\n   Classification Report:")
    print(classification_report(labels, predictions))
    
    return accuracy


def evaluate_with_urls(url_csv_path: str):
    """
    Evaluate using a CSV that only has URLs (simulates backend evaluation).
    This is the TRUE test - converting URLs to pseudo-headlines.
    """
    print("=" * 60)
    print("LOCAL EVALUATION (URL-only, simulates backend)")
    print("=" * 60)
    
    # Use prepare_data which handles URL->pseudo-headline conversion
    print("\n[1/3] Preparing data from URLs...")
    X, y = prepare_data(url_csv_path)
    print(f"   Extracted {len(X)} pseudo-headlines")
    
    # Show some examples
    print("\n   Example URL -> Pseudo-headline conversions:")
    df = pd.read_csv(url_csv_path)
    url_col = None
    for col in ['url', 'URL', 'link']:
        if col in df.columns:
            url_col = col
            break
    
    if url_col:
        for i in range(min(3, len(df))):
            url = df[url_col].iloc[i]
            pseudo = url_to_pseudo_headline(url)
            print(f"   {url[:60]}...")
            print(f"   -> '{pseudo}'")
            print()
    
    # Load model
    print("[2/3] Loading model...")
    model = get_model()
    
    # Predict
    print("\n[3/3] Running predictions...")
    predictions = predict(model, X)
    
    # If we have labels, calculate accuracy
    if y and all(y):
        correct = sum(1 for p, l in zip(predictions, y) if p == l)
        total = len(y)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n   Accuracy: {accuracy:.4f} ({correct}/{total})")
        print("\n   Classification Report:")
        print(classification_report(y, predictions))
        
        return accuracy
    else:
        # No labels, just show predictions
        print("\n   No labels available. Prediction distribution:")
        fox_count = sum(1 for p in predictions if p == 'FoxNews')
        nbc_count = sum(1 for p in predictions if p == 'NBC')
        print(f"   FoxNews: {fox_count}, NBC: {nbc_count}")
        return None


def main():
    # Check for model.pt
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    if not os.path.exists(model_path):
        print("ERROR: model.pt not found!")
        print("Please run: python create_model_pt.py")
        return
    
    # Get test file
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Try to find test data
        test_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'test_data.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'url_only_data.csv'),
        ]
        test_path = None
        for p in test_paths:
            if os.path.exists(p):
                test_path = p
                break
        
        if not test_path:
            print("ERROR: No test data found!")
            print("Please provide a CSV file: python eval_local.py path/to/test.csv")
            return
    
    print(f"Using test file: {test_path}")
    
    # Determine test type based on columns
    df = pd.read_csv(test_path)
    has_headlines = any(col in df.columns for col in ['headline', 'text', 'title'])
    has_urls = any(col in df.columns for col in ['url', 'URL', 'link'])
    
    if has_headlines:
        # Test with real headlines
        evaluate_with_headlines(test_path)
    elif has_urls:
        # Test with URLs only (simulates backend)
        evaluate_with_urls(test_path)
    else:
        print("ERROR: CSV must have either headline or url columns!")


if __name__ == '__main__':
    main()
