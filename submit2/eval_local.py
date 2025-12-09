"""
Local Evaluation Script - Version 2
==============================================================================
Tests the model on URL data, simulating backend evaluation.
==============================================================================
"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from preprocess import prepare_data, url_to_pseudo_headline, clean_text
from model import get_model, predict


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
        test_path = os.path.join(os.path.dirname(__file__), '..', 'url_only_data.csv')
    
    if not os.path.exists(test_path):
        print(f"ERROR: Test file not found: {test_path}")
        return
    
    print("=" * 60)
    print("LOCAL EVALUATION - Version 2")
    print("(Trained on pseudo-headlines, tested on pseudo-headlines)")
    print("=" * 60)
    
    # Prepare data
    print(f"\n[1/3] Loading data from: {test_path}")
    X, y = prepare_data(test_path)
    print(f"   Extracted {len(X)} pseudo-headlines")
    
    # Show examples
    print("\n   Sample pseudo-headlines:")
    for i in range(min(3, len(X))):
        print(f"   [{y[i]}] {X[i][:50]}...")
    
    # Load model
    print("\n[2/3] Loading model...")
    model = get_model()
    print(f"   Model loaded: {model._loaded}")
    
    # Predict
    print("\n[3/3] Running predictions...")
    predictions = predict(model, X)
    
    # Calculate accuracy
    if y and all(y):
        correct = sum(1 for p, l in zip(predictions, y) if p == l)
        total = len(y)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n   Accuracy: {accuracy:.4f} ({correct}/{total})")
        print("\n   Classification Report:")
        print(classification_report(y, predictions))
    else:
        print("\n   No labels available. Prediction distribution:")
        fox_count = sum(1 for p in predictions if p == 'FoxNews')
        nbc_count = sum(1 for p in predictions if p == 'NBC')
        print(f"   FoxNews: {fox_count}, NBC: {nbc_count}")


if __name__ == '__main__':
    main()

