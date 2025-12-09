"""
==============================================================================
Script 4: Feature Analysis & Generalization Test
==============================================================================
Purpose: Analyze feature importance and test model robustness
Input:   data/news_train.json, news_val.json, news_test.json
Output:  data/feature_analysis.png

Analysis:
    1. Top features for each class (Fox News vs NBC)
    2. High-risk features (person names that may not generalize)
    3. Robustness test: accuracy with/without person names

Key Finding:
    - Accuracy drop when removing names: only 1.07%
    - Model learns writing style, not just specific names
    - Good generalization expected

Usage:
    python analyze_features.py
==============================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

def load_data():
    """Load data"""
    with open('data/news_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/news_val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('data/news_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    train_val = train_data + val_data
    
    X_train = [d['headline'] for d in train_val]
    y_train = [1 if d['source'] == 'FoxNews' else 0 for d in train_val]
    X_test = [d['headline'] for d in test_data]
    y_test = [1 if d['source'] == 'FoxNews' else 0 for d in test_data]
    
    return X_train, y_train, X_test, y_test

def analyze_feature_importance(X_train, y_train):
    """Analyze feature importance"""
    print("="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Train model
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X_train)
    
    # Use Logistic Regression to get feature weights
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_vec, y_train)
    
    # Get feature names and weights
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Sort by coefficient
    sorted_idx = np.argsort(coefficients)
    
    # Top features for each class
    print("\n[Top 20 Features indicating FOX NEWS (positive coefficients)]")
    for i in sorted_idx[-20:][::-1]:
        print(f"  {feature_names[i]}: {coefficients[i]:.4f}")
    
    print("\n[Top 20 Features indicating NBC (negative coefficients)]")
    for i in sorted_idx[:20]:
        print(f"  {feature_names[i]}: {coefficients[i]:.4f}")
    
    return feature_names, coefficients

def identify_risky_features(feature_names, coefficients):
    """Identify high-risk features that may affect generalization"""
    print("\n" + "="*60)
    print("RISKY FEATURES ANALYSIS")
    print("="*60)
    
    # Define high-risk patterns (person names, specific events, etc.)
    risky_patterns = [
        r'^trump$', r'^biden$', r'^harris$', r'^kamala$', r'^walz$',
        r'^obama$', r'^clinton$', r'^pelosi$', r'^mcconnell$',
        r'^desantis$', r'^vance$', r'^pence$',
        r'^2024$', r'^2023$', r'^2022$',
    ]
    
    risky_features = []
    for pattern in risky_patterns:
        for i, name in enumerate(feature_names):
            if re.search(pattern, name, re.I):
                risky_features.append({
                    'feature': name,
                    'coefficient': coefficients[i],
                    'abs_coef': abs(coefficients[i])
                })
    
    # Sort by impact
    risky_features = sorted(risky_features, key=lambda x: x['abs_coef'], reverse=True)
    
    print("\n[High-Risk Features (may not generalize well)]")
    for rf in risky_features[:15]:
        direction = "-> FoxNews" if rf['coefficient'] > 0 else "-> NBC"
        print(f"  {rf['feature']}: {rf['coefficient']:.4f} {direction}")
    
    return risky_features

def test_robustness(X_train, y_train, X_test, y_test):
    """Test model robustness - performance after removing person names"""
    print("\n" + "="*60)
    print("ROBUSTNESS TEST")
    print("="*60)
    
    # List of names to remove
    names_to_remove = [
        'trump', 'biden', 'harris', 'kamala', 'walz', 'vance',
        'obama', 'clinton', 'pelosi', 'mcconnell', 'desantis', 'pence'
    ]
    
    def remove_names(text):
        for name in names_to_remove:
            text = re.sub(r'\b' + name + r'\b', '', text, flags=re.I)
        return text.strip()
    
    # Test with original data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)
    y_pred_original = model.predict(X_test_vec)
    acc_original = accuracy_score(y_test, y_pred_original)
    
    # Test after removing names
    X_train_clean = [remove_names(x) for x in X_train]
    X_test_clean = [remove_names(x) for x in X_test]
    
    vectorizer_clean = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
    X_train_vec_clean = vectorizer_clean.fit_transform(X_train_clean)
    X_test_vec_clean = vectorizer_clean.transform(X_test_clean)
    
    model_clean = MultinomialNB(alpha=0.5)
    model_clean.fit(X_train_vec_clean, y_train)
    y_pred_clean = model_clean.predict(X_test_vec_clean)
    acc_clean = accuracy_score(y_test, y_pred_clean)
    
    print(f"\nOriginal accuracy: {acc_original:.4f}")
    print(f"Without names:     {acc_clean:.4f}")
    print(f"Difference:        {acc_original - acc_clean:.4f}")
    
    if acc_original - acc_clean > 0.05:
        print("\n[WARNING] Model relies heavily on person names!")
        print("  This may affect generalization to future news.")
    else:
        print("\n[GOOD] Model does not overly rely on person names.")
        print("  Generalization should be reasonable.")
    
    return acc_original, acc_clean

def visualize_analysis(feature_names, coefficients, risky_features):
    """Visualize analysis results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Top features
    ax1 = axes[0]
    sorted_idx = np.argsort(coefficients)
    top_pos = sorted_idx[-15:]
    top_neg = sorted_idx[:15]
    
    top_idx = np.concatenate([top_neg, top_pos])
    top_names = [feature_names[i] for i in top_idx]
    top_coefs = [coefficients[i] for i in top_idx]
    
    colors = ['#4ECDC4' if c < 0 else '#FF6B6B' for c in top_coefs]
    ax1.barh(range(len(top_names)), top_coefs, color=colors)
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names, fontsize=8)
    ax1.set_xlabel('Coefficient')
    ax1.set_title('Top Features by Class\n(Red=FoxNews, Cyan=NBC)', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Risky features
    ax2 = axes[1]
    if risky_features:
        rf_names = [rf['feature'] for rf in risky_features[:10]]
        rf_coefs = [rf['coefficient'] for rf in risky_features[:10]]
        colors2 = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in rf_coefs]
        ax2.barh(range(len(rf_names)), rf_coefs, color=colors2)
        ax2.set_yticks(range(len(rf_names)))
        ax2.set_yticklabels(rf_names)
        ax2.set_xlabel('Coefficient')
        ax2.set_title('Risky Features (Person Names)\nMay affect generalization', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('data/feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to: data/feature_analysis.png")

def main():
    print("="*60)
    print("MODEL GENERALIZATION ANALYSIS")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Analyze feature importance
    feature_names, coefficients = analyze_feature_importance(X_train, y_train)
    
    # Identify risky features
    risky_features = identify_risky_features(feature_names, coefficients)
    
    # Robustness test
    acc_orig, acc_clean = test_robustness(X_train, y_train, X_test, y_test)
    
    # Visualization
    visualize_analysis(feature_names, coefficients, risky_features)
    
    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
Model Generalization Assessment:

1. If accuracy drop > 5% when removing names:
   - Model relies too much on specific persons
   - May perform worse on future news with different figures
   
2. If accuracy drop < 5%:
   - Model learns more general patterns
   - Should generalize reasonably well

Recommendations for better generalization:
- Use more abstract features (sentiment, writing style)
- Consider character n-grams for stylistic patterns
- Add regularization to reduce overfitting
- Collect more diverse time-period data
""")

if __name__ == '__main__':
    main()

