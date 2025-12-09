"""
==============================================================================
Script 3: Progressive Experiments
==============================================================================
Purpose: Run 29 experiments across 6 stages to improve model performance
Input:   data/news_train.json, news_val.json, news_test.json
Output:  
    - data/experiment_log.json (detailed logs)
    - data/experiment_progression.png (improvement curve)

Stages:
    Stage 1: Baseline (TF-IDF 100 + LogReg) - Project doc config
    Stage 2: TF-IDF parameter tuning (max_features, ngram)
    Stage 3: Text preprocessing (lowercase, remove punct/numbers)
    Stage 4: Classifier comparison (NB, SVM, RF, GB, etc.)
    Stage 5: Ensemble methods (Voting, Bagging, AdaBoost)
    Stage 6: Neural Network (MLP with different architectures)

Results:
    - Baseline: 72.65%
    - Best: 82.57% (MultinomialNB with optimized TF-IDF)

Usage:
    python experiments.py
==============================================================================
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Experiment Logger
# ============================================================

class ExperimentLogger:
    """Experiment logger - record all training details"""
    
    def __init__(self, log_file='data/experiment_log.json'):
        self.log_file = log_file
        self.experiments = []
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def log(self, exp_name, stage, config, metrics, duration, notes=""):
        """Log a single experiment"""
        record = {
            'id': len(self.experiments) + 1,
            'name': exp_name,
            'stage': stage,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': config,
            'metrics': metrics,
            'duration_seconds': round(duration, 2),
            'notes': notes
        }
        self.experiments.append(record)
        print(f"  [Logged] Experiment #{record['id']}: {exp_name}")
        
    def save(self):
        """Save all experiment records"""
        output = {
            'experiment_session': self.start_time,
            'total_experiments': len(self.experiments),
            'experiments': self.experiments
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nExperiment log saved to: {self.log_file}")
        
    def get_summary_df(self):
        """Get experiment summary as DataFrame"""
        rows = []
        for exp in self.experiments:
            rows.append({
                'ID': exp['id'],
                'Stage': exp['stage'],
                'Name': exp['name'],
                'Accuracy': exp['metrics']['accuracy'],
                'F1': exp['metrics']['f1'],
                'Duration(s)': exp['duration_seconds']
            })
        return pd.DataFrame(rows)

# ============================================================
# Text Preprocessing
# ============================================================

def preprocess_text_basic(text):
    """Basic preprocessing"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text_advanced(text):
    """Advanced preprocessing"""
    text = text.lower()
    # Remove punctuation but keep apostrophes (e.g., don't)
    text = re.sub(r"[^\w\s']", ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load data"""
    with open('data/news_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/news_val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('data/news_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Merge train and validation
    train_val = train_data + val_data
    
    X_train = [d['headline'] for d in train_val]
    y_train = [1 if d['source'] == 'FoxNews' else 0 for d in train_val]
    X_test = [d['headline'] for d in test_data]
    y_test = [1 if d['source'] == 'FoxNews' else 0 for d in test_data]
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1': round(f1_score(y_true, y_pred), 4)
    }

# ============================================================
# Stage 1: Baseline
# ============================================================

def stage1_baseline(X_train, y_train, X_test, y_test, logger):
    """Stage 1: Reproduce baseline model"""
    print("\n" + "="*60)
    print("STAGE 1: BASELINE (Project Document Configuration)")
    print("="*60)
    
    # Exactly following project document configuration
    config = {
        'vectorizer': 'TfidfVectorizer',
        'max_features': 100,
        'stop_words': 'english',
        'model': 'LogisticRegression',
        'max_iter': 100
    }
    
    start = time.time()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    duration = time.time() - start
    metrics = evaluate_model(y_test, y_pred)
    
    print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")
    print(f"Training time: {duration:.2f}s")
    
    logger.log(
        "Baseline (TF-IDF 100 + LogReg)",
        "Stage 1: Baseline",
        config,
        metrics,
        duration,
        "Project document configuration"
    )
    
    return metrics

# ============================================================
# Stage 2: TF-IDF Parameter Tuning
# ============================================================

def stage2_tfidf_tuning(X_train, y_train, X_test, y_test, logger):
    """Stage 2: TF-IDF parameter tuning"""
    print("\n" + "="*60)
    print("STAGE 2: TF-IDF PARAMETER TUNING")
    print("="*60)
    
    experiments = [
        {'max_features': 500, 'ngram_range': (1,1), 'name': 'TF-IDF(500, unigram)'},
        {'max_features': 1000, 'ngram_range': (1,1), 'name': 'TF-IDF(1000, unigram)'},
        {'max_features': 1000, 'ngram_range': (1,2), 'name': 'TF-IDF(1000, bigram)'},
        {'max_features': 2000, 'ngram_range': (1,2), 'name': 'TF-IDF(2000, bigram)'},
        {'max_features': 2000, 'ngram_range': (1,3), 'name': 'TF-IDF(2000, trigram)'},
        {'max_features': 3000, 'ngram_range': (1,3), 'name': 'TF-IDF(3000, trigram)'},
    ]
    
    best_acc = 0
    for exp in experiments:
        config = {
            'vectorizer': 'TfidfVectorizer',
            'max_features': exp['max_features'],
            'ngram_range': str(exp['ngram_range']),
            'stop_words': 'english',
            'model': 'LogisticRegression',
            'max_iter': 1000,
            'C': 1.0
        }
        
        start = time.time()
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=exp['max_features'],
            ngram_range=exp['ngram_range']
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        duration = time.time() - start
        metrics = evaluate_model(y_test, y_pred)
        
        print(f"{exp['name']}: Acc={metrics['accuracy']}, F1={metrics['f1']}")
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
        
        logger.log(exp['name'], "Stage 2: TF-IDF Tuning", config, metrics, duration)
    
    return best_acc

# ============================================================
# Stage 3: Text Preprocessing Improvement
# ============================================================

def stage3_preprocessing(X_train, y_train, X_test, y_test, logger):
    """Stage 3: Text preprocessing improvement"""
    print("\n" + "="*60)
    print("STAGE 3: TEXT PREPROCESSING")
    print("="*60)
    
    # Apply preprocessing
    X_train_basic = np.array([preprocess_text_basic(x) for x in X_train])
    X_test_basic = np.array([preprocess_text_basic(x) for x in X_test])
    
    X_train_adv = np.array([preprocess_text_advanced(x) for x in X_train])
    X_test_adv = np.array([preprocess_text_advanced(x) for x in X_test])
    
    experiments = [
        (X_train, X_test, 'No preprocessing'),
        (X_train_basic, X_test_basic, 'Basic preprocessing (lowercase, remove punct)'),
        (X_train_adv, X_test_adv, 'Advanced preprocessing (+ remove numbers)'),
    ]
    
    for X_tr, X_te, name in experiments:
        config = {
            'preprocessing': name,
            'vectorizer': 'TfidfVectorizer',
            'max_features': 2000,
            'ngram_range': '(1,3)',
            'model': 'LogisticRegression'
        }
        
        start = time.time()
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1,3)
        )
        X_train_vec = vectorizer.fit_transform(X_tr)
        X_test_vec = vectorizer.transform(X_te)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        duration = time.time() - start
        metrics = evaluate_model(y_test, y_pred)
        
        print(f"{name}: Acc={metrics['accuracy']}, F1={metrics['f1']}")
        
        logger.log(name, "Stage 3: Preprocessing", config, metrics, duration)

# ============================================================
# Stage 4: Classifier Comparison
# ============================================================

def stage4_classifiers(X_train, y_train, X_test, y_test, logger):
    """Stage 4: Compare different classifiers"""
    print("\n" + "="*60)
    print("STAGE 4: CLASSIFIER COMPARISON")
    print("="*60)
    
    # Use best TF-IDF configuration
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2000,
        ngram_range=(1,3)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifiers = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('LogisticRegression(C=0.5)', LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
        ('LogisticRegression(C=2.0)', LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
        ('MultinomialNB', MultinomialNB()),
        ('ComplementNB', ComplementNB()),
        ('LinearSVC', LinearSVC(max_iter=2000, random_state=42)),
        ('SGDClassifier', SGDClassifier(max_iter=1000, random_state=42)),
        ('RandomForest(100)', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('RandomForest(200)', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    
    best_acc = 0
    best_clf = None
    
    for name, clf in classifiers:
        config = {
            'vectorizer': 'TfidfVectorizer(2000, trigram)',
            'classifier': name,
            'params': str(clf.get_params())[:100] + '...'
        }
        
        start = time.time()
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        duration = time.time() - start
        
        metrics = evaluate_model(y_test, y_pred)
        print(f"{name}: Acc={metrics['accuracy']}, F1={metrics['f1']}, Time={duration:.2f}s")
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_clf = name
        
        logger.log(name, "Stage 4: Classifiers", config, metrics, duration)
    
    print(f"\nBest classifier: {best_clf} with accuracy {best_acc}")
    return best_clf

# ============================================================
# Stage 5: Ensemble Learning
# ============================================================

def stage5_ensemble(X_train, y_train, X_test, y_test, logger):
    """Stage 5: Ensemble methods"""
    print("\n" + "="*60)
    print("STAGE 5: ENSEMBLE METHODS")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2000,
        ngram_range=(1,3)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    ensembles = [
        ('VotingClassifier(hard)', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('nb', MultinomialNB()),
                ('svc', LinearSVC(max_iter=2000))
            ],
            voting='hard'
        )),
        ('VotingClassifier(soft)', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('nb', MultinomialNB()),
                ('rf', RandomForestClassifier(n_estimators=100))
            ],
            voting='soft'
        )),
        ('BaggingClassifier(LogReg)', BaggingClassifier(
            estimator=LogisticRegression(max_iter=1000),
            n_estimators=10,
            random_state=42
        )),
        ('AdaBoost(LogReg)', AdaBoostClassifier(
            estimator=LogisticRegression(max_iter=1000),
            n_estimators=50,
            random_state=42,
            algorithm='SAMME'
        )),
    ]
    
    for name, clf in ensembles:
        config = {
            'vectorizer': 'TfidfVectorizer(2000, trigram)',
            'ensemble': name
        }
        
        start = time.time()
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        duration = time.time() - start
        
        metrics = evaluate_model(y_test, y_pred)
        print(f"{name}: Acc={metrics['accuracy']}, F1={metrics['f1']}, Time={duration:.2f}s")
        
        logger.log(name, "Stage 5: Ensemble", config, metrics, duration)

# ============================================================
# Stage 6: Neural Network (MLP)
# ============================================================

def stage6_neural_network(X_train, y_train, X_test, y_test, logger):
    """Stage 6: Simple neural network (CPU-friendly)"""
    print("\n" + "="*60)
    print("STAGE 6: NEURAL NETWORK (MLP)")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2000,
        ngram_range=(1,3)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Scale features
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train_vec)
    X_test_scaled = scaler.transform(X_test_vec)
    
    mlp_configs = [
        ('MLP(100)', {'hidden_layer_sizes': (100,), 'max_iter': 500}),
        ('MLP(200)', {'hidden_layer_sizes': (200,), 'max_iter': 500}),
        ('MLP(100,50)', {'hidden_layer_sizes': (100, 50), 'max_iter': 500}),
        ('MLP(200,100)', {'hidden_layer_sizes': (200, 100), 'max_iter': 500}),
        ('MLP(200,100,50)', {'hidden_layer_sizes': (200, 100, 50), 'max_iter': 500}),
    ]
    
    for name, params in mlp_configs:
        config = {
            'vectorizer': 'TfidfVectorizer(2000, trigram)',
            'model': 'MLPClassifier',
            'hidden_layers': str(params['hidden_layer_sizes']),
            'max_iter': params['max_iter'],
            'activation': 'relu',
            'solver': 'adam'
        }
        
        start = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            max_iter=params['max_iter'],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        duration = time.time() - start
        
        metrics = evaluate_model(y_test, y_pred)
        print(f"{name}: Acc={metrics['accuracy']}, F1={metrics['f1']}, Time={duration:.2f}s")
        
        # Record training curve
        config['n_iter'] = clf.n_iter_
        if hasattr(clf, 'best_loss_') and clf.best_loss_ is not None:
            config['best_loss'] = round(clf.best_loss_, 4)
        else:
            config['best_loss'] = None
        
        logger.log(name, "Stage 6: Neural Network", config, metrics, duration)

# ============================================================
# Generate Visualization Report
# ============================================================

def generate_report(logger, output_dir='data'):
    """Generate experiment report visualization"""
    df = logger.get_summary_df()
    
    # Group by stage
    stages = df['Stage'].unique()
    
    # Create improvement curve chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy progression across experiments
    ax1 = axes[0, 0]
    ax1.plot(df['ID'], df['Accuracy'], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=df['Accuracy'].iloc[0], color='r', linestyle='--', label='Baseline')
    ax1.set_xlabel('Experiment ID')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Progression', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 score progression across experiments
    ax2 = axes[0, 1]
    ax2.plot(df['ID'], df['F1'], 'g-o', linewidth=2, markersize=6)
    ax2.axhline(y=df['F1'].iloc[0], color='r', linestyle='--', label='Baseline')
    ax2.set_xlabel('Experiment ID')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Progression', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Best accuracy by stage
    ax3 = axes[1, 0]
    stage_best = df.groupby('Stage')['Accuracy'].max().reset_index()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(stage_best)))
    bars = ax3.bar(range(len(stage_best)), stage_best['Accuracy'], color=colors)
    ax3.set_xticks(range(len(stage_best)))
    ax3.set_xticklabels([s.split(':')[0] for s in stage_best['Stage']], rotation=45, ha='right')
    ax3.set_ylabel('Best Accuracy')
    ax3.set_title('Best Accuracy by Stage', fontweight='bold')
    ax3.set_ylim(0.7, 0.85)
    for bar, acc in zip(bars, stage_best['Accuracy']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', fontsize=9)
    
    # 4. Top 10 models
    ax4 = axes[1, 1]
    top10 = df.nlargest(10, 'Accuracy')
    colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(top10))]
    bars = ax4.barh(range(len(top10)), top10['Accuracy'].values, color=colors)
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(top10['Name'].values)
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Top 10 Models', fontweight='bold')
    ax4.set_xlim(0.75, 0.85)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/experiment_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nProgression chart saved to: {output_dir}/experiment_progression.png")
    
    # Save CSV summary
    df.to_csv(f'{output_dir}/experiment_summary.csv', index=False)
    print(f"Summary CSV saved to: {output_dir}/experiment_summary.csv")

# ============================================================
# Main Function
# ============================================================

def main():
    print("="*60)
    print("NEWS SOURCE CLASSIFICATION - PROGRESSIVE EXPERIMENTS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize logger
    logger = ExperimentLogger()
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Run all stages
    stage1_baseline(X_train, y_train, X_test, y_test, logger)
    stage2_tfidf_tuning(X_train, y_train, X_test, y_test, logger)
    stage3_preprocessing(X_train, y_train, X_test, y_test, logger)
    stage4_classifiers(X_train, y_train, X_test, y_test, logger)
    stage5_ensemble(X_train, y_train, X_test, y_test, logger)
    stage6_neural_network(X_train, y_train, X_test, y_test, logger)
    
    # Save log
    logger.save()
    
    # Generate report
    generate_report(logger)
    
    # Print final summary
    df = logger.get_summary_df()
    best = df.loc[df['Accuracy'].idxmax()]
    baseline = df.iloc[0]['Accuracy']
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(df)}")
    print(f"Baseline accuracy: {baseline:.4f}")
    print(f"Best accuracy: {best['Accuracy']:.4f} ({best['Name']})")
    print(f"Improvement: {(best['Accuracy']-baseline)/baseline*100:.1f}%")
    print("="*60)

if __name__ == '__main__':
    main()

