# News Source Classification

CIS 4190/5190 Applied Machine Learning - Project B  
Classify news headlines as Fox News or NBC News  
Best Accuracy: 82.84%

## Project Structure

```
5190project/
├── news_scraper.py          # Data collection from URLs
├── data_preprocessing.py    # Clean, analyze, split data
├── experiments.py           # 29 progressive experiments
├── analyze_features.py      # Feature importance & robustness
├── complete_experiment.py   # 5 models x 2 datasets comparison
│
├── data/                    # Datasets and results
│   ├── news_train.json      # Training (2977)
│   ├── news_val.json        # Validation (372)
│   ├── news_test.json       # Test (373)
│   └── *.png                # Visualizations
│
├── extra_data/              # Extended dataset (RSS/News feeds)
├── submit/                  # Leaderboard submission v1
├── submit2/                 # Leaderboard submission v2 (best)
└── Project_resource/        # Course materials
```

## Scripts

| Script | Purpose |
|--------|---------|
| news_scraper.py | Scrape headlines from 3800+ URLs |
| data_preprocessing.py | Clean text, remove duplicates, train/val/test split |
| experiments.py | 6 stages: baseline → TF-IDF tuning → classifiers → ensemble → MLP |
| analyze_features.py | Feature weights, robustness test (name removal) |
| complete_experiment.py | Full comparison on original vs extended data |

## Results

| Model | Accuracy |
|-------|----------|
| MLP Neural Network | 82.84% |
| MultinomialNB | 82.57% |
| Logistic Regression | 82.04% |
| LinearSVC | 81.50% |
| Baseline | 66.49% |

Improvement: +24.5% over baseline

## Key Findings

1. Best features: TF-IDF with (1,3)-gram, 5000 features
2. Best classifier: MultinomialNB or MLP
3. Robustness: Only 1% accuracy drop when removing person names
4. Extended data did not improve performance (quality > quantity)

## Quick Start

```bash
pip install -r requirements.txt
python complete_experiment.py
```

## Leaderboard Submission

Best scores (Group 30):
- url_val: 79.08%
- url_val16k: 61.78%

See `submit2/` for submission files.
