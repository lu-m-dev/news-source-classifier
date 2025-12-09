"""
==============================================================================
Script 2: Data Preprocessing
==============================================================================
Purpose: Clean, analyze, and split the scraped dataset
Input:   data/news_data_clean_v2.json
Output:  
    - data/news_train.json (80%)
    - data/news_val.json (10%)
    - data/news_test.json (10%)
    - data/news_data_processed.json/csv
    - data/data_analysis.png

Steps:
    1. Remove duplicates
    2. Clean text (HTML entities, whitespace, etc.)
    3. Analyze data distribution
    4. Generate visualizations
    5. Split into train/val/test

Usage:
    python data_preprocessing.py
==============================================================================
"""

import json
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    """Load JSON data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {filepath}")
    return data

def clean_headline(text):
    """Clean a single headline"""
    if not text or not isinstance(text, str):
        return None
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common website suffixes
    text = re.sub(r'\s*[-|]\s*(Fox News|NBC News|NBCNEWS|FoxNews).*$', '', text, flags=re.IGNORECASE)
    
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.!?,])\1+', r'\1', text)
    
    # Remove leading/trailing whitespace again
    text = text.strip()
    
    # If too short, consider invalid
    if len(text) < 10:
        return None
    
    return text

def remove_duplicates(data):
    """Remove duplicate data"""
    seen_headlines = set()
    unique_data = []
    duplicates = 0
    
    for item in data:
        headline = item.get('headline', '').lower().strip()
        if headline and headline not in seen_headlines:
            seen_headlines.add(headline)
            unique_data.append(item)
        else:
            duplicates += 1
    
    print(f"Removed {duplicates} duplicate headlines")
    return unique_data

def preprocess_data(data):
    """Preprocess all data"""
    processed = []
    removed = 0
    
    for item in data:
        headline = clean_headline(item.get('headline'))
        if headline:
            processed.append({
                'headline': headline,
                'source': item.get('source'),
                'label': 1 if item.get('source') == 'FoxNews' else 0  # FoxNews=1, NBC=0
            })
        else:
            removed += 1
    
    print(f"Removed {removed} invalid headlines during cleaning")
    return processed

def analyze_data(data):
    """Analyze data statistics"""
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n[Basic Statistics]")
    print(f"   Total samples: {len(df)}")
    
    # Source distribution
    source_counts = df['source'].value_counts()
    print(f"\n[Source Distribution]")
    for source, count in source_counts.items():
        pct = count / len(df) * 100
        print(f"   {source}: {count} ({pct:.2f}%)")
    
    # Headline length analysis
    df['headline_length'] = df['headline'].str.len()
    df['word_count'] = df['headline'].str.split().str.len()
    
    print(f"\n[Headline Length (characters)]")
    print(f"   Mean: {df['headline_length'].mean():.1f}")
    print(f"   Median: {df['headline_length'].median():.1f}")
    print(f"   Min: {df['headline_length'].min()}")
    print(f"   Max: {df['headline_length'].max()}")
    
    print(f"\n[Word Count]")
    print(f"   Mean: {df['word_count'].mean():.1f}")
    print(f"   Median: {df['word_count'].median():.1f}")
    print(f"   Min: {df['word_count'].min()}")
    print(f"   Max: {df['word_count'].max()}")
    
    # Statistics by source
    print(f"\n[Statistics by Source]")
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        print(f"\n   {source}:")
        print(f"      Avg length: {source_df['headline_length'].mean():.1f} chars")
        print(f"      Avg words: {source_df['word_count'].mean():.1f} words")
    
    return df

def create_visualizations(df, output_dir='data'):
    """Create visualization charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Source distribution pie chart
    ax1 = axes[0, 0]
    source_counts = df['source'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('News Source Distribution', fontsize=12, fontweight='bold')
    
    # 2. Headline length distribution histogram
    ax2 = axes[0, 1]
    for source, color in zip(['FoxNews', 'NBC'], colors):
        source_df = df[df['source'] == source]
        ax2.hist(source_df['headline_length'], bins=30, alpha=0.6, 
                label=source, color=color, edgecolor='white')
    ax2.set_xlabel('Headline Length (characters)')
    ax2.set_ylabel('Count')
    ax2.set_title('Headline Length Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Word count box plot
    ax3 = axes[1, 0]
    fox_words = df[df['source'] == 'FoxNews']['word_count']
    nbc_words = df[df['source'] == 'NBC']['word_count']
    bp = ax3.boxplot([fox_words, nbc_words], labels=['FoxNews', 'NBC'], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors[0])
    bp['boxes'][1].set_facecolor(colors[1])
    ax3.set_ylabel('Word Count')
    ax3.set_title('Word Count by Source', fontsize=12, fontweight='bold')
    
    # 4. Source bar chart
    ax4 = axes[1, 1]
    bars = ax4.bar(source_counts.index, source_counts.values, color=colors, edgecolor='white')
    ax4.set_ylabel('Count')
    ax4.set_title('Sample Count by Source', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, source_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Visualization saved to {output_dir}/data_analysis.png]")

def get_common_words(df, n=20):
    """Get most common words"""
    print(f"\n[Top {n} Most Common Words by Source]")
    
    for source in ['FoxNews', 'NBC']:
        source_df = df[df['source'] == source]
        all_words = ' '.join(source_df['headline']).lower()
        
        # Simple tokenization (remove punctuation)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_words)
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 
                    'has', 'have', 'was', 'were', 'will', 'can', 'its', 'his', 
                    'her', 'they', 'their', 'what', 'how', 'who', 'says', 'said',
                    'after', 'about', 'over', 'into', 'new', 'out', 'more', 'than'}
        words = [w for w in words if w not in stopwords]
        
        word_counts = Counter(words).most_common(n)
        
        print(f"\n   {source}:")
        for word, count in word_counts[:10]:
            print(f"      {word}: {count}")

def save_processed_data(data, output_dir='data'):
    """Save processed data"""
    # Save as JSON
    json_path = f'{output_dir}/news_data_processed.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved processed data to {json_path}")
    
    # Save as CSV (for easy viewing)
    df = pd.DataFrame(data)
    csv_path = f'{output_dir}/news_data_processed.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved processed data to {csv_path}")
    
    # Create train/val/test splits
    from sklearn.model_selection import train_test_split
    
    # Stratified split by source
    train_data, temp_data = train_test_split(data, test_size=0.2, 
                                              stratify=[d['source'] for d in data],
                                              random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5,
                                            stratify=[d['source'] for d in temp_data],
                                            random_state=42)
    
    # Save split data
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(f"\n[Dataset Split]")
    for split_name, split_data in splits.items():
        split_path = f'{output_dir}/news_{split_name}.json'
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        fox_count = sum(1 for d in split_data if d['source'] == 'FoxNews')
        nbc_count = sum(1 for d in split_data if d['source'] == 'NBC')
        print(f"   {split_name}: {len(split_data)} samples (Fox: {fox_count}, NBC: {nbc_count})")
    
    return splits

def main():
    # Configuration - use updated data file
    INPUT_FILE = 'data/news_data_clean_v2.json'
    OUTPUT_DIR = 'data'
    
    print("="*60)
    print("NEWS HEADLINES DATA PREPROCESSING")
    print("="*60)
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    data = load_data(INPUT_FILE)
    
    # 2. Remove duplicates
    print("\n[Step 2] Removing duplicates...")
    data = remove_duplicates(data)
    
    # 3. Clean data
    print("\n[Step 3] Cleaning headlines...")
    data = preprocess_data(data)
    
    # 4. Analyze data
    print("\n[Step 4] Analyzing data...")
    df = analyze_data(data)
    
    # 5. Common words analysis
    get_common_words(df)
    
    # 6. Create visualizations
    print("\n[Step 5] Creating visualizations...")
    create_visualizations(df, OUTPUT_DIR)
    
    # 7. Save processed data
    print("\n[Step 6] Saving processed data...")
    splits = save_processed_data(data, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nFinal dataset: {len(data)} samples")
    print(f"   - Training: {len(splits['train'])} samples")
    print(f"   - Validation: {len(splits['val'])} samples")
    print(f"   - Test: {len(splits['test'])} samples")

if __name__ == '__main__':
    main()

