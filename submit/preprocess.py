"""
Preprocess Module for News Headline Classifier
==============================================================================
IMPORTANT: For backend evaluation, NO HTTP requests allowed!
Must convert raw URLs to pseudo-headlines using string processing only.

Input: CSV with URLs (like url_only_data.csv)
Output: (X, y) where X is list of pseudo-headlines extracted from URLs
==============================================================================
"""

import pandas as pd
import re
from typing import List, Tuple
from urllib.parse import urlparse, unquote


def url_to_pseudo_headline(url: str) -> str:
    """
    Convert a URL to a pseudo-headline WITHOUT making HTTP requests.
    
    Example:
        https://www.nbcnews.com/politics/congress/senate-passes-climate-bill-rcna12345
        -> "senate passes climate bill"
        
        https://www.foxnews.com/politics/trump-announces-new-policy
        -> "trump announces new policy"
    """
    if not url or not isinstance(url, str):
        return ""
    
    try:
        # Decode URL encoding
        url = unquote(url)
        
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path
        
        # Remove leading/trailing slashes
        path = path.strip('/')
        
        # Get the last segment (slug)
        segments = path.split('/')
        
        # Find the best segment (usually the last one with meaningful text)
        slug = ""
        for seg in reversed(segments):
            # Skip empty, short, or purely numeric segments
            if seg and len(seg) > 5 and not seg.isdigit():
                slug = seg
                break
        
        if not slug:
            # Fallback: use the last non-empty segment
            for seg in reversed(segments):
                if seg:
                    slug = seg
                    break
        
        # Clean the slug
        # Remove common suffixes like rcna12345, ncna12345, n12345
        slug = re.sub(r'[-.]*(rcna|ncna|n)\d+$', '', slug, flags=re.I)
        
        # Remove .print, .html, .amp suffixes
        slug = re.sub(r'\.(print|html|amp|php)$', '', slug, flags=re.I)
        
        # Remove trailing numbers/IDs
        slug = re.sub(r'[-_]\d+$', '', slug)
        
        # Replace hyphens and underscores with spaces
        headline = re.sub(r'[-_]+', ' ', slug)
        
        # Remove extra whitespace
        headline = re.sub(r'\s+', ' ', headline).strip()
        
        # Basic cleaning
        headline = headline.lower()
        
        return headline
        
    except Exception:
        return ""


def identify_source_from_url(url: str) -> str:
    """
    Identify news source from URL domain.
    """
    if not url:
        return ""
    
    url_lower = url.lower()
    
    if 'foxnews.com' in url_lower:
        return 'FoxNews'
    elif 'nbcnews.com' in url_lower or 'msnbc.com' in url_lower:
        return 'NBC'
    
    return ""


def clean_text(text: str) -> str:
    """
    Clean extracted pseudo-headline for model input.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Already lowercase from url_to_pseudo_headline
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Prepare data from CSV file for model inference.
    
    IMPORTANT: This function must work WITHOUT internet access!
    It extracts pseudo-headlines from URLs using string processing only.
    
    Args:
        path: Path to CSV file containing URLs
        
    Returns:
        X: List of pseudo-headlines (extracted from URLs)
        y: List of labels ('FoxNews' or 'NBC', or empty if not determinable)
    """
    # Read CSV
    df = pd.read_csv(path)
    
    # Find URL column
    url_cols = ['url', 'URL', 'link', 'urls', 'links']
    url_col = None
    for col in url_cols:
        if col in df.columns:
            url_col = col
            break
    
    # If no URL column, check for headline column (for compatibility with our test data)
    headline_cols = ['headline', 'scraped_headline', 'alternative_headline', 'title', 'text']
    headline_col = None
    for col in headline_cols:
        if col in df.columns:
            headline_col = col
            break
    
    X = []
    y = []
    
    if url_col is not None:
        # Process URLs -> pseudo-headlines
        for idx, row in df.iterrows():
            url = str(row[url_col])
            
            # Extract pseudo-headline from URL
            pseudo_headline = url_to_pseudo_headline(url)
            pseudo_headline = clean_text(pseudo_headline)
            
            # Skip if we couldn't extract anything meaningful
            if len(pseudo_headline) < 5:
                continue
            
            X.append(pseudo_headline)
            
            # Try to get label from source column or infer from URL
            label = ""
            if 'source' in df.columns:
                label = str(row['source'])
            elif 'label' in df.columns:
                label = str(row['label'])
            else:
                label = identify_source_from_url(url)
            
            y.append(label)
    
    elif headline_col is not None:
        # Fallback: use headline column directly (for testing with our own data)
        for idx, row in df.iterrows():
            headline = clean_text(str(row[headline_col]))
            if len(headline) < 5:
                continue
            
            X.append(headline)
            
            # Get label
            label = ""
            if 'source' in df.columns:
                label = str(row['source'])
            elif 'label' in df.columns:
                label = str(row['label'])
            
            y.append(label)
    
    else:
        # Try first column as URL
        first_col = df.columns[0]
        for idx, row in df.iterrows():
            url = str(row[first_col])
            
            if url.startswith('http'):
                pseudo_headline = url_to_pseudo_headline(url)
                pseudo_headline = clean_text(pseudo_headline)
            else:
                pseudo_headline = clean_text(url)
            
            if len(pseudo_headline) < 5:
                continue
            
            X.append(pseudo_headline)
            y.append(identify_source_from_url(url))
    
    return X, y


# For local testing
if __name__ == '__main__':
    # Test URL to headline conversion
    test_urls = [
        "https://www.foxnews.com/politics/trump-announces-new-immigration-policy",
        "https://www.nbcnews.com/politics/congress/senate-passes-climate-bill-rcna12345",
        "https://www.foxnews.com/entertainment/taylor-swift-breaks-records.print",
        "https://www.nbcnews.com/news/world/ukraine-russia-conflict-latest-ncna1234567",
    ]
    
    print("URL to Pseudo-Headline Conversion Test:")
    print("="*60)
    for url in test_urls:
        headline = url_to_pseudo_headline(url)
        source = identify_source_from_url(url)
        print(f"URL: {url}")
        print(f"  -> Headline: '{headline}'")
        print(f"  -> Source: {source}")
        print()
