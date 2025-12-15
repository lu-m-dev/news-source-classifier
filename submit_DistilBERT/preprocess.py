"""
Preprocess for LLM submission (submit_llm)
=========================================
Functions to convert URL CSV -> pseudo-headlines compatible with DistilBERT training.
This mirrors the interface used in submit2 (prepare_data(path) -> (X, y)).
"""

import pandas as pd
import re
from typing import List, Tuple
from urllib.parse import urlparse, unquote


def url_to_pseudo_headline(url: str) -> str:
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
    if not url:
        return ""
    url_lower = url.lower()
    if 'foxnews.com' in url_lower:
        return 'FoxNews'
    elif 'nbcnews.com' in url_lower or 'msnbc.com' in url_lower:
        return 'NBC'
    return ""


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Read a CSV compatible with `url_only_data.csv` and return (X, y)
    where X is list of pseudo-headlines (strings) and y is list of labels.

    This function intentionally makes NO HTTP requests and extracts
    pseudo-headlines from URLs using string processing only.
    """
    df = pd.read_csv(path)

    url_cols = ['url', 'URL', 'link', 'urls', 'links']
    url_col = None
    for col in url_cols:
        if col in df.columns:
            url_col = col
            break

    headline_cols = ['headline', 'scraped_headline', 'alternative_headline', 'title', 'text']
    headline_col = None
    for col in headline_cols:
        if col in df.columns:
            headline_col = col
            break

    X = []
    y = []

    if url_col is not None:
        for idx, row in df.iterrows():
            url = str(row[url_col])
            pseudo_headline = url_to_pseudo_headline(url)
            pseudo_headline = clean_text(pseudo_headline)
            if len(pseudo_headline) < 5:
                continue
            X.append(pseudo_headline)

            label = ""
            if 'source' in df.columns:
                label = str(row['source'])
            elif 'label' in df.columns:
                label = str(row['label'])
            else:
                label = identify_source_from_url(url)

            y.append(label)

    elif headline_col is not None:
        for idx, row in df.iterrows():
            headline = clean_text(str(row[headline_col]))
            if len(headline) < 5:
                continue
            X.append(headline)
            label = ""
            if 'source' in df.columns:
                label = str(row['source'])
            elif 'label' in df.columns:
                label = str(row['label'])
            y.append(label)

    else:
        first_col = df.columns[0]
        for idx, row in df.iterrows():
            val = str(row[first_col])
            if val.startswith('http'):
                pseudo_headline = url_to_pseudo_headline(val)
                pseudo_headline = clean_text(pseudo_headline)
            else:
                pseudo_headline = clean_text(val)
            if len(pseudo_headline) < 5:
                continue
            X.append(pseudo_headline)
            y.append(identify_source_from_url(val))

    return X, y


if __name__ == '__main__':
    print('Testing prepare_data on url_only_data.csv (if present)')
    try:
        X, y = prepare_data('../url_only_data.csv')
        print(f'Extracted {len(X)} examples')
        for i in range(min(3, len(X))):
            print(y[i], X[i])
    except Exception as e:
        print('Error:', e)
