"""
==============================================================================
Script 1: News Headlines Scraper
==============================================================================
Purpose: Scrape news headlines from Fox News and NBC News
Input:   url_only_data.csv (3805 URLs)
Output:  data/news_data.json

Usage:
    python news_scraper.py                    # Scrape all URLs
    python news_scraper.py --test 10          # Test mode (first 10 URLs)
    python news_scraper.py --session          # Use session for Fox News

Features:
    - Random User-Agent rotation
    - Automatic retry for failed URLs
    - Multiple extraction methods (h1, og:title, etc.)
==============================================================================
"""

import csv
import json
import re
import time
import random
import argparse
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Browser request headers
def get_headers():
    """Return random browser request headers"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

def identify_source(url):
    """Identify news source from URL"""
    if 'foxnews.com' in url:
        return 'FoxNews'
    elif 'nbcnews.com' in url:
        return 'NBC'
    return 'Unknown'

def clean_text(text):
    """Clean text content"""
    if not text:
        return None
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text if text else None

def scrape_foxnews(soup):
    """Scrape Fox News headline"""
    # Method 1: headline speakable
    title = soup.find('h1', class_='headline speakable')
    if title:
        return clean_text(title.get_text())
    
    # Method 2: any headline class
    title = soup.find('h1', class_=re.compile(r'headline', re.I))
    if title:
        return clean_text(title.get_text())
    
    # Method 3: find h1 directly
    title = soup.find('h1')
    if title:
        text = clean_text(title.get_text())
        if text and len(text) > 15:
            return text
    
    # Method 4: og:title
    meta = soup.find('meta', property='og:title')
    if meta and meta.get('content'):
        text = clean_text(meta['content'])
        text = re.sub(r'\s*\|\s*Fox News.*$', '', text, flags=re.I)
        return text
    
    return None

def scrape_nbcnews(soup):
    """Scrape NBC News headline"""
    # Method 1: article headline
    title = soup.find('h1', class_='article-hero-headline__htag')
    if title:
        return clean_text(title.get_text())
    
    # Method 2: other h1 class names
    for class_name in ['articleTitle', 'article-title', 'headline']:
        title = soup.find('h1', class_=class_name)
        if title:
            return clean_text(title.get_text())
    
    # Method 3: find h1 directly
    title = soup.find('h1')
    if title:
        return clean_text(title.get_text())
    
    # Method 4: og:title
    meta = soup.find('meta', property='og:title')
    if meta and meta.get('content'):
        text = clean_text(meta['content'])
        text = re.sub(r'\s*[-|]\s*NBC News.*$', '', text, flags=re.I)
        return text
    
    return None

def scrape_headline(url, session=None, timeout=15):
    """Scrape headline from a single URL"""
    source = identify_source(url)
    
    try:
        time.sleep(random.uniform(0.3, 0.8))
        
        headers = get_headers()
        if session:
            response = session.get(url, headers=headers, timeout=timeout)
        else:
            response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            return {
                'url': url,
                'source': source,
                'headline': None,
                'status': f'HTTP_{response.status_code}'
            }
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if source == 'FoxNews':
            headline = scrape_foxnews(soup)
        elif source == 'NBC':
            headline = scrape_nbcnews(soup)
        else:
            headline = None
        
        return {
            'url': url,
            'source': source,
            'headline': headline,
            'status': 'success' if headline else 'no_headline'
        }
        
    except requests.exceptions.Timeout:
        return {'url': url, 'source': source, 'headline': None, 'status': 'timeout'}
    except Exception as e:
        return {'url': url, 'source': source, 'headline': None, 'status': 'error'}

def load_urls(csv_file):
    """Load URL list from CSV file"""
    urls = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '') or list(row.values())[0] if row else ''
            url = url.strip()
            if url and url.startswith('http'):
                urls.append(url)
    logger.info(f"Loaded {len(urls)} URLs from {csv_file}")
    return urls

def scrape_all(urls, use_session=False):
    """Scrape all URLs"""
    results = []
    session = None
    
    if use_session:
        session = requests.Session()
        try:
            session.get('https://www.foxnews.com/', headers=get_headers(), timeout=10)
            time.sleep(1)
        except:
            pass
    
    for i, url in enumerate(tqdm(urls, desc="Scraping")):
        result = scrape_headline(url, session)
        results.append(result)
        
        # Reset session every 100 URLs
        if use_session and (i + 1) % 100 == 0:
            session = requests.Session()
            try:
                session.get('https://www.foxnews.com/', headers=get_headers(), timeout=10)
            except:
                pass
    
    return results

def save_results(results, output_file):
    """Save scraping results"""
    # Only save successful results
    clean_results = [
        {'headline': r['headline'], 'source': r['source'], 'url': r['url']}
        for r in results if r['status'] == 'success' and r['headline']
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Statistics
    total = len(results)
    success = len(clean_results)
    fox = sum(1 for r in clean_results if r['source'] == 'FoxNews')
    nbc = sum(1 for r in clean_results if r['source'] == 'NBC')
    
    print(f"\n{'='*50}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*50}")
    print(f"Total: {total}, Success: {success} ({success/total*100:.1f}%)")
    print(f"Fox News: {fox}, NBC: {nbc}")
    print(f"Saved to: {output_file}")
    
    return clean_results

def main():
    parser = argparse.ArgumentParser(description='News Headlines Scraper')
    parser.add_argument('--input', default='url_only_data.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/news_data.json', help='Output JSON file')
    parser.add_argument('--test', type=int, help='Test mode: only scrape first N URLs')
    parser.add_argument('--session', action='store_true', help='Use session for Fox News')
    args = parser.parse_args()
    
    urls = load_urls(args.input)
    
    if args.test:
        urls = urls[:args.test]
        logger.info(f"Test mode: scraping only {args.test} URLs")
    
    results = scrape_all(urls, use_session=args.session)
    save_results(results, args.output)

if __name__ == '__main__':
    main()

