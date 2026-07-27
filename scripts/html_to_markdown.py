#!/usr/bin/env python3
"""
HTML to Markdown converter for PyTorch blog posts.
Fetches HTML from a URL and converts it to Markdown format.
"""

import requests
from bs4 import BeautifulSoup
import html2text
import re
import sys
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path


def extract_metadata(soup, base_url):
    """Extract metadata from the blog post."""
    metadata = {}
    
    # Title
    title_tag = soup.find('h1')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    
    # Author
    author_tag = soup.find('span', class_='fn')
    if author_tag:
        metadata['author'] = author_tag.get_text(strip=True)
    
    # Date
    date_tag = soup.find('span', class_='date')
    if date_tag:
        metadata['date'] = date_tag.get_text(strip=True)
    
    # Extract images
    images = []
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src:
            # Convert relative URLs to absolute URLs
            if not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            images.append({
                'src': src,
                'alt': img.get('alt', ''),
                'class': img.get('class', '')
            })
    metadata['images'] = images
    
    return metadata


def clean_html_for_markdown(soup):
    """
    Clean HTML elements that should not be in the markdown output.
    """
    # Remove header, footer, navigation, scripts, styles
    for tag in soup.find_all(['header', 'footer', 'nav', 'script', 'style', 'noscript']):
        tag.decompose()
    
    # Remove skip links and other non-content elements
    for element in soup.find_all(class_=lambda x: x and any(c in str(x) for c in ['skip-to-content', 'screen-reader', 'search-', 'footer-', 'header-'])):
        element.decompose()
    
    # Keep only the article content
    article = soup.find('article')
    if article:
        # Create a new soup with only article content
        new_soup = BeautifulSoup('<div></div>', 'html.parser')
        article_copy = article.__copy__()
        new_soup.div.append(article_copy)
        
        # Remove comments section
        for comments in new_soup.find_all(class_=lambda x: x and 'comment' in str(x).lower()):
            comments.decompose()
        
        return new_soup
    else:
        return soup


def convert_html_to_markdown(html_content, base_url):
    """
    Convert HTML content to Markdown format.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract metadata first
    metadata = extract_metadata(soup, base_url)
    
    # Clean HTML for markdown conversion
    cleaned_soup = clean_html_for_markdown(soup)
    
    # Convert to markdown using html2text
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = False
    h2t.body_width = 0  # Don't wrap lines
    h2t.ignore_emphasis = False
    
    markdown_content = h2t.handle(str(cleaned_soup))
    
    # Clean up markdown
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines
    markdown_content = markdown_content.strip()
    
    return markdown_content, metadata


def download_image(img_url, output_dir):
    """
    Download an image and save it to the output directory.
    Returns the local path.
    """
    try:
        response = requests.get(img_url, timeout=30)
        response.raise_for_status()
        
        # Get filename from URL
        parsed_url = urlparse(img_url)
        filename = Path(parsed_url.path).name
        
        # If no filename, generate one
        if not filename or '.' not in filename:
            filename = f"image_{hash(img_url)}.png"
        
        output_path = output_dir / filename
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return str(output_path)
    except Exception as e:
        print(f"Warning: Failed to download image {img_url}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python html_to_markdown.py <url> [output_file]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python html_to_markdown.py https://pytorch.org/blog/post-slug/", file=sys.stderr)
        print("  python html_to_markdown.py https://pytorch.org/blog/post-slug/ output.md", file=sys.stderr)
        sys.exit(1)
    
    url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Fetch HTML
    print(f"Fetching: {url}")
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PyTorchBlogBot/1.0)'
        })
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}", file=sys.stderr)
        sys.exit(1)
    
    html_content = response.text
    
    # Convert to markdown
    markdown_content, metadata = convert_html_to_markdown(html_content, url)
    
    # Output markdown
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"\nMarkdown saved to: {output_file}")
    else:
        print("\n" + "=" * 80)
        print(markdown_content)
        print("=" * 80)
    
    # Output metadata as JSON
    print("\n--- Metadata ---", file=sys.stderr)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
