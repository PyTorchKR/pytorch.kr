#!/usr/bin/env python3
"""Check generated community pages and local destinations without network access."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import sys

ROOT = Path(__file__).resolve().parent.parent / '_site'
BASE_URL = 'https://pytorch.kr/'


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.h1_count, self.duplicates = set(), [], 0, []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        self.h1_count += tag == 'h1'
        for key in ('href', 'src'):
            if attrs.get(key):
                self.links.append(attrs[key])


def check():
    pages = [ROOT / 'index.html']
    for name in ('projects', 'groups', 'events'):
        pages.extend(sorted((ROOT / name).rglob('*.html')))
    errors, cache, links_checked = [], {}, 0
    for path in pages:
        if not path.exists():
            errors.append(f'Missing build output: {path}')
            continue
        page = cache.setdefault(path, Page(path))
        relative = path.relative_to(ROOT).as_posix()
        if page.duplicates:
            errors.append(f'{relative}: duplicate IDs {page.duplicates}')
        if page.h1_count != 1:
            errors.append(f'{relative}: expected one h1, found {page.h1_count}')
        for link in page.links:
            url = urlsplit(urljoin(BASE_URL + relative, link))
            if url.netloc != 'pytorch.kr' or url.scheme not in ('http', 'https'):
                continue
            target = ROOT / unquote(url.path).lstrip('/')
            if target.is_dir():
                target = target / 'index.html'
            links_checked += 1
            if not target.is_file():
                errors.append(f'{relative}: missing {link}')
            elif url.fragment and target.suffix == '.html':
                target_page = cache.setdefault(target, Page(target))
                if unquote(url.fragment) not in target_page.ids:
                    errors.append(f'{relative}: missing anchor {link}')
    if errors:
        sys.exit('\n'.join(sorted(set(errors))))
    print(f'Generated pages: {len(pages)}; local links/assets checked: {links_checked}; no missing destinations, duplicate IDs or h1 errors.')


if __name__ == '__main__':
    check()
