# full, runnable code here
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebBrowser:
    """
    A lightweight, text-only web browser for the AI. It uses a reliable,
    professional Search API instead of brittle scraping.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # --- NEW: API Key for Brave Search ---
        self.search_api_key = os.environ.get("BRAVE_API_KEY")
        if not self.search_api_key:
            print("WARNING: BRAVE_API_KEY environment variable not set. Web search will be disabled.")
        print("WebBrowser initialized for API-based search.")

    def _is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme) and parsed.scheme in ['http', 'httpss']

    def fetch_page_text(self, url: str) -> str | None:
        print(f"BROWSER: Fetching text from {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                script_or_style.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Could not fetch URL {url}. Reason: {e}")
            return None

    def find_links(self, base_url: str) -> list[str]:
        # This function is unchanged but may be used less frequently
        print(f"BROWSER: Finding links on {base_url}")
        try:
            response = requests.get(base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            for a_tag in soup.find_all('a', href=True):
                absolute_url = urljoin(base_url, a_tag['href'])
                if self._is_valid_url(absolute_url):
                    links.append(absolute_url)
            return list(set(links))
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Could not fetch URL {base_url} to find links. Reason: {e}")
            return []

    # --- START OF FINAL FIX: USE BRAVE SEARCH API ---
    def search(self, query: str, num_results: int = 1) -> list[str]:
        """
        Performs a web search using the Brave Search API.
        """
        if not self.search_api_key:
            print("BROWSER_ERROR: Search failed, BRAVE_API_KEY not set.")
            return []

        print(f"BROWSER: Searching for '{query}' using Brave API...")
        search_url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.search_api_key
        }
        params = {"q": query, "count": num_results}
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract URLs from the 'web' results
            results = [item['url'] for item in data.get('web', {}).get('results', []) if 'url' in item]

            if results:
                print(f"  - Search successful. Found URL: {results[0]}")
            else:
                print(f"  - Search API returned no valid results.")
            
            return results[:num_results]
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Search API request failed for query '{query}'. Reason: {e}")
            return []
    # --- END OF FINAL FIX ---