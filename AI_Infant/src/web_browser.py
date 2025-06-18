# full, runnable code here
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv # <<< ADD THIS LINE

class WebBrowser:
    """
    A lightweight, text-only web browser for the AI.
    --- UPGRADE: It now uses a reliable, professional Search API. ---
    """
    def __init__(self):
        load_dotenv() # <<< ADD THIS LINE TO LOAD KEYS FROM .env FILE
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_api_key = os.environ.get("BRAVE_API_KEY")
        if not self.search_api_key:
            print("WARNING: BRAVE_API_KEY environment variable not set. Web search will be disabled.")
        print("WebBrowser initialized for API-based search.")

    # ... the rest of the file remains exactly the same ...
# ...
# ...

    # --- START OF FINAL FIX: USE BRAVE SEARCH API ---
    def search(self, query: str, num_results: int = 1) -> list[str]:
        """
        Performs a web search using the Brave Search API. This is far more
        reliable and professional than scraping a search engine's HTML.
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
            
            results = [item['url'] for item in data.get('web', {}).get('results', []) if 'url' in item]

            if results:
                print(f"  - Search successful. Found URL: {results[0]}")
            else:
                print(f"  - Search API returned no valid results for query '{query}'.")
            
            return results[:num_results]
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Search API request failed for query '{query}'. Reason: {e}")
            return []
    # --- END OF FINAL FIX ---