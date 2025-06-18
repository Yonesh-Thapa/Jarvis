# full, runnable code here
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebBrowser:
    """
    A lightweight, text-only web browser for the AI.
    It fetches content and finds links, following the 'no data hoarding' principle
    by processing and immediately discarding the raw HTML.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        print("WebBrowser initialized.")

    def _is_valid_url(self, url):
        """Checks if a URL is valid and not a fragment or mailto link."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme) and parsed.scheme in ['http', 'httpss']

    def fetch_page_text(self, url: str) -> str | None:
        """
        Fetches the clean, readable text content from a given URL.

        Returns:
            The text content of the page, or None on failure.
        """
        print(f"BROWSER: Fetching text from {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status() # Raises an exception for bad status codes
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Could not fetch URL {url}. Reason: {e}")
            return None

    def find_links(self, base_url: str) -> list[str]:
        """
        Fetches a page and returns all valid, absolute URLs linked from it.
        """
        print(f"BROWSER: Finding links on {base_url}")
        try:
            response = requests.get(base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Create absolute URL from relative link
                absolute_url = urljoin(base_url, href)
                if self._is_valid_url(absolute_url):
                    links.append(absolute_url)
            
            return list(set(links)) # Return only unique links
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Could not fetch URL {base_url} to find links. Reason: {e}")
            return []

    def search(self, query: str, num_results: int = 1) -> list[str]:
        """
        Performs a web search and returns the top result URLs.
        Uses DuckDuckGo for simplicity and privacy.
        """
        print(f"BROWSER: Searching for '{query}'...")
        search_url = "https://html.duckduckgo.com/html/"
        params = {'q': query}
        try:
            response = requests.post(search_url, headers=self.headers, data=params, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            results = []
            for link in soup.find_all('a', class_='result__a'):
                if len(results) >= num_results:
                    break
                url = link['href']
                # DDG uses a redirect, we need to clean the URL.
                cleaned_url = requests.utils.unquote(url).split("uddg=")[1].split("&rut=")[0]
                if self._is_valid_url(cleaned_url):
                    results.append(cleaned_url)
            
            return results
        except requests.RequestException as e:
            print(f"BROWSER_ERROR: Search failed for query '{query}'. Reason: {e}")
            return []