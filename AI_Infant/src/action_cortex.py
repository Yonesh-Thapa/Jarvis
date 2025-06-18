# full, runnable code here
import queue
from .neural_fabric import NeuralFabric
from .knowledge_oracle import KnowledgeOracle
from .language_cortex import LanguageCortex
from .web_browser import WebBrowser

class ActionCortex:
    def __init__(self, fabric: NeuralFabric, speech_queue: queue.Queue, 
                 oracle: KnowledgeOracle, language_cortex: LanguageCortex, browser: WebBrowser):
        self.fabric = fabric
        self.speech_queue = speech_queue
        self.oracle = oracle
        self.language_cortex = language_cortex
        self.browser = browser
        self._action_registry = {}
        print("ActionCortex initialized.")

    def register_action(self, symbol: str, function):
        pattern = self.fabric.recall(symbol)
        if pattern:
            self._action_registry[symbol] = {"pattern": pattern, "function": function}
            print(f"Action '{symbol}' registered.")

    def _get_most_specific_symbol(self, context_pattern: set) -> str | None:
        if not context_pattern: return None
        best_symbol, max_subset_size = None, 0
        for symbol, pattern in self.fabric.symbol_table.items():
            if symbol.startswith(("action_", "goal_", "state_")): continue
            if pattern.issubset(context_pattern) and len(pattern) > max_subset_size:
                max_subset_size, best_symbol = len(pattern), symbol
        return best_symbol

    def _speak_action(self, context_pattern: set):
        symbol_to_speak = self._get_most_specific_symbol(context_pattern)
        if symbol_to_speak:
            text = symbol_to_speak.replace("_", " ")
            self.speech_queue.put(text)

    def _ask_oracle_action(self, context_pattern: set):
        symbol_to_ask_about = self._get_most_specific_symbol(context_pattern)
        prompt = f"What is a {symbol_to_ask_about.replace('_', ' ')}?" if symbol_to_ask_about else "What is this object?"
        response = self.oracle.query_llm(prompt)
        if response:
            self.language_cortex.perceive_text_block(response)
            self.speech_queue.put(prompt)

    def _search_web_action(self, context_pattern: set):
        symbol_to_search = self._get_most_specific_symbol(context_pattern)
        if not symbol_to_search:
            print("ACTION_FAIL: Search action had no topic in context.")
            return []
        
        query = symbol_to_search.replace('_', ' ').replace('word_', '')
        urls = self.browser.search(query, num_results=1)
        return urls

    def _browse_page_action(self, context_pattern: set):
        url_symbol = self._get_most_specific_symbol(context_pattern) # Assumes context is just the URL symbol
        if not url_symbol:
            print("ACTION_FAIL: Browse action had no URL in context.")
            return None, []
        
        url = url_symbol.replace('url_', '').replace('_', '://').replace('__', '.') # Decode URL from symbol
        text = self.browser.fetch_page_text(url)
        links = self.browser.find_links(url)
        return text, links

    def step(self, fired_uids: set):
        if not fired_uids or not self._action_registry: return None
        for symbol, action_data in self._action_registry.items():
            action_pattern = action_data["pattern"]
            if action_pattern and action_pattern.issubset(fired_uids):
                context = fired_uids - action_pattern
                # Actions can now return results for the planner to use
                return action_data["function"](context)
        return None