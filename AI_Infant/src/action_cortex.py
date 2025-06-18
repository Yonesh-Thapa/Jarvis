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

    def _get_words_for_pattern(self, context_pattern: set) -> str:
        if not context_pattern: return "something"
        best_symbol, max_subset_size = None, 0
        for symbol, pattern in self.fabric.symbol_table.items():
            if symbol.startswith(("action_", "goal_", "state_", "url_")): continue
            if pattern.issubset(context_pattern) and len(pattern) > max_subset_size:
                max_subset_size, best_symbol = len(pattern), symbol
        if best_symbol:
            if best_symbol.startswith("event_"): return best_symbol.replace("event_", "").replace("_", " ")
            elif best_symbol.startswith("word_"):
                for word, p_map in self.language_cortex.word_to_pattern_map.items():
                    if p_map == self.fabric.recall(best_symbol): return word
        return "an unknown concept"

    def _ask_oracle_action(self, context_pattern: set):
        topic = self._get_words_for_pattern(context_pattern)
        prompt = f"What is {topic}?"
        response = self.oracle.query_llm(prompt)
        if response:
            self.language_cortex.perceive_text_block(response)
            self.speech_queue.put(prompt)

    def _search_web_action(self, context_pattern: set):
        topic = self._get_words_for_pattern(context_pattern)
        if topic == "an unknown concept":
            print("ACTION_FAIL: Search had no known topic in context.")
            return []
        return self.browser.search(topic, num_results=1)

    def _browse_page_action(self, context_pattern: set):
        # --- THE FIX: Compare the frozenset context to the frozenset map values ---
        # The context_pattern is a frozenset. The language_cortex map also stores frozensets.
        
        url = None
        # Find the URL string by looking for the pattern in the language cortex's authoritative map.
        for word, pattern in self.language_cortex.word_to_pattern_map.items():
            if pattern == context_pattern:
                url = word # The "word" for a URL pattern is the URL symbol itself.
                break
        
        if not url or not url.startswith("url_"):
            print("ACTION_FAIL: Browse action could not resolve URL from context.")
            return None, []
        
        # Decode the URL from its symbol representation.
        url_string = url.replace('url_', '')
        
        text = self.browser.fetch_page_text(url_string)
        return text, []

    def step(self, fired_uids: set):
        if not fired_uids or not self._action_registry: return None
        for symbol, action_data in self._action_registry.items():
            action_pattern = action_data.get("pattern")
            if action_pattern and action_pattern.issubset(fired_uids):
                # The context is correctly calculated as a frozenset here.
                context = frozenset(fired_uids - action_pattern)
                return action_data["function"](context)
        return None