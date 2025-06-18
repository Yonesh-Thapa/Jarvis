# full, runnable code here
import queue
from .neural_fabric import NeuralFabric
from .knowledge_oracle import KnowledgeOracle
from .language_cortex import LanguageCortex
from .web_browser import WebBrowser

class ActionCortex:
    def __init__(self, fabric, speech_queue, oracle, language_cortex, browser):
        self.fabric, self.speech_queue, self.oracle, self.language_cortex, self.browser = fabric, speech_queue, oracle, language_cortex, browser
        self._action_registry = {}
        print("ActionCortex initialized.")

    def register_action(self, symbol, function):
        self._action_registry[symbol] = function

    def _get_words_for_pattern(self, context_pattern):
        # ... (unchanged)
        if not context_pattern: return "an unknown concept"
        best_symbol, max_size = None, 0
        for symbol, pattern in self.fabric.symbol_table.items():
            if symbol.startswith(("action_", "goal_", "state_", "url_")): continue
            if pattern.issubset(context_pattern) and len(pattern) > max_size:
                max_size, best_symbol = len(pattern), symbol
        if best_symbol:
            if best_symbol.startswith("event_"): return best_symbol.replace("event_", "").replace("_", " ")
            elif best_symbol.startswith("word_"):
                for word, p_map in self.language_cortex.word_to_pattern_map.items():
                    if p_map == self.fabric.recall(best_symbol): return word
        return "an unknown concept"

    def _ask_oracle_action(self, context_pattern):
        # ... (unchanged)
        topic = self._get_words_for_pattern(context_pattern)
        if topic == "an unknown concept": return
        prompt = f"In one simple sentence, what is {topic}?"; response = self.oracle.query_llm(prompt)
        if response: self.language_cortex.perceive_text_block(response)

    def _search_web_action(self, context_pattern):
        # ... (unchanged)
        topic = self._get_words_for_pattern(context_pattern)
        if topic == "an unknown concept": return []
        return self.browser.search(topic, num_results=1)

    def _browse_page_action(self, context_string):
        # This action now receives a simple string, not a pattern
        url = context_string
        text = self.browser.fetch_page_text(url)
        return text, []

    def execute_action(self, plan):
        """Executes a plan formulated by the PlanningCortex."""
        if not plan or 'action' not in plan: return None
        action_symbol = plan['action']
        
        if action_symbol in self._action_registry:
            function = self._action_registry[action_symbol]
            # Pass the correct type of context to the function
            if 'context_pattern' in plan:
                return function(plan['context_pattern'])
            elif 'context_string' in plan:
                return function(plan['context_string'])
        return None