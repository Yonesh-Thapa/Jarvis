# full, runnable code here
import queue
from .neural_fabric import NeuralFabric
from .knowledge_oracle import KnowledgeOracle
from .language_cortex import LanguageCortex
from .web_browser import WebBrowser

class ActionCortex:
    def __init__(self, fabric: NeuralFabric, speech_queue: queue.Queue, oracle: KnowledgeOracle, language_cortex: LanguageCortex, browser: WebBrowser):
        self.fabric, self.speech_queue, self.oracle, self.language_cortex, self.browser = fabric, speech_queue, oracle, language_cortex, browser
        self._action_registry = {}
        print("ActionCortex initialized.")

    def register_action(self, symbol: str, function):
        self._action_registry[symbol] = function

    def _get_words_for_pattern(self, context_pattern: frozenset) -> str:
        """Finds the most likely word/concept for a given neural pattern."""
        if not context_pattern: return "an unknown concept"
        
        # --- FIX: Use the authoritative function in relational_cortex ---
        # This is more efficient and reliable than searching the whole symbol table here.
        return self.fabric.relation._get_symbol_for_pattern(context_pattern, default="unknown concept")

    def _ask_oracle_action(self, context_pattern: frozenset):
        """Action: Asks the KnowledgeOracle about a topic pattern."""
        topic = self._get_words_for_pattern(context_pattern)
        if topic == "an unknown concept": return
        prompt = f"In one simple sentence, what is {topic}?"; response = self.oracle.query_llm(prompt)
        if response: self.language_cortex.perceive_text_block(response)

    def _search_web_action(self, context_pattern: frozenset) -> list[str]:
        """Action: Searches the web for a topic pattern."""
        topic = self._get_words_for_pattern(context_pattern)
        if topic == "an unknown concept": return []
        return self.browser.search(topic, num_results=1)

    def _browse_page_action(self, context_string: str) -> tuple[str | None, list]:
        """Action: Fetches text from a URL string."""
        url = context_string
        text = self.browser.fetch_page_text(url)
        # Returns the text and an empty list (formerly for links)
        return text, []

    def execute_action(self, plan: dict) -> any:
        """Executes a plan formulated by the PlanningCortex."""
        if not plan or 'action' not in plan: return None
        action_symbol = plan['action']
        
        if action_symbol in self._action_registry:
            function = self._action_registry[action_symbol]
            # --- FIX: Pass the correct type of context to the function ---
            # Some actions operate on neural patterns, others on simple strings (like URLs).
            if 'context_pattern' in plan:
                return function(plan['context_pattern'])
            elif 'context_string' in plan:
                return function(plan['context_string'])
            # Add a default case for actions that might not need context
            else:
                return function() 
                
        print(f"ACTION_CORTEX_WARN: Unknown action symbol '{action_symbol}' in plan.")
        return None