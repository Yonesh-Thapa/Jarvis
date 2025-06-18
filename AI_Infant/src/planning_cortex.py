# full, runnable code here
import random
import time
from .neural_fabric import NeuralFabric
from .logic_cortex import LogicCortex
from .emotion_module import EmotionModule
from .memory_core import MemoryCore
from .action_cortex import ActionCortex
from .language_cortex import LanguageCortex

class PlanningCortex:
    def __init__(self, fabric: NeuralFabric, logic: LogicCortex, emotion: EmotionModule, 
                 memory: MemoryCore, action_cortex: ActionCortex, language_cortex: LanguageCortex):
        self.fabric = fabric
        self.logic = logic
        self.emotion = emotion
        self.memory = memory
        self.action_cortex = action_cortex
        self.language_cortex = language_cortex
        self.goals = {}
        self.active_goal = None
        self.last_plan_time = 0
        self.plan_cooldown = 2
        self.curiosity_queue = []
        self.visited_urls = set()
        self._initialize_goals()
        print("PlanningCortex initialized for self-correcting exploration.")

    def _initialize_goals(self):
        goal_zone_uids = list(self.fabric.zones.get('goal', []))
        if len(goal_zone_uids) < 20:
            print("WARNING: Not enough 'goal' neurons. Planning disabled."); return
        
        neurons = set(goal_zone_uids)
        p_positive_valence = set(random.sample(list(neurons), 5)); neurons -= p_positive_valence
        self.logic.bind_symbol_to_pattern("goal_seek_positive_valence", p_positive_valence)
        self.goals["positive_valence"] = {"pattern": p_positive_valence, "priority": 1}
        
        p_achieved_positive_valence = set(random.sample(list(neurons), 5)); neurons -= p_achieved_positive_valence
        self.logic.bind_symbol_to_pattern("state_is_positive", p_achieved_positive_valence)
        self.emotion.positive_state_pattern = p_achieved_positive_valence
        
        p_resolve_uncertainty = set(random.sample(list(neurons), 5)); neurons -= p_resolve_uncertainty
        self.logic.bind_symbol_to_pattern("goal_resolve_uncertainty", p_resolve_uncertainty)
        self.goals["resolve_uncertainty"] = {"pattern": p_resolve_uncertainty, "priority": 2}
        
        # --- NEW: Meta-memory patterns for plan effectiveness ---
        p_high_effectiveness = set(random.sample(list(neurons), 5))
        self.logic.bind_symbol_to_pattern("meta_high_effectiveness", p_high_effectiveness)

    def _is_uncertain(self, context_pattern: frozenset) -> bool:
        if not context_pattern: return False
        symbol = next((s for s, p in self.fabric.symbol_table.items() if p == context_pattern), None)
        if not symbol or not symbol.startswith("word_"): return False
        num_connections = sum(len(self.fabric.synapses.get(uid, {})) for uid in context_pattern)
        return (num_connections / len(context_pattern)) < 2 if context_pattern else False

    def add_curiosity_targets(self, perceived_patterns: set):
        new_curiosities = 0
        for p in perceived_patterns:
            if self._is_uncertain(p) and p not in self.curiosity_queue:
                self.curiosity_queue.append(p)
                new_curiosities += 1
        print(f"  - Found {new_curiosities} new points of uncertainty to explore.")

    def _assess_plan_effectiveness(self, action_pattern, concepts_before, concepts_after):
        uncertainty_reduction = concepts_before - concepts_after
        if uncertainty_reduction > 2: # If the plan resolved more than 2 points of curiosity
            print(f"  - META: Action was highly effective! (Reduced uncertainty by {uncertainty_reduction})")
            # Create a memory linking this action to being effective
            self.fabric.relation.create_and_integrate_relation(
                action_pattern,
                self.fabric.recall("goal_resolve_uncertainty"), # a placeholder for "results in"
                self.fabric.recall("meta_high_effectiveness")
            )

    def step(self):
        self._evaluate_goals()
        if self.active_goal and (time.time() - self.last_plan_time > self.plan_cooldown):
            self._formulate_and_execute_plan()
            self.last_plan_time = time.time()

    def _evaluate_goals(self):
        self.active_goal = None
        if self.curiosity_queue: self.active_goal = "resolve_uncertainty"
        elif self.emotion.get_current_valence() < 0.1: self.active_goal = "positive_valence"

    def _formulate_and_execute_plan(self):
        if self.active_goal != "resolve_uncertainty" or not self.curiosity_queue: return
        
        topic_pattern = self.curiosity_queue.pop(0)
        topic_symbol = self.fabric.relation._get_symbol_for_pattern(topic_pattern)
        print(f"\n--- PLAN: Resolving uncertainty for '{topic_symbol}' ---")
        
        concepts_before = len(self.curiosity_queue)

        # --- SELF-CORRECTION: Choose the best strategy ---
        search_action = self.fabric.recall("action_search_web")
        oracle_action = self.fabric.recall("action_ask_oracle")

        search_eff = self.logic.query_association("action_search_web", "meta_high_effectiveness")
        oracle_eff = self.logic.query_association("action_ask_oracle", "meta_high_effectiveness")
        
        chosen_action_pattern, action_symbol_str = (search_action, "action_search_web") if search_eff >= oracle_eff else (oracle_action, "action_ask_oracle")
        print(f"  - Strategy chosen: '{action_symbol_str}' (Effectiveness score: {max(search_eff, oracle_eff):.2f})")

        thought = chosen_action_pattern.union(topic_pattern)
        self.fabric.activate_pattern(thought, 1.1)
        result = self.action_cortex.step(self.fabric.step_simulation())

        # If we searched and got a URL, we now browse it
        if action_symbol_str == "action_search_web" and result:
            url = result[0]
            if url in self.visited_urls: print(f"  - Already visited {url}."); return
            self.visited_urls.add(url)
            
            _, url_symbol = self.language_cortex._get_or_create_pattern_for_word(f"url_{url}")
            url_pattern, _ = self.language_cortex._get_or_create_pattern_for_word(url_symbol)

            browse_action = self.fabric.recall("action_browse_page")
            thought = browse_action.union(url_pattern)
            self.fabric.activate_pattern(thought, 1.1)
            text, _ = self.action_cortex.step(self.fabric.step_simulation())
            
            if text:
                perceived_concepts, main_idea = self.language_cortex.perceive_text_block(text)
                if main_idea: self.logic.integrate_textual_knowledge(main_idea, perceived_concepts)
                self.add_curiosity_targets(perceived_concepts)
        
        concepts_after = len(self.curiosity_queue)
        self._assess_plan_effectiveness(chosen_action_pattern, concepts_before, concepts_after)