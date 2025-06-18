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
        self.fabric, self.logic, self.emotion, self.memory, self.action_cortex, self.language_cortex = fabric, logic, emotion, memory, action_cortex, language_cortex
        self.goals, self.active_goal, self.last_plan_time, self.plan_cooldown = {}, None, 0, 2
        self.current_plan, self.curiosity_queue, self.visited_urls = {}, [], set()
        self._initialize_goals()
        print("PlanningCortex initialized for explicit action formulation.")

    def _initialize_goals(self):
        goal_zone_uids = list(self.fabric.zones.get('goal', []));
        if len(goal_zone_uids) < 20: print("WARNING: Not enough 'goal' neurons."); return
        neurons = set(goal_zone_uids)
        p_pos = set(random.sample(list(neurons), 5)); neurons -= p_pos; self.fabric.bind("goal_seek_positive_valence", p_pos)
        p_achieved = set(random.sample(list(neurons), 5)); neurons -= p_achieved; self.fabric.bind("state_is_positive", p_achieved); self.emotion.positive_state_pattern = p_achieved
        p_resolve = set(random.sample(list(neurons), 5)); neurons -= p_resolve; self.fabric.bind("goal_resolve_uncertainty", p_resolve)
        p_effective = set(random.sample(list(neurons), 5)); self.fabric.bind("meta_high_effectiveness", p_effective)

    def _is_uncertain(self, p: frozenset) -> bool:
        """Determines if a concept is "uncertain" (has few connections)."""
        if not p: return False
        s = self.fabric.relation._get_symbol_for_pattern(p)
        if not s: return False # Not a recognized concept
        
        # A concept is uncertain if its neurons have very few connections on average
        conns = sum(len(self.fabric.synapses.get(uid, {})) for uid in p)
        return (conns / len(p)) < 2.5 if p else False # Increased threshold slightly

    def add_curiosity_targets(self, patterns: set):
        """Adds uncertain concepts to the curiosity queue to be researched."""
        new = 0
        for p in patterns:
            if self._is_uncertain(p) and p not in self.curiosity_queue:
                self.curiosity_queue.append(p); new += 1
        if new > 0:
            print(f"PLANNING: Found {new} new points of uncertainty to explore.")

    def step(self):
        self._evaluate_goals()
        if self.active_goal and (time.time() - self.last_plan_time > self.plan_cooldown):
            next_action = self._formulate_next_action()
            self.last_plan_time = time.time()
            return next_action
        return None

    def _evaluate_goals(self):
        if self.current_plan: self.active_goal = self.current_plan['type']; return
        self.active_goal = None
        if self.curiosity_queue:
            self.active_goal = "research"
            self.current_plan = {'type': 'research', 'topic_pattern': self.curiosity_queue.pop(0), 'state': 'start'}
        elif self.emotion.get_current_valence() < -0.1: self.active_goal = "positive_valence"

    def _formulate_next_action(self):
        if not self.current_plan: return None
        state = self.current_plan.get('state')
        topic_pattern = self.current_plan.get('topic_pattern')
        topic_word = self.action_cortex._get_words_for_pattern(topic_pattern)
        print(f"\n--- PLAN: Formulating action for '{topic_word}'. State: {state} ---")
        if state == 'start':
            action_sym = "action_search_web"
            print(f"  - Strategy chosen: '{action_sym}'")
            return {'action': action_sym, 'context_pattern': topic_pattern}
        elif state == 'has_url':
            url = self.current_plan['url']
            if url in self.visited_urls:
                print(f"  - Aborting plan: Already visited {url}."); self.current_plan = {}; return None
            self.visited_urls.add(url)
            return {'action': 'action_browse_page', 'context_string': url}
        return None

    def update_plan_with_result(self, result: any):
        if not self.current_plan: return
        state = self.current_plan.get('state')
        print(f"--- PLAN: Updating plan with result. State was: {state} ---")
        if state == 'start': # Result of a web search
            if result and isinstance(result, list) and len(result) > 0:
                self.current_plan['state'] = 'has_url'
                self.current_plan['url'] = result[0]
                print(f"  - Plan state advanced to 'has_url' with URL: {result[0]}")
            else:
                print("  - Plan failed: Search returned no URLs."); self.current_plan = {}
        elif state == 'has_url': # Result of browsing a page
            text, _ = result if isinstance(result, tuple) else (None, None)
            if text:
                # --- THE FIX: The result of perception is new potential curiosity targets ---
                # The language cortex itself handles the learning process. The planner
                # just needs to know if any *new* uncertain things were discovered.
                perceived, _, _ = self.language_cortex.perceive_text_block(text)
                self.add_curiosity_targets(perceived)
            print("  - Plan complete: Browsing finished."); self.current_plan = {}