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
    def __init__(self, fabric, logic, emotion, memory, action_cortex, language_cortex):
        self.fabric, self.logic, self.emotion, self.memory, self.action_cortex, self.language_cortex = fabric, logic, emotion, memory, action_cortex, language_cortex
        self.goals, self.active_goal, self.last_plan_time, self.plan_cooldown = {}, None, 0, 2
        self.current_plan, self.curiosity_queue, self.visited_urls = {}, [], set()
        self._initialize_goals()
        print("PlanningCortex initialized for explicit action formulation.")

    def _initialize_goals(self):
        # This method is unchanged
        goal_zone_uids = list(self.fabric.zones.get('goal', []));
        if len(goal_zone_uids) < 20: print("WARNING: Not enough 'goal' neurons."); return
        neurons = set(goal_zone_uids)
        p_pos = set(random.sample(list(neurons), 5)); neurons -= p_pos; self.logic.bind_symbol_to_pattern("goal_seek_positive_valence", p_pos)
        p_achieved = set(random.sample(list(neurons), 5)); neurons -= p_achieved; self.logic.bind_symbol_to_pattern("state_is_positive", p_achieved); self.emotion.positive_state_pattern = p_achieved
        p_resolve = set(random.sample(list(neurons), 5)); neurons -= p_resolve; self.logic.bind_symbol_to_pattern("goal_resolve_uncertainty", p_resolve)
        p_effective = set(random.sample(list(neurons), 5)); self.logic.bind_symbol_to_pattern("meta_high_effectiveness", p_effective)

    def _is_uncertain(self, p):
        # This method is unchanged
        if not p: return False
        s = next((s for s, pat in self.fabric.symbol_table.items() if pat == p), None)
        if not s or not s.startswith("word_"): return False
        conns = sum(len(self.fabric.synapses.get(uid, {})) for uid in p)
        return (conns / len(p)) < 2 if p else False

    def add_curiosity_targets(self, patterns):
        # This method is unchanged
        new = 0
        for p in patterns:
            if self._is_uncertain(p) and p not in self.curiosity_queue:
                self.curiosity_queue.append(p); new += 1
        print(f"  - Found {new} new points of uncertainty to explore.")

    def step(self):
        # This method is unchanged
        self._evaluate_goals()
        if self.active_goal and (time.time() - self.last_plan_time > self.plan_cooldown):
            next_action = self._formulate_next_action()
            self.last_plan_time = time.time()
            return next_action
        return None

    def _evaluate_goals(self):
        # This method is unchanged
        if self.current_plan: self.active_goal = self.current_plan['type']; return
        self.active_goal = None
        if self.curiosity_queue:
            self.active_goal = "research"
            self.current_plan = {'type': 'research', 'topic_pattern': self.curiosity_queue.pop(0), 'state': 'start'}
        elif self.emotion.get_current_valence() < 0.1: self.active_goal = "positive_valence"

    def _formulate_next_action(self):
        # This method is unchanged
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

    def update_plan_with_result(self, result):
        if not self.current_plan: return
        state = self.current_plan.get('state')
        print(f"--- PLAN: Updating plan with result. State was: {state} ---")
        if state == 'start':
            if result:
                self.current_plan['state'] = 'has_url'
                self.current_plan['url'] = result[0]
            else:
                print("  - Plan failed: Search returned no URLs."); self.current_plan = {}
        elif state == 'has_url':
            text, _ = result if isinstance(result, tuple) else (None, None)
            if text:
                # --- THE FIX: Use the correct attribute name self.language_cortex ---
                perceived, main_idea = self.language_cortex.perceive_text_block(text)
                if main_idea:
                    self.logic.integrate_textual_knowledge(main_idea, perceived)
                self.add_curiosity_targets(perceived)
            print("  - Plan complete: Browsing finished."); self.current_plan = {}