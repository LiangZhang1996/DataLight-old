"""
Fixed-Time agent.
Use pre-assigned time duration for each phase.
Add random select to increase diversety.
"""

from .agent import Agent
import numpy as np


class FixedtimeAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(FixedtimeAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.action = np.random.randint(4)
        self.IDX = 0

    # fixedtime agent
    def choose_action(self, state):
        self.action += 1
        self.action = self.action % 4
        
        self.IDX += 1
        if self.IDX % 20 == 0:
            # for randomness
            self.action = np.random.randint(4)
        
        return self.action
    
    def train_network(self):
        pass
    
    def save_network(self, path):
        pass
