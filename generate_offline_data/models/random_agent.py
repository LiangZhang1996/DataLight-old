"""
Random agent.
Randomly select each phase.
"""

from .agent import Agent
import random
import numpy as np


class RandomAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(FixedtimeAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.action = np.random.randint(4)
        self.IDX = 0
        
    
    # random agent
    def choose_action(self, state):
        action = np.random.randint(4)
        return action
    
    def train_network(self):
        pass
    
    def save_network(self, path):
        pass
