from collections import deque
import numpy as np 

class window_deq(object):
    def __init__(self, state_dim, window_size):
        self.state_dim = state_dim
        self.window_size = window_size
        self.state_deque = deque(maxlen = window_size)  
        #initialize 
        for _ in range(window_size):
            self.state_deque.append([0]*state_dim)
    
    def concat_state(self):
        return np.concatenate(self.state_deque)
    
    def put_info(self, state):
        self.state_deque.append(state)