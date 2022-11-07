import numpy as np

class UCB():
    def __init__(self, k):
        self.k = k
        self.t = 0
        self.u = np.zeros(k)
        self.N_k = np.ones(k)
    
    def get_action(self):
        list_ucb = np.zeros(self.k)
        for k in range(self.k):
            list_ucb[k] = self.u[k] + np.sqrt(2*np.log(self.t)/self.N_k[k])
        chosen_action = np.argmax(list_ucb)
        self.N_k[chosen_action] += 1
        self.t += 1
        
        return chosen_action
        
    def update(self, action, reward):
        self.u[action] = (self.u[action]*(self.N_k[action]-1) + reward)/self.N_k[action]
    
    def clean_all_variables(self):
        #self.t = 0
        self.u = np.zeros(self.k)
        self.N_k = np.ones(self.k)
        

def test_UCB():
    count_action_2 = 0
    ucb = UCB(2)
    reward1 = np.random.normal(1,2)
    reward2 = np.random.normal(3,2)
    ucb.reward(0, reward1)
    ucb.reward(1, reward2)
    for i in range(1000):
        chosen_action = ucb.ucb()
        if chosen_action == 0:
            reward1 = np.random.normal(1,3)
            ucb.reward(0, reward1)
        else:
            reward2 = np.random.normal(3,2)
            ucb.reward(1, reward2)
            count_action_2 += 1
    assert count_action_2 > 900
    
    
if __name__== '__main__':
    test_UCB()
