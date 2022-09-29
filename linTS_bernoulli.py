import numpy as np
import matplotlib.pyplot as plt

class linTS_disjoint_arm():

    def __init__(self, arm_index, d):
        
        self.d = d
        
        self.sigma = 1
        
        # Track arm index
        self.arm_index = arm_index
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        
        
        
    def sample(self, x, arm_index):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        
        x = x.reshape(-1,1)
        
        DTD = self.A - np.identity(self.d)
        
        theta = np.dot(A_inv, self.b)
        mu = np.dot(x.T, theta)[0][0]
        
        Var = self.sigma**2*np.linalg.norm(np.dot(x.T, np.dot(A_inv, np.dot(DTD, A_inv))))
        alpha = mu*(mu*(1-mu)/Var - 1)

        beta = (1-mu)*(mu*(1-mu)/Var - 1)
        if alpha < 1 or np.isnan(alpha):
            alpha = 1
        
        if beta < 1 or np.isnan(beta):
            beta = 1

        r_hat = np.random.beta(alpha, beta)
        
        return r_hat
    
    def update_reward(self, x, r):
        
        x = x.reshape([-1,1])
        
        
        self.A += np.dot(x, x.T)
        
        
        self.b += r * x
    
    
    
class linTS_policy():
    
    def __init__(self, K_arms, d):
        self.K_arms = K_arms
        self.linTS_arms = [linTS_disjoint_arm(arm_index = 1, d = d) for i in range(K_arms)]
        
        
    def select_arm(self, x):
        
        highest_reward = -1
        
        candidate_arms = []
        
        for arm_index in range(self.K_arms):

            current_reward = self.linTS_arms[arm_index].sample(x, arm_index)
            
            # If current_reward is highest than current highest_reward
            if current_reward > highest_reward:
                
                # Set new max ucb
                highest_reward = current_reward
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if current_reward == highest_reward:
                
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        
        
        return chosen_arm
        
        
        
def generate_dataset():
    random_list = np.array(np.random.rand(1000, 2))
    reward = []
    for element, element2 in random_list:
        if element > 0.5:
            reward.append([0,1])
        elif element2 > 0.5:
            reward.append([1,0])
        elif np.random.rand(1)[0] > 0.5:
            reward.append([0,1])
        else:
            reward.append([1,0])
    
    reward = np.array(reward)
    return random_list, reward
    
random_list, reward = generate_dataset()

lints = linTS_policy(2, 2)

lints.linTS_arms[0].update_reward(random_list[0], reward[0][0])
lints.linTS_arms[1].update_reward(random_list[0], reward[0][1])
list_reward = []
for i in range(1, len(random_list)):
    x = random_list[i]
    r = reward[i]
    chosen_arm = lints.select_arm(x)
    lints.linTS_arms[chosen_arm].update_reward(x, r[chosen_arm])
    if chosen_arm == 0:
        lints.linTS_arms[1].update_reward(x, r[1])
    elif chosen_arm == 1:
        lints.linTS_arms[0].update_reward(x, r[0])

    list_reward.append(r[chosen_arm])

print(np.count_nonzero(list_reward)/1000)

def plot_reward(list_reward):
    sum_reward = 0
    list_plot = []
    for i in range(len(list_reward)):
        sum_reward += list_reward[i]
        list_plot.append(sum_reward)
    
    plt.plot(list_plot)
        
plot_reward(list_reward)        
        
        
        