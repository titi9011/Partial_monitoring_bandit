import numpy as np

class linTS_disjoint_arm():

    def __init__(self, arm_index, d):
        
        self.d = d
        
        self.sigma = 1
        
        # Track arm index
        self.arm_index = arm_index
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A_star = np.identity(d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        
        
        
    def sample(self, x):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        A_star_inv = np.linalg.inv(self.A_star)
        
        x = x.reshape(-1,1)
        
        theta = np.dot(A_inv, self.b).T[0]
        
        Var = self.sigma**2*A_star_inv

        r_hat = np.random.multivariate_normal(theta, Var)
        
        self.A_star += np.dot(x, x.T)

        return r_hat
    
    def update_reward(self, x, r):
        
        x = x.reshape([-1,1])
        
        self.A += np.dot(x, x.T)
        
        self.b += r * x
        
    def clean_variables(self):
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(self.d)
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A_star = np.identity(self.d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([self.d,1])
    
    
    
class linTS_policy():
    
    def __init__(self, K_arms, d):
        self.K_arms = K_arms
        self.linTS_arms = [linTS_disjoint_arm(arm_index = 1, d = d) for i in range(K_arms)]
        
        
    def get_action(self, x):
        
        x = x.reshape(-1,1)
        
        highest_reward = -10
        
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            
            mu_tilde = self.linTS_arms[arm_index].sample(x)
            
            current_reward = np.dot(x.T, mu_tilde)
            
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
    
    def update(self, action, feedback, outcome, t, x_array):
        self.linTS_arms[action].reward_update(feedback, x_array)
    
    def clean_all_variables(self):
        for arm in self.linTS_arms:
            arm.clean_variables()
        
        
def generate_dataset():
    random_list = np.array(np.random.rand(500, 2))
    reward = []
    for element, element2 in random_list:
        if element > 0.9 and element2 > 0.9:
            reward.append([0,1])
        else:
            reward.append([1,0])
    
    reward = np.array(reward)
    return random_list, reward

def cumulative_regret(list_regret):
    cumulative_list = []
    sum = 0
    for regret in list_regret:
        sum += regret
        cumulative_list.append(sum)
    return np.array(cumulative_list)
    
list_list_regret = []
convertor = [1,0]
for i in range(100):
    random_list, reward = generate_dataset()
    
    lints = linTS_policy(2, 2)
    
    lints.linTS_arms[0].update_reward(random_list[0], reward[0][0])
    lints.linTS_arms[1].update_reward(random_list[0], reward[0][1])
    list_regret = []
    for i in range(1, len(random_list)):
        x = random_list[i]
        r = reward[i]
        chosen_arm = lints.get_action(x)
        lints.linTS_arms[chosen_arm].update_reward(x, r[chosen_arm])
        if chosen_arm == 0:
            lints.linTS_arms[1].update_reward(x, r[1])
        elif chosen_arm == 1:
            lints.linTS_arms[0].update_reward(x, r[0])
    
        list_regret.append(r[convertor[chosen_arm]])


    list_list_regret.append(list_regret)

    
list_list_regret = np.array(list_list_regret)
list_regret = np.mean(list_list_regret, axis=0)
list_regret = cumulative_regret(list_regret)