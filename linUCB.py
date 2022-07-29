import numpy as np

class linucb_disjoint_arm():
    
    def __init__(self, arm_index, d, k):
        
        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of k
        self.k = k
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        # A_star: (d x d) matrix = D^star_a.T * D^star_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A_star = np.identity(d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        
        
    def calc_UCB(self, x, T):
        
        
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        
        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)
        
        # Find A_star inverse for ridge regression
        A_star_inv = np.linalg.inv(self.A_star)
        
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        ucb = np.dot(self.theta.T,x) +  self.k* np.sqrt(np.log(T)) * np.sqrt(np.dot(x.T, np.dot(A_star_inv,x)))
        
        if T < 100:
            print(ucb)
            
        
        return ucb
    
    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x
        
        
class linucb_policy():
    
    def __init__(self, K_arms, d, k):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index = 1, d = d, k = k) for i in range(K_arms)]
        self.T = 0
        
        
    def select_arm(self, x_array):
        
        # Update T
        self.T += 1
        x = x_array.reshape([-1,1])
        
        # Initiate ucb to be 0
        highest_ucb = -1
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x, self.T)
            
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                
                # Set new max ucb
                highest_ucb = arm_ucb
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        
        # Update A_star
        self.linucb_arms[chosen_arm].A_star += np.dot(x, x.T)
        
        
        return chosen_arm
    
    
class UCB():
    def __init__(self, k):
        self.k = k
        self.t = 0
        self.u = np.zeros(k)
        self.N_k = np.zeros(k)
    
    def ucb(self):
        list_ucb = np.zeros(self.k)
        for t in range(len(self.u)):
            list_ucb[t] = self.u[t] + np.sqrt(2*np.log(self.t)/self.N_k[t])
        return np.argmax(list_ucb)
        
    def reward(self, action, reward):
        self.t += 1
        self.N_k[action] += 1
        self.u[action] = (self.u[action]*(self.N_k[action]-1) + reward)/self.N_k[action]
        

def test_linucb_policy():
    # Action 1: If x > 5 reward = 1 else reward = 0
    # Action 2: If x > 3 reward = 1 else reward = 0    ucb = 0

    LinUCB = linucb_policy(K_arms = 2, d = 1, k = 1)
    first_arm = LinUCB.select_arm(np.array([[2]]))
    LinUCB.linucb_arms[0].reward_update(0, np.array([[3]]))
    LinUCB.linucb_arms[1].reward_update(1, np.array([[3]]))
    if first_arm == 0:
        A = 1+3*3
        A_star = 1 + 2*2
        b = 0*3
        ucb = (1/A)*b*2 + 1*np.sqrt(np.log(2))*np.sqrt(1/A_star*2*2)
        assert LinUCB.linucb_arms[0].calc_UCB(np.array([[2]]), 2)[0][0] == ucb
    elif first_arm == 1:
        A = 1+3*3
        A_star = 1 + 2*2
        b = 1*3
        ucb = (1/A)*b*2 + 1*np.sqrt(np.log(2))*np.sqrt(1/A_star*2*2)
        assert LinUCB.linucb_arms[1].calc_UCB(np.array([[2]]), 2)[0][0] == ucb


if __name__== '__main__':
    test_linucb_policy()
    







        
        