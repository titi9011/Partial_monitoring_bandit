import numpy as np
import matplotlib.pyplot as plt
import random


class linucb_disjoint_arm():
    
    def __init__(self, arm_index, d, alpha):
        
        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of alpha
        self.alpha = alpha
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        
    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        
        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)
        
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))
        
        return p
    
    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x
        
        
class linucb_policy():
    
    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index = 1, d = d, alpha = alpha) for i in range(K_arms)]
        
    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)
            
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
        
        return chosen_arm



class partial_monitoring:
    def __init__(self):
        self.randomized = True
        self.linucb_policy_object = linucb_policy(K_arms = 2, d = 1, alpha = 1)
        
    def next_state(self, current_state, random):
        if random == True:
            state = np.random.choice([i for i in range(10)], p=self.markov[current_state])
        else:
            state = np.argmax(self.markov[current_state])
        return state
    
    def predict_states(self, first_state, nb_predictions, random):
        visited_states = [first_state]
        for i in range(nb_predictions):
            next_state = self.next_state(visited_states[-1], random)
            visited_states.append(next_state)
        return visited_states
            
    def next_state_black_box(self, current_state, random):
        if random == True:
            state = np.random.choice([i for i in range(10)], p=self.black_box[current_state])
        else:
            state = np.argmax(self.black_box[current_state])
        return state
    
    def predict_states_black_box(self, first_state, nb_predictions, random):
        visited_states = [first_state]
        for i in range(nb_predictions):
            next_state = self.next_state_black_box(visited_states[-1], random)
            visited_states.append(next_state)
        return visited_states

    def plot_states(self, visited_states):
        plt.figure(dpi=1000)
        plt.ylabel('State')
        plt.xlabel('Step')
        plt.plot(visited_states)
        plt.show()
    
    def create_reduced_matrix(self):
        #Create a markov matrix with a path that has a tendency to decrease
        gaussian_reduced_random = np.zeros((10,10))
        for i in range(len(gaussian_reduced_random)):
            count, bins, ignored = plt.hist(np.random.normal(9, 1, 100), bins=range(0, 20,1))
            plt.clf()
            gaussian_reduced_random[i] = count[9-i:19-i]
        gaussian_reduced_random = np.abs(gaussian_reduced_random)
        row_sums = gaussian_reduced_random.sum(axis=1)
        gaussian_reduced_random = gaussian_reduced_random / row_sums[:, np.newaxis]
        self.markov = np.arrya(gaussian_reduced_random)
    
    def create_augmented_matrix(self):
        #Create a markov matrix with a path that has a tendency to rise
        gaussian_augmented_random = np.zeros((10,10))
        for i in range(len(gaussian_augmented_random)):
            sigma = i*0.15
            count, bins, ignored = plt.hist(np.random.normal(10, sigma, 100), bins=range(0, 20,1))
            plt.clf()
            gaussian_augmented_random[i] = count[9-i:19-i]
        gaussian_augmented_random = np.abs(gaussian_augmented_random)
        row_sums = gaussian_augmented_random.sum(axis=1)
        gaussian_augmented_random = gaussian_augmented_random / row_sums[:, np.newaxis]
        self.markov = np.array(gaussian_augmented_random)
        
    def create_black_box_gaussian(self):
        len_row = len(self.markov)
        noise = np.random.normal(0, 0.01, len_row**2)
        noise = noise.reshape(self.markov.shape)
        self.black_box = np.abs(self.markov + noise)
        row_sums = self.black_box.sum(axis=1)
        self.black_box = self.black_box / row_sums[:, np.newaxis]


    def create_black_box_proportional(self):
        self.black_box = np.empty(np.shape(self.markov))
        for i in range(len(self.markov)):
            for j in range(len(self.markov[0])):
                noise = np.random.normal(0, self.markov[i][j])
                self.black_box[i][j] = self.markov[i][j] + noise
        self.black_box = np.abs(self.black_box)
        row_sums = self.black_box.sum(axis=1)
        self.black_box = self.black_box / row_sums[:, np.newaxis]


    def list_find_first_greater_than(self, list, threshold):
        for count,item in enumerate(list):
            if item >= threshold:
                return count

    def threshold_prob(self, state, threshold):
        nb_iter = 10000
        prob = 95
        self.positions = []
        for i in range(nb_iter):
            predictions = self.predict_states(state, 30, True)
            position = self.list_find_first_greater_than(predictions, threshold)
            self.positions.append(position)
        self.positions = list(filter(None, self.positions))
        self.positions.sort()
        return self.positions[int(nb_iter*(100-prob)/100)]


    def threshold_prob_black_box(self, state, threshold):
        #On estime qu'il y a au minimum 95% de probabilité que l'état threshold
        #sera atteint à la journée retourné ou plus.
        #On estime qu'il y a au minimum 5% de probabilité que l'état threshold
        #sera atteint à la journée retourné ou moins.
        nb_iter = 10000
        prob = 95
        self.positions_black_box = []
        for i in range(nb_iter):
            predictions = self.predict_states_black_box(state, 30, True)
            position = self.list_find_first_greater_than(predictions, threshold)
            self.positions_black_box.append(position)
        self.positions_black_box = list(filter(None, self.positions_black_box))
        self.positions_black_box.sort()
        return self.positions_black_box[int(nb_iter*(100-prob)/100)]
        
    
    def plot_positions(self):
        plt.figure(figsize=(10,8))
        plt.hist(self.positions, bins=[i for i in range(30)], alpha=0.5, label='Marvok matrix')
        plt.hist(self.positions_black_box, bins=[i for i in range(30)], alpha=0.5, label='Black box matrix')
        plt.xlabel('Day')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.plot()
        


    def threshold(self, first_state, nb_day, threshold):
        real_states = self.predict_states(first_state, nb_day, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_surveys = []#
        for t in range(1, nb_day+1):
            predicted_state = self.next_state_black_box(predicted_state, False)
            if predicted_state >= threshold:
                list_predicted_states.append(predicted_state)
                predicted_state = real_states[t]
                list_surveys.append(True)
            else:
                list_predicted_states.append(predicted_state)
                list_surveys.append(False)
        return real_states, list_predicted_states, list_surveys

    def performance_threshold(self, first_state, nb_day, threshold):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        for i in range(1):
            real_states, list_predicted_states, list_surveys = self.threshold(first_state, nb_day, threshold)
            print(real_states, list_predicted_states, list_surveys)
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
        return np.array([[vp, fn],[fp, vn]])
        

    def linUCB(self, first_state, nb_day, threshold):
        real_states = self.predict_states(first_state, nb_day, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_surveys = []#
        for t in range(1, nb_day+1):
            predicted_state = self.next_state_black_box(predicted_state, False)
            survey_tomorrow = self.linucb_policy_object.select_arm(predicted_state)
            if survey_tomorrow == 1:
                list_surveys.append(True)#
                list_predicted_states.append(predicted_state)#
                predicted_state = real_states[t]
                
                if real_states[t] >= threshold:
                    self.linucb_policy_object.linucb_arms[survey_tomorrow].reward_update(1, predicted_state)

                else:
                    self.linucb_policy_object.linucb_arms[survey_tomorrow].reward_update(0, predicted_state)
                    
            elif survey_tomorrow == 0:
                list_surveys.append(False)#
                self.linucb_policy_object.linucb_arms[survey_tomorrow].reward_update(0.5, predicted_state)
                list_predicted_states.append(predicted_state)#
                
        return real_states, list_predicted_states, list_surveys


    '''
    def performance_linUCB(self, first_state, nb_day, threshold):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        for i in range(1000):
            real_states, list_predicted_states, list_surveys = self.linUCB(first_state, nb_day, threshold)
            #print(real_states, list_predicted_states, list_surveys)
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
        return np.array([[vp, fn],[fp, vn]])

    '''

    def performance_linUCB(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        for i in range(1000):
            nb_day = random.randint(0,30)
            first_state = random.randint(0,9)
            threshold = random.randint(0,9)
            real_states, list_predicted_states, list_surveys = self.linUCB(first_state, nb_day, threshold)
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
        return np.array([[vp, fn],[fp, vn]])

        
x = partial_monitoring()
x.create_augmented_matrix()
x.create_black_box_proportional()
y = x.performance_linUCB()


print((y[0][0]+y[1][1])/(y[0][0] + y[1][0] + y[0][1] + y[1][1])*100)
print(y[0][1]/(y[0][0] + y[1][0] + y[0][1] + y[0][1])*100)





    
    

