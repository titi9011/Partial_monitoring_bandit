import numpy as np
import matplotlib.pyplot as plt
import random
from UCB import UCB
from PMlinUCB import PM_linucb_policy
from linTS import linTS_policy
from scipy import signal




class Patient_problem():
    def __init__(self):
        self.LossMatrix = [[1,0],[0,0]]

        
    def next_state(self, current_state):
        #np.random.seed(0) #Thierry SEED
        state = np.random.choice([i for i in range(10)], p=self.markov[current_state])
        return state
    
    def predict_states(self, first_state, nb_predictions):
        visited_states = [first_state]
        for i in range(nb_predictions):
            next_state = self.next_state(visited_states[-1])
            visited_states.append(next_state)
        return visited_states
            
    def next_state_black_box(self, current_state):
        next_state = np.argmax(self.black_box[current_state])
        return next_state
    

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
        self.markov = np.array(gaussian_reduced_random)
    
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
        
    def create_spike_matrix(self):
        spike_random = np.zeros((10,10))
        s = signal.triang(19)
        random = np.random.normal(0.1, 0.2, 100).reshape((10,10))
        for i in range(len(spike_random)):
            spike_random[i] = s[9-i:19-i]
        spike_random += random
        spike_random = np.abs(spike_random)
        row_sums = spike_random.sum(axis=1)
        spike_random = spike_random / row_sums[:, np.newaxis]
        self.markov = np.array(spike_random)
    
    def create_gaussian_matrix(self):
        gaussian_random = np.zeros((10,10))
        for i in range(len(gaussian_random)):
            count, bins, ignored = plt.hist(np.random.normal(10, 1, 100), bins=range(0, 20,1))
            plt.clf()
            gaussian_random[i] = count[9-i:19-i]
        gaussian_random = np.abs(gaussian_random)
        row_sums = gaussian_random.sum(axis=1)
        gaussian_random = gaussian_random / row_sums[:, np.newaxis]
        self.markov = np.array(gaussian_random)
        
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
    

        


        


class Evaluation:
    def __init__(self, algo, game):
        self.algo = algo
        self.game = game
        self.reward = 1
        self.iteration_for_mean = 1
        self.horizon = 30
        self.nb_horizon = 100
        self.first_state = 3
        self.threshold = 5
        self.easy_states = self.game.predict_states(self.first_state, self.horizon)
        #self.easy_states = [5 for i in range(self.horizon)]


    def anwser(self, state, info):
        if state >= self.threshold and info==True:
            return True
        elif state < self.threshold and info==True:
            return False
        elif state >= self.threshold and info==False:
            return False
        elif state < self.threshold and info==False:
            return True
    

    def eval_policy_once(self):
        real_states = self.game.predict_states(self.first_state, self.horizon)
        predicted_state = self.first_state
        list_predicted_states = []#
        list_gather_info = []#
        for t in range(self.horizon):
            predicted_state = int(real_states[t] + np.random.normal(0, 0.1, 1)[0])
            if predicted_state > 9:
                predicted_state = 9
            context = np.zeros(10)
            context[predicted_state] = 1
            gather_info = self.algo.get_action(t, context)
            #print('linUCB',predicted_state, gather_info, self.anwser(predicted_state, gather_info))
            if gather_info == True:
                list_gather_info.append(True)
                list_predicted_states.append(predicted_state)
                predicted_state = real_states[t] #Now the algo know the real state
                
                if real_states[t] >= self.threshold:
                    self.algo.update(True, self.reward, 0, t, context)
                    
                else:
                    self.algo.update(True, 0, 0, t, context)
                    
            elif gather_info == False:
                list_gather_info.append(False)#
                '''

                list_predicted_states.append(predicted_state)#
                if real_states[t] >= self.threshold:
                    self.algo.update(False, 0, 0, t, context)
                    
                else:
                    self.algo.update(False, 1, 0, t, context)
                '''


        return real_states, list_gather_info


    
    '''
    #black box
    def eval_policy_once(self):
        real_states = self.game.predict_states(self.first_state, self.horizon)
        predicted_state = self.first_state
        list_predicted_states = []#
        list_gather_info = []#
        for t in range(self.horizon):
            predicted_state = self.game.next_state_black_box(predicted_state)
            context = np.zeros(10)
            context[predicted_state] = 1
            gather_info = self.algo.get_action(t, context)
            #print('linUCB',predicted_state, gather_info, self.error(predicted_state, gather_info))
            if gather_info == True:
                list_gather_info.append(True)
                list_predicted_states.append(predicted_state)
                predicted_state = real_states[t] #Now the algo know the real state
                
                if real_states[t] >= self.threshold:
                    self.algo.update(True, self.reward, 0, t, context)
                    
                else:
                    self.algo.update(True, 0, 0, t, context)
                    
            elif gather_info == False:
                list_gather_info.append(False)#
                list_predicted_states.append(predicted_state)#
                if real_states[t] >= self.threshold:
                    self.algo.update(False, 0, 0, t, context)
                    
                else:
                    self.algo.update(False, 1, 0, t, context)

        return real_states, list_gather_info
    '''

    '''
    def eval_policy_once(self):
        states = self.easy_states
        list_gather_info = []
        for t in range( self.horizon):
            context = np.zeros(10)
            context[states[t]] = 1
            gather_info = self.algo.get_action(t, context)
            #print('linUCB',states[t], gather_info, self.anwser(states[t], gather_info))
            if gather_info == True:
                list_gather_info.append(True)
                
                if states[t] >= self.threshold:
                    self.algo.update(True, self.reward, 0, t, context)
                    
                else:
                    self.algo.update(True, 0, 0, t, context)
                
            elif gather_info == False:

                if states[t] >= self.threshold:
                    self.algo.update(False, 0, 0, t, context)
                    
                else:
                    self.algo.update(False, 1, 0, t, context)

                list_gather_info.append(False)


        
        return states, list_gather_info
    '''

    def performance_algo_once(self):
        list_regret = []
        cumulative_regret = 0
        for i in range(self.nb_horizon):
            real_states, list_gather_info = self.eval_policy_once()
            for t in range(self.horizon):
                if real_states[t] >= self.threshold and list_gather_info[t] == True:
                    cumulative_regret += 0
                elif real_states[t] >= self.threshold and list_gather_info[t] == False:
                    cumulative_regret += 1
                elif real_states[t] < self.threshold and list_gather_info[t] == True:
                    cumulative_regret += 1
                elif real_states[t] < self.threshold and list_gather_info[t] == False:
                    cumulative_regret += 0
                list_regret.append(cumulative_regret)
                
        return list_regret
    
    def performance_algo(self, algo_name):
        list_list_cumulative_regret = []
        
        for i in range(self.iteration_for_mean):
            self.algo.clean_all_variables()
            self.easy_states = self.game.predict_states(self.first_state, self.horizon) #à enlever
            
            list_cumulative_regret = self.performance_algo_once()
            list_list_cumulative_regret.append(list_cumulative_regret)

        list_list_cumulative_regret = np.array(list_list_cumulative_regret)
        
        mean = np.mean(list_list_cumulative_regret, axis=0)
        std = np.std(list_list_cumulative_regret, axis=0)
        upper_bound = mean + std
        lower_bound = mean - std
        
        x = [i for i in range(len(mean))]
        plt.plot(mean, label=algo_name)
        plt.fill_between(x, lower_bound, upper_bound, alpha = 0.3)
        plt.xlabel('Timestep')
        plt.ylabel('Mean cumulative regret')
        plt.legend(loc='upper right')
        plt.show()


game = Patient_problem()
game.create_augmented_matrix()
game.black_box = game.markov


algo = PM_linucb_policy(2,10)


evaluation = Evaluation(algo, game)
evaluation.performance_algo('PMLinUCB')

plt.plot(np.array(algo.list_UCB[1]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_UCB[0]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_bound[0]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_bound[1]).reshape(-1,1))
plt.show()

'''

evaluation.algo = UCB(2)

evaluation.performance_algo('UCB')
'''


