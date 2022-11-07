import numpy as np
import matplotlib.pyplot as plt
import random
from UCB import UCB
from linUCB import linucb_policy
from linTS import linTS_policy
from scipy import signal
import seaborn as sns

#np.random.seed(0)

class partial_monitoring:
    def __init__(self):
        self.randomized = True
        self.linucb_policy_object = linucb_policy(K_arms = 2, d = 2)
        self.linTS_policy = linTS_policy(2,1)
        self.UCB = UCB(2)
        self.reward = 1
        self.iteration_plot = 20
        self.time_horizon = 30
        self.first_state = 4
        self.threshold = 6
        
    def next_state(self, current_state, random):
        if random == True:
            state = np.random.choice([i for i in range(10)], p=self.markov[current_state])
        else:
            state = np.argmax(self.markov[current_state])
        return state
    
    def predict_states(self, first_state, nb_predictions, random):
        #np.random.seed(0) #Thierry SEED
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
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.plot()
        


    def threshold(self, first_state, time_horizon, threshold):
        real_states = self.predict_states(first_state, time_horizon, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_gather_info = []#
        for t in range(1, time_horizon+1):
            predicted_state = self.next_state_black_box(predicted_state, False)
            if predicted_state >= threshold:
                list_predicted_states.append(predicted_state)
                predicted_state = real_states[t]
                list_gather_info.append(True)
            else:
                list_predicted_states.append(predicted_state)
                list_gather_info.append(False)
        return real_states, list_predicted_states, list_gather_info

    def performance_threshold(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_predicted_days = 0
            time_horizon = self.time_horizon
            first_state = self.first_state
            threshold = self.threshold
            real_states, list_predicted_states, list_gather_info = self.threshold(first_state, time_horizon, threshold)
            for t in range(time_horizon):
                if real_states[t+1] >= threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    vp += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    fn += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    fp += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    vn += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret
        
    def apply_UCB(self, first_state, time_horizon, threshold):
        real_states = self.predict_states(first_state, time_horizon, True)
        list_gather_info = []#
        for t in range(1, time_horizon+1):
            action = self.UCB.get_action()
            #print('UCB',real_states[t], action)
            #print(real_states[t], action)
#            if t == 1:
#                list_gather_info.append(False)
#            elif t == 2:
#                list_gather_info.append(True)
#                if real_states[t] >= threshold:
#                    self.UCB.update(0, self.reward)
#                elif real_states[t] < threshold:
#                    self.UCB.update(0, 0)
            if action == 1:
                list_gather_info.append(True)
                if real_states[t] >= threshold:
                    self.UCB.update(1, self.reward)
                    self.UCB.update(0, 0)
                elif real_states[t] < threshold:
                    self.UCB.update(1, 0)
                    self.UCB.update(0, self.reward)
            elif action == 0:
                list_gather_info.append(False)
        return real_states, list_gather_info
    
    def performance_UCB(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            time_horizon = self.time_horizon
            first_state = self.first_state
            threshold = self.threshold
            real_states, list_gather_info = self.apply_UCB(first_state, time_horizon, threshold)
            for t in range(time_horizon):
                if real_states[t+1] >= threshold and list_gather_info[t] == True:
                    vp += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_gather_info[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == True:
                    fp += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == False:
                    vn += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret
        

    def linUCB(self, first_state, time_horizon, threshold):
        real_states = self.predict_states(first_state, time_horizon, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_gather_info = []#
        for t in range(1, time_horizon+1):
            #predicted_state = self.next_state_black_box(predicted_state, False)
            predicted_state = real_states[t]
            gather_info = self.linucb_policy_object.get_action(t, np.array([[predicted_state],[9-predicted_state]]))
            #print('linUCB',predicted_state, gather_info)
            if gather_info == True:
                list_gather_info.append(True)#
                list_predicted_states.append(predicted_state)#
                predicted_state = real_states[t]
                
                if real_states[t] >= threshold:
                    self.linucb_policy_object.linucb_arms[gather_info].reward_update(self.reward, np.array([[predicted_state],[9-predicted_state]]))
                    self.linucb_policy_object.linucb_arms[False].reward_update(0, np.array([[predicted_state],[9-predicted_state]]))
                    
                
                else:
                    self.linucb_policy_object.linucb_arms[gather_info].reward_update(0, np.array([[predicted_state],[9-predicted_state]]))
                    self.linucb_policy_object.linucb_arms[False].reward_update(self.reward, np.array([[predicted_state],[9-predicted_state]]))
                    
            elif gather_info == False:
                list_gather_info.append(False)#
                list_predicted_states.append(predicted_state)#
        return real_states, list_predicted_states, list_gather_info


    
    def performance_linUCB(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            time_horizon = self.time_horizon
            first_state = self.first_state
            threshold = self.threshold
            real_states, list_predicted_states, list_gather_info = self.linUCB(first_state, time_horizon, threshold)
            for t in range(time_horizon):
                if real_states[t+1] >= threshold and list_gather_info[t] == True:
                    vp += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_gather_info[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == True:
                    fp += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == False:
                    vn += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret

    def linTS(self, first_state, time_horizon, threshold):
        real_states = self.predict_states(first_state, time_horizon, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_gather_info = []#
        for t in range(1, time_horizon+1):
            predicted_state = self.next_state_black_box(predicted_state, False)
            gather_info = self.linTS_policy.get_action(predicted_state)
            if gather_info == True:
                list_gather_info.append(True)#
                list_predicted_states.append(predicted_state)#
                predicted_state = real_states[t]
                
                if real_states[t] >= threshold:
                    self.linTS_policy.linTS_arms[gather_info].update_reward(predicted_state, np.array([self.reward]))
                
                else:
                    self.linTS_policy.linTS_arms[gather_info].update_reward(predicted_state, np.array([0]))
                    
            elif gather_info == False:
                list_gather_info.append(False)#
                list_predicted_states.append(predicted_state)#
        return real_states, list_predicted_states, list_gather_info

    def performance_linTS(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_predicted_days = 0
            time_horizon = self.time_horizon
            first_state = self.first_state
            threshold = self.threshold
            real_states, list_predicted_states, list_gather_info = self.linTS(first_state, time_horizon, threshold)
            for t in range(time_horizon):
                if real_states[t+1] >= threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    vp += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    fn += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    fp += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    vn += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret
    
    def performance_random(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_predicted_days = 0
            time_horizon = self.time_horizon
            first_state = self.first_state
            threshold = self.threshold
            real_states = self.predict_states(first_state, time_horizon, True)
            list_gather_info = [random.choice([True, False]) for i in range(len(real_states)-1)]
            for t in range(time_horizon):
                if real_states[t+1] >= threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    vp += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    fn += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == True:
                    nb_predicted_days = 0
                    fp += 1
                    pseudo_regret_cumulatif += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_gather_info[t] == False:
                    nb_predicted_days += 1
                    vn += 1
                    pseudo_regret_cumulatif += 0
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret
    
    def plot_confusion_matrix(self, confusion_matrix, title):
        #ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
        #                 fmt='.2%', cmap='Blues')
        ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ax.xaxis.set_ticklabels(['True','False'])
        ax.yaxis.set_ticklabels(['True','False'])
        ax.set_title(title)
        plt.show()
    
    def UCBvslinUCBvsRandom(self):
        list_list_regret_UCB = []
        list_list_regret_linUCB = []
        list_list_regret_random = []#
        #list_list_regret_linTS = []#
        
        list_list_confusion_UCB = []
        list_list_confusion_linUCB = []
        list_list_confusion_random = []#
        #list_list_confusion_linTS = []#
        for i in range(50):
            self.linucb_policy_object.clean_all_variables()
            self.UCB.clean_all_variables()
            #self.linTS_policy.clean_all_variables()
            matrix, list_regret_UCB = self.performance_UCB()
            list_list_regret_UCB.append(list_regret_UCB)
            list_list_confusion_UCB.append(matrix)
            matrix, list_regret_linUCB = self.performance_linUCB()
            list_list_regret_linUCB.append(list_regret_linUCB)
            list_list_confusion_linUCB.append(matrix)
            matrix, list_regret_random = self.performance_random()#
            list_list_regret_random.append(list_regret_random)#
            list_list_confusion_random.append(matrix)#
            #matrix, list_regret_linTS = self.performance_linTS()#
            #list_list_regret_linTS.append(list_regret_linTS)#
            #list_list_confusion_linTS.append(matrix)#
           
        list_list_regret_UCB = np.array(list_list_regret_UCB)
        list_list_regret_linUCB = np.array(list_list_regret_linUCB)
        list_list_regret_random = np.array(list_list_regret_random)
        list_list_confusion_UCB = np.array(list_list_confusion_UCB)
        list_list_confusion_linUCB = np.array(list_list_confusion_linUCB)
        list_list_confusion_random = np.array(list_list_confusion_random)
        #list_list_confusion_linTS = np.array(list_list_confusion_linTS)
        
        #Confusion matrix
        np.set_printoptions(suppress=True)
        mean_confusion_UCB = np.mean(list_list_confusion_UCB, axis=0)
        self.plot_confusion_matrix(mean_confusion_UCB, 'UCB')
        mean_confusion_linUCB = np.mean(list_list_confusion_linUCB, axis=0)
        self.plot_confusion_matrix(mean_confusion_linUCB, 'LinUCB')
        mean_confusion_random = np.mean(list_list_confusion_random, axis=0)#
        self.plot_confusion_matrix(mean_confusion_random, 'Random')
        #mean_confusion_linTS = np.mean(list_list_confusion_linTS, axis=0)#
        #self.plot_confusion_matrix(mean_confusion_linTS, 'linTS')
        
        mean_UCB = np.mean(list_list_regret_UCB, axis=0)
        std_UCB = np.std(list_list_regret_UCB, axis=0)
        UCB_upper = mean_UCB + std_UCB
        UCB_lower = mean_UCB - std_UCB
        
        mean_linUCB = np.mean(list_list_regret_linUCB, axis=0)
        std_linUCB = np.std(list_list_regret_linUCB, axis=0)
        linUCB_upper = mean_linUCB + std_linUCB
        linUCB_lower = mean_linUCB - std_linUCB
        x = [i for i in range(len(mean_linUCB))]
        
        mean_random = np.mean(list_list_regret_random, axis=0)#
        std_random = np.std(list_list_regret_random, axis=0)#
        random_upper = mean_random + std_random#
        random_lower = mean_random - std_random#

        plt.plot(mean_UCB, label='UCB mean')
        plt.fill_between(x, UCB_lower, UCB_upper, alpha = 0.3)
        plt.plot(mean_linUCB, label='LinUCB mean')
        plt.fill_between(x, linUCB_upper, linUCB_lower, alpha = 0.3)
        #plt.plot(mean_random, label='Random mean')#
        #plt.fill_between(x, random_upper, random_lower, alpha = 0.3)#
        plt.xlabel('Day')
        plt.ylabel('Mean cumulative regret')
        plt.legend(loc='upper right')
        plt.show()
        
    def linUCBvslinTS(self):
        list_list_regret_linTS = []
        list_list_regret_linUCB = []
        
        for i in range(10):
            self.linucb_policy_object.clean_all_variables()
            self.linTS_policy.clean_all_variables()
            matrix, list_regret_linTS = self.performance_linTS()
            list_list_regret_linTS.append(list_regret_linTS)
            matrix, list_regret_linUCB = self.performance_linUCB()
            list_list_regret_linUCB.append(list_regret_linUCB)
           
        list_list_regret_linTS = np.array(list_list_regret_linTS)
        list_list_regret_linUCB = np.array(list_list_regret_linUCB)

        mean_linTS = np.mean(list_list_regret_linTS, axis=0)
        std_linTS = np.std(list_list_regret_linTS, axis=0)
        linTS_upper = mean_linTS + std_linTS
        linTS_lower = mean_linTS - std_linTS
        
        mean_linUCB = np.mean(list_list_regret_linUCB, axis=0)
        std_linUCB = np.std(list_list_regret_linUCB, axis=0)
        linUCB_upper = mean_linUCB + std_linUCB
        linUCB_lower = mean_linUCB - std_linUCB
        x = [i for i in range(len(mean_linUCB))]
        
        plt.plot(mean_linTS, label='linTS mean')
        plt.fill_between(x, linTS_lower, linTS_upper, alpha = 0.3)
        plt.plot(mean_linUCB, label='LinUCB mean')
        plt.fill_between(x, linUCB_upper, linUCB_lower, alpha = 0.3)
        plt.xlabel('Day')
        plt.ylabel('Mean cumulative regret')
        plt.legend(loc='upper right')
        plt.show()
            
    def plot_mean_states_matrices(self):
        time_horizon = 30
        list_list_spike = []
        list_list_gaussian = []
        list_list_augmented = []
        list_list_reduced = []
        for i in range(20):
            self.create_spike_matrix()
            list_list_spike.append(self.predict_states(5, time_horizon, True))
            self.create_gaussian_matrix()
            list_list_gaussian.append(self.predict_states(1, time_horizon, True))
            self.create_augmented_matrix()
            list_list_augmented.append(self.predict_states(1, time_horizon, True))
            self.create_reduced_matrix()
            list_list_reduced.append(self.predict_states(9, time_horizon, True))
            
        list_list_spike = np.array(list_list_spike)
        list_list_gaussian = np.array(list_list_gaussian)
        list_list_augmented = np.array(list_list_augmented)
        list_list_reduced = np.array(list_list_reduced)
        
        mean_spike = np.mean(list_list_spike, axis=0)
        std_spike = np.std(list_list_spike, axis=0)
        spike_upper = mean_spike + std_spike
        spike_lower = mean_spike - std_spike
        
        mean_gaussian = np.mean(list_list_gaussian, axis=0)
        std_gaussian = np.std(list_list_gaussian, axis=0)
        gaussian_upper = mean_gaussian + std_gaussian
        gaussian_lower = mean_gaussian - std_gaussian
        
        mean_augmented = np.mean(list_list_augmented, axis=0)
        std_augmented = np.std(list_list_augmented, axis=0)
        augmented_upper = mean_augmented + std_augmented
        augmented_lower = mean_augmented - std_augmented
        
        mean_reduced = np.mean(list_list_reduced, axis=0)
        std_reduced = np.std(list_list_reduced, axis=0)
        reduced_upper = mean_reduced + std_reduced
        reduced_lower = mean_reduced - std_reduced
        
        x = [i for i in range(len(mean_augmented))]
        
        plt.figure(figsize=(10,7))
        plt.plot(mean_spike, label='Spike matrix')
        plt.fill_between(x, spike_lower, spike_upper, alpha = 0.3)
        plt.plot(mean_gaussian, label='Gaussian matrix')
        plt.fill_between(x, gaussian_upper, gaussian_lower, alpha = 0.3)
        plt.plot(mean_augmented, label='Augmented matrix')
        plt.fill_between(x, augmented_upper, augmented_lower, alpha = 0.3)
        plt.plot(mean_reduced, label='Reduced matrix')
        plt.fill_between(x, reduced_upper, reduced_lower, alpha = 0.3)
        plt.xlabel('Day')
        plt.ylabel('Mean path')
        plt.legend(loc='upper right')
        plt.show()
        
    def pseudo_regret(self, first_state, threshold, n_prediction, action_took):
        list_probability = self.markov[first_state]
        for n in range(n_prediction):
            list_probability_n = np.zeros(10)
            for i in range(10):
                list_probability_n += list_probability[i]*self.markov[i]
            list_probability = list_probability_n
        sum_under_thresold = np.sum(list_probability[:threshold-1])
        sum_after_thresold = np.sum(list_probability[threshold-1:])
        
        sums = [sum_under_thresold, sum_after_thresold]
        
        pseudo_regret = np.max(sums) - sums[action_took]
        
        return pseudo_regret

        


x = partial_monitoring()
x.create_augmented_matrix()

#x.markov = np.identity(10)


#x.markov = np.zeros((10,10))
#x.markov[0,0] = 0.5
#x.markov[1,0] = 0.5
#x.markov[0,1] = 0.5
#x.markov[1,1] = 0.5

x.black_box = x.markov


#x.create_black_box_gaussian()

x.UCBvslinUCBvsRandom()

print(x.predict_states(3, 30, True))
print(x.predict_states(3, 30, True))


