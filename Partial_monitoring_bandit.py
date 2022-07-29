import numpy as np
import matplotlib.pyplot as plt
import random
from linUCB import linucb_policy, UCB
from scipy import signal
import seaborn as sns

class partial_monitoring:
    def __init__(self):
        self.randomized = True
        self.linucb_policy_object = linucb_policy(K_arms = 2, d = 1, k = 1)
        self.UCB = UCB(2)
        self.reward = 1
        #self.unkown_reward = 0
        self.iteration_plot = 200
        
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

    def performance_threshold(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_day = 30
            first_state = 5
            threshold = 8
            real_states, list_predicted_states, list_surveys = self.threshold(first_state, nb_day, threshold)
            
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += self.reward - self.unkown_reward
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                    pseudo_regret_cumulatif += self.unkown_reward
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
                    list_regret.append(pseudo_regret_cumulatif)
        plt.plot(list_regret)
        return np.array([[vp, fn],[fp, vn]]), (vp+vn)/(fn+fp)
        
    def apply_UCB(self, first_state, nb_day, threshold):
        real_states = self.predict_states(first_state, nb_day, True)
        list_surveys = []#
        for t in range(1, nb_day+1):
            if self.UCB.t == 0:
                list_surveys.append(False)
            elif self.UCB.t == 1:
                list_surveys.append(True)
                if real_states[t] >= threshold:
                    self.UCB.reward(0, self.reward)
                elif real_states[t] < threshold:
                    self.UCB.reward(0, 0)
            elif self.UCB.ucb() == 0:
                list_surveys.append(True)
                if real_states[t] >= threshold:
                    self.UCB.reward(0, self.reward)
                elif real_states[t] < threshold:
                    self.UCB.reward(0, 0)
            elif self.UCB.ucb() == 1:
                list_surveys.append(False)
        return real_states, list_surveys
    
    def performance_UCB(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_day = 30
            first_state = 5
            threshold = 8
            real_states, list_surveys = self.apply_UCB(first_state, nb_day, threshold)
            
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += self.reward
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
                    list_regret.append(pseudo_regret_cumulatif)
        #plt.plot(list_regret)
        return np.array([[vp, fn],[fp, vn]]), list_regret
        

    def linUCB(self, first_state, nb_day, threshold):
        real_states = self.predict_states(first_state, nb_day, True)
        predicted_state = first_state
        list_predicted_states = []#
        list_surveys = []#
        for t in range(1, nb_day+1):
            predicted_state = self.next_state_black_box(predicted_state, False)
            survey = self.linucb_policy_object.select_arm(predicted_state)
            if survey == True:
                list_surveys.append(True)#
                list_predicted_states.append(predicted_state)#
                predicted_state = real_states[t]
                
                if real_states[t] >= threshold:
                    self.linucb_policy_object.linucb_arms[survey].reward_update(self.reward, predicted_state)
                
                else:
                    self.linucb_policy_object.linucb_arms[survey].reward_update(-self.reward/2, predicted_state)
                    
            elif survey == False:
                list_surveys.append(False)#
                list_predicted_states.append(predicted_state)#
        return real_states, list_predicted_states, list_surveys


    def performance_linUCB(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_day = 5
            first_state = 4
            threshold = 6
            real_states, list_predicted_states, list_surveys = self.linUCB(first_state, nb_day, threshold)
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += self.reward
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
                    list_regret.append(pseudo_regret_cumulatif)
        #plt.plot(list_regret)
        return np.array([[vp, fn],[fp, vn]]), list_regret
    
    def performance_random(self):
        fp = 0
        fn = 0
        vp = 0
        vn = 0
        pseudo_regret_cumulatif = 0
        list_regret = []
        for i in range(self.iteration_plot):
            nb_day = 30
            first_state = 3
            threshold = 9
            real_states = self.predict_states(first_state, nb_day, True)
            list_surveys = [random.choice([True, False]) for i in range(len(real_states)-1)]
            
            for t in range(nb_day):
                if real_states[t+1] >= threshold and list_surveys[t] == True:
                    vp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] >= threshold and list_surveys[t] == False:
                    fn += 1
                    pseudo_regret_cumulatif += self.reward
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == True:
                    fp += 1
                    list_regret.append(pseudo_regret_cumulatif)
                elif real_states[t+1] < threshold and list_surveys[t] == False:
                    vn += 1
                    list_regret.append(pseudo_regret_cumulatif)
        return np.array([[vp, fn],[fp, vn]]), list_regret
    
    def plot_confusion_matrix(self, confusion_matrix, title):
        #ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
        #                 fmt='.2%', cmap='Blues')
        ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        ax.set_title(title)
        plt.show()
    
    def UCBvslinUCBvsRandom(self):
        #list_list_regret_UCB = []
        list_list_regret_linUCB = []
        list_list_regret_random = []#
        
        list_list_confusion_UCB = []
        list_list_confusion_linUCB = []
        list_list_confusion_random = []#
        for i in range(100):
            #matrix, list_regret_UCB = self.performance_UCB()
            #list_list_regret_UCB.append(list_regret_UCB)
            #list_list_confusion_UCB.append(matrix)
            matrix, list_regret_linUCB = self.performance_linUCB()
            list_list_regret_linUCB.append(list_regret_linUCB)
            list_list_confusion_linUCB.append(matrix)
            matrix, list_regret_random = self.performance_random()#
            list_list_regret_random.append(list_regret_random)#
            list_list_confusion_random.append(matrix)#
           
        #list_list_regret_UCB = np.array(list_list_regret_UCB)
        list_list_regret_linUCB = np.array(list_list_regret_linUCB)
        list_list_regret_random = np.array(list_list_regret_random)#
        list_list_confusion_UCB = np.array(list_list_confusion_UCB)
        list_list_confusion_linUCB = np.array(list_list_confusion_linUCB)
        list_list_confusion_random = np.array(list_list_confusion_random)
        
        #Confusion matrix
        np.set_printoptions(suppress=True)
        #mean_confusion_UCB = np.mean(list_list_confusion_UCB, axis=0)
        #self.plot_confusion_matrix(mean_confusion_UCB)
        mean_confusion_linUCB = np.mean(list_list_confusion_linUCB, axis=0)
        self.plot_confusion_matrix(mean_confusion_linUCB, 'LinUCB')
        mean_confusion_random = np.mean(list_list_confusion_random, axis=0)#
        self.plot_confusion_matrix(mean_confusion_random, 'Random')
        
        #mean_UCB = np.mean(list_list_regret_UCB, axis=0)
        #std_UCB = np.std(list_list_regret_UCB, axis=0)
        #UCB_upper = mean_UCB + std_UCB
        #UCB_lower = mean_UCB - std_UCB
        
        mean_linUCB = np.mean(list_list_regret_linUCB, axis=0)
        std_linUCB = np.std(list_list_regret_linUCB, axis=0)
        linUCB_upper = mean_linUCB + std_linUCB
        linUCB_lower = mean_linUCB - std_linUCB
        x = [i for i in range(len(mean_linUCB))]
        
        #mean_random = np.mean(list_list_regret_random, axis=0)#
        #std_random = np.std(list_list_regret_random, axis=0)#
        #random_upper = mean_random + std_random#
        #random_lower = mean_random - std_random#

        #plt.plot(mean_UCB, label='UCB mean')
        #plt.fill_between(x, UCB_lower, UCB_upper, alpha = 0.3)
        plt.plot(mean_linUCB, label='LinUCB mean')
        plt.fill_between(x, linUCB_upper, linUCB_lower, alpha = 0.3)
        #plt.plot(mean_random, label='Random mean')#
        #plt.fill_between(x, random_upper, random_lower, alpha = 0.3)#
        plt.xlabel('Day')
        plt.ylabel('Mean cumulative regret')
        plt.legend(loc='upper right')
        plt.show()
            
    def plot_mean_states_matrices(self):
        nb_day = 30
        list_list_spike = []
        list_list_gaussian = []
        list_list_augmented = []
        list_list_reduced = []
        for i in range(200):
            self.create_spike_matrix()
            list_list_spike.append(self.predict_states(5, nb_day, True))
            self.create_gaussian_matrix()
            list_list_gaussian.append(self.predict_states(1, nb_day, True))
            self.create_augmented_matrix()
            list_list_augmented.append(self.predict_states(1, nb_day, True))
            self.create_reduced_matrix()
            list_list_reduced.append(self.predict_states(9, nb_day, True))
            
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
        
        
        
x = partial_monitoring()
x.create_augmented_matrix()
x.create_black_box_gaussian()
x.UCBvslinUCBvsRandom()







    
    

