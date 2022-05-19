import numpy as np
from scipy import signal
import matplotlib.pyplot as plt





#create spike_random matrix
spike_random = np.zeros((10,10))
s = signal.triang(19)
random = np.random.normal(0.1, 0.2, 100).reshape((10,10))
for i in range(len(spike_random)):
    spike_random[i] = s[9-i:19-i]
spike_random += random
spike_random = np.abs(spike_random)
row_sums = spike_random.sum(axis=1)
spike_random = spike_random / row_sums[:, np.newaxis]


#create gaussian_random matrix
gaussian_random = np.zeros((10,10))
for i in range(len(gaussian_random)):
    count, bins, ignored = plt.hist(np.random.normal(10, 1, 100), bins=range(0, 20,1))
    plt.clf()
    gaussian_random[i] = count[9-i:19-i]
gaussian_random = np.abs(gaussian_random)
row_sums = gaussian_random.sum(axis=1)
gaussian_random = gaussian_random / row_sums[:, np.newaxis]




#create gaussian_augmented_random matrix
gaussian_augmented_random = np.zeros((10,10))
for i in range(len(gaussian_augmented_random)):
    sigma = i*0.2
    count, bins, ignored = plt.hist(np.random.normal(10, sigma, 100), bins=range(0, 20,1))
    plt.clf()
    gaussian_augmented_random[i] = count[9-i:19-i]
gaussian_augmented_random = np.abs(gaussian_augmented_random)
row_sums = gaussian_augmented_random.sum(axis=1)
gaussian_augmented_random = gaussian_augmented_random / row_sums[:, np.newaxis]




#create gaussian_reduced_random matrix
gaussian_reduced_random = np.zeros((10,10))
for i in range(len(gaussian_reduced_random)):
    count, bins, ignored = plt.hist(np.random.normal(9, 2, 100), bins=range(0, 20,1))
    plt.clf()
    gaussian_reduced_random[i] = count[9-i:19-i]
gaussian_reduced_random = np.abs(gaussian_reduced_random)
row_sums = gaussian_reduced_random.sum(axis=1)
gaussian_reduced_random = gaussian_reduced_random / row_sums[:, np.newaxis]





class partial_monitoring:
    def __init__(self, markov):
        self.markov = markov
        self.randomized = True
        self.visited_states = []
        
    def next_state(self, current_state):
        if self.randomized == True:
            state = np.random.choice([i for i in range(10)], p=self.markov[current_state])
        else:
            state = np.argmax(self.markov[current_state])
        return state
    
    def predict_states(self, first_state, nb_predictions):
        self.visited_states.append(first_state)
        for i in range(nb_predictions):
            next_state = self.next_state(self.visited_states[-1])
            self.visited_states.append(next_state)

    
    def plot_states(self):
        plt.figure(dpi=1000)
        plt.ylabel('State')
        plt.xlabel('Step')
        plt.plot(self.visited_states)
        plt.show()
    
    
x = partial_monitoring(gaussian_reduced_random)
x.predict_states(8, 20)
x.plot_states()

    
    
    
    
    
    

