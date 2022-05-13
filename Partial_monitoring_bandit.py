import numpy as np
from scipy import signal
import matplotlib.pyplot as plt




test = np.array([[0,1,0,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0],
                 [0,0,0,0,1,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1,0,0],
                 [0,0,0,0,1,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,1]])



#create spike_random matrix
spike_random = np.zeros((10,10))
s = signal.triang(19)
random = np.random.normal(0.1, 0.2, 100).reshape((10,10))
for i in range(len(spike_random)):
    spike_random[i] = s[9-i:19-i]
spike_random += random



#create spike_random matrix
gaussian_random = np.zeros((10,10))
for i in range(len(gaussian_random)):
    count, bins, ignored = plt.hist(np.random.normal(10, 1, 100), bins=range(0, 20,1))
    plt.clf()
    gaussian_random[i] = count[9-i:19-i]

print(gaussian_random)



class bandit:
    def __init__(self, markov):
        self.markov = markov
        self.states = []

    def find_states(self, first_state, nb_iteration):
        state = first_state
        for i in range(nb_iteration):
            self.states.append(state)
            state = np.argmax(self.markov[state])
    
    def plot_states(self):
        plt.figure(figsize=(5,5))
        plt.title('State evolution')
        plt.ylabel('State')
        plt.xlabel('Step')
        plt.plot(self.states)
        plt.show()
    

    
    
x = bandit(spike_random)

x.find_states(1, 20)
x.plot_states()
    
    
    
    
    
    
    

