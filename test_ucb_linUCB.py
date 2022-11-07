import random
from UCB import UCB
from linUCB import linucb_policy
import matplotlib.pyplot as 
import numpy as np

list_list_regret_UCB = []
list_list_regret_linUCB = []
for j in range(100):
    context = [random.choices([0,1], weights=(9, 9)) for i in range(100)]

    peu_importe = 0

    
    UCB_ai = UCB(2)
    
    UCB_ai.t = 2
    
    UCB_ai.update(0, 1)
    UCB_ai.update(1, 1)
    
    list_regret_UCB = []
    cumulative_regret = 0
    for i in context:
        action = UCB_ai.get_action()
        if action == i:
            UCB_ai.update(i, 1)
    
        else:
            UCB_ai.update(i, 0)
            cumulative_regret += 1
        
        list_regret_UCB.append(cumulative_regret)
        
    
    linUCB = linucb_policy(2,1)
    linUCB.get_action(0, np.array([1]))
    linUCB.get_action(1, np.array([1]))
    
    linUCB.update(0, 1, peu_importe, 1, np.array([1]))
    linUCB.update(1, 1, peu_importe, 2, np.array([1]))
    
    list_regret_linUCB = []
    cumulative_regret = 0
    for t, t_context in enumerate(context):
        action = linUCB.get_action(t,np.array([t_context]))
        if action == t_context:
            linUCB.update(action, 1, peu_importe, t, np.array([t_context]))
    
        else:
            linUCB.update(action, 0, peu_importe, t, np.array([t_context]))
            cumulative_regret += 1
        
        list_regret_linUCB.append(cumulative_regret)
    
    plt.plot(list_regret_UCB)
    plt.plot(list_regret_linUCB)
    #list_list_regret_UCB.append(list_regret_UCB)
    #list_list_regret_linUCB.append(list_regret_linUCB)
    
    
plt.show()