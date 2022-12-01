import numpy as np
import matplotlib.pyplot as plt
from PMlinUCB import PM_linucb_policy
from UCB import UCB



class Evaluation:
    def __init__(self, algo):
        self.context = []
        self.solution = []
        self.horizon = 1000
        self.algo = algo
        self.iteration_for_mean = 10

    def find_solution(self, context):
        alpha_0 = 0
        beta_0 = 1
        alpha_1 = 1
        beta_1 = 0
        b_0 = context[0]*alpha_0 + context[1]*beta_0
        b_1 = context[0]*alpha_1 + context[1]*beta_1
        if b_0 > b_1:
            return 0
        elif b_0 < b_1:
            return 1

    def create_game(self):
        self.context = []
        self.solution = []
        #self.context = np.random.rand(self.horizon,2)
        list_01 = np.random.randint(2, size=self.horizon)
        for i in list_01:
            self.context.append([i,[1,0][i]])
        self.context = np.array(self.context)
        for i in range(self.horizon):
            self.solution.append(self.find_solution(self.context[i]))

        
    def eval_policy_once(self):
        list_gather_info = []
        for t in range( self.horizon):
            gather_info = self.algo.get_action(t, self.context[t])
            list_gather_info.append(gather_info)
            print(gather_info, self.solution[t]==gather_info)
            if gather_info == True:
                if gather_info == self.solution[t]:
                    self.algo.update(gather_info, 1, 0, t, self.context[t])

                else:
                    self.algo.update(gather_info, 0, 0, t, self.context[t])

            elif gather_info == False:
                '''
                if gather_info == self.solution[t]:
                    self.algo.update(gather_info, 1, 0, t, self.context[t])

                else:
                    self.algo.update(gather_info, 0, 0, t, self.context[t])
                '''


        return self.solution, list_gather_info



    def performance_algo_once(self):
        list_regret = []
        cumulative_regret = 0
        solution, list_gather_info = self.eval_policy_once()
        for t in range(self.horizon):
            if solution[t] == list_gather_info[t]:
                cumulative_regret += 0
            elif solution[t] != list_gather_info[t]:
                cumulative_regret += 1
            list_regret.append(cumulative_regret)
                
        return list_regret
    
    def performance_algo(self, algo_name):
        list_list_cumulative_regret = []
        
        for i in range(self.iteration_for_mean):
            self.algo.clean_all_variables()
            self.create_game()
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







algo = PM_linucb_policy(2,2)

x = Evaluation( algo)
x.performance_algo('PMLinUCB')


'''
plt.plot(np.array(algo.list_UCB[1]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_UCB[0]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_bound[0]).reshape(-1,1))
plt.show()
plt.plot(np.array(algo.list_bound[1]).reshape(-1,1))
plt.show()
'''


algo = UCB(2)

x = Evaluation( algo)
x.performance_algo('UCB')






