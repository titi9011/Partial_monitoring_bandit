import numpy as np
from polyagamma import random_polyagamma
from scipy.special import logit

class PGIDSratio():
    def __init__(self, game, horizon, d):

        self.game = game
        self.horizon = horizon
        self.N = game.n_actions

        self.gibbsits = 100
        self.d = d
        self.pmean = np.ones(self.d) * 0.5
    
        self.pcovar = np.identity( self.d )
        self.pcovar_inv = np.linalg.inv(self.pcovar)

        self.contexts = {'features':[], 'labels':[],'Vmat': np.identity(self.d) } 

    def thetagibbs(self, contexts, outcomes, initial_sample):

        thetamat = np.zeros( ( self.d, self.gibbsits+1 ) )
        thetamat[:,0] = initial_sample
        
        kappa = np.array(outcomes) - 0.5

        features = np.array(contexts)
        features = np.squeeze(features, 2)

        for m in range(1,self.gibbsits):
            omega = np.zeros( len(outcomes) )
            for i in range(len(outcomes)):
                omega[i] = random_polyagamma( 1 , features[i,] @ thetamat[:,m-1] , size=1 ) 

            Omegamat = np.diag( omega ) 

            Vomega   = np.linalg.inv(  features.T @ Omegamat @ features + self.pcovar ) #variance
            momega   = Vomega @ ( features.T @ kappa + self.pcovar_inv @ self.pmean ) #mean
            thetamat[:,m+1] = np.random.multivariate_normal(momega, Vomega, 1) 

        return thetamat

    def rewfunc(self, action, param, context):
        if action==0:
            return self.game.LossMatrix[1,0] * logit( param.T @ context ) + self.game.LossMatrix[0,0] * logit( -param.T @ context )
        elif action==1:
            return self.game.LossMatrix[1,1] * logit( param.T @ context ) + self.game.LossMatrix[0,1] * logit( -param.T @ context )

    def get_action(self,  t, X):

        if t == 0:
            # Always intervene on the first anomaly to get some data
            action = 1
            self.thetasamples = np.zeros( ( self.d, self.gibbsits) )
            self.thetasamples[:,-1] = self.pmean

        else:
            # Gibbs sampling
            self.thetasamples = self.thetagibbs( self.contexts['features'], self.contexts['labels'], self.thetasamples[:,-1] )

            # Compute gap estimates
            delta0, delta1 = np.zeros(self.gibbsits),np.zeros(self.gibbsits)
            for j in range(self.gibbsits):
                delta1[j] = max( self.rewfunc( 1, self.thetasamples[:,j], X), self.rewfunc(0, self.thetasamples[:,j], X) ) - self.rewfunc( 1, self.thetasamples[:,j], X )
                delta0[j] = max( self.rewfunc( 1, self.thetasamples[:,j], X), self.rewfunc(0, self.thetasamples[:,j], X) ) - self.rewfunc( 0, self.thetasamples[:,j], X )
        
            deltaone = np.mean(delta1)
            deltazero = np.mean(delta0)

            #Compute expected information gain
            tuneidsparam2 = min(1, deltazero / ( abs(deltazero-deltaone) ) ) 

            #Step 7) Select action
            action = np.random.binomial( 1, max(0,min(1,tuneidsparam2)) )
        
        return action

    def update(self,action, feedback, bandit_feedback, outcome, t, context):

        if action ==1 :
            self.contexts['features'].append( context )
            self.contexts['labels'].append( outcome )
            self.contexts['Vmat'] += context @ context.T

    def reset(self,):
        self.n = np.zeros( self.N )
        self.contexts = {'features':[], 'labels':[],'Vmat': np.identity(self.d) }