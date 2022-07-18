import plotly.graph_objects as go
import numpy as np
n_cores = 16
n_folds = 16
horizon = 100000
# outcome_distribution =  {'spam':0.05,'ham':0.95}
game = games.label_efficient(  )
algos = [ random_algo.Random( ),  ucb.UCB1(alpha),  linucb.LinUCB(alpha) ]
colors = [ [0,0,0], [29,176,0],  [155,155,0] ]
labels = [  'random' ,   'ucb1 (Auer)' , 'linUCB (Li)' ]
fig = go.Figure( )
for alg, color, label in zip( algos, colors, labels):
    r,g,b = color
    result = evaluate_parallel(n_cores, n_folds, horizon, alg, game)
    regret =  np.mean(result,0)
    xcoords = np.arange(0,horizon,1).tolist()
    std =  np.std(result,0)
    upper_regret = regret + std
    fig.add_trace(go.Scatter(x=xcoords, y=regret, line=dict(color='rgb({},{},{})'.format(r,g,b)), mode='lines',  name=label )) #
    fig.add_trace(   go.Scatter( x=xcoords+xcoords[::-1], y=upper_regret.tolist()+regret.tolist()[::-1],  fill='toself', fillcolor='rgba({},{},{},0.2)'.format(r,g,b),
                         line=dict(color='rgba(255,255,255,0)'),   hoverinfo="skip",  showlegend=True )
    )
fig.show(legend=True)
fig.update_yaxes(range=[0, 100] )
fig.show() (edited) 