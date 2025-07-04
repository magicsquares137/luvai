import numpy as np
import pandas as pd 
from scipy.optimize import minimize 
from scipy import stats 
import plotly.offline 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 
import cufflinks as cf 
cf.go_offline()
cf.set_config_file(world_readable = True, theme = 'white')

class UserGenerator:
	def __init__(self):
		self.beta = {}
		self.beta['A'] = np.array([-4, -0.1, -3, -0.1])
		self.beta['B'] = np.array([-6, -0.1, 1, 0.1])
		self.beta['C'] = np.array([2, 0.1, 1, -0.1])
		self.beta['D'] = np.array([4, 0.1, -3, -0.2])
		self.beta['E'] = np.array([-0.1, 0, 0.5, -0.01])
		self.context = None # would be an array [1, device, location, age]
		# each represents a set of beta parameters for an ad a->e, where 
		# f_a(x) = beta_0 + beta_1*device + beta_2*location + beta_3*age

	def logistic(self, beta, context):
		# prob of a user click modeled by a logistic function of user param set and context
		return 1/(1 + np.exp(-np.dot(beta, context)))

	def display_ad(self, ad):
		# selection of an add, and subsequent use of that ads beta params and user context
		# to determine a click probability
		if ad in ['A', 'B', 'C', 'D', 'E']:
			p = self.logistic(self.beta[ad], self.context)
			reward = np.random_binomial(n=1, p=p)
			return reward
		else:
			raise Exception('Unknown ad')

	def generate_user_with_context(self):
		location = np.random_binomial(n=1, p=0.6) # 1 USA, 0 internat
		device = np.random_binomial(n=1,p=0.8) # 0 desktop, 1 mobile
		age = 10 + int(np.random,beta(2,3)*60)
		self.context = [1, device, location, age]
		return self.context

def get_scatter(x, y, name, showlegend):
    dashmap = {'A': 'solid',
               'B': 'dot',
               'C': 'dash',
               'D': 'dashdot',
               'E': 'longdash'}
    s = go.Scatter(x=x, 
                   y=y, 
                   legendgroup=name, 
                   showlegend=showlegend,
                   name=name, 
                   line=dict(color='blue', 
                             dash=dashmap[name]))
    return s
    
def visualize_bandits(ug):
    ad_list = 'ABCDE'
    ages = np.linspace(10, 70)
    fig = make_subplots(rows=2, cols=2, 
            subplot_titles=("Desktop, International", 
                            "Desktop, U.S.", 
                            "Mobile, International", 
                            "Mobile, U.S."))
    for device in [0, 1]:
        for loc in [0, 1]:
            showlegend = (device == 0) & (loc == 0)
            for ad in ad_list:
                probs = [ug.logistic(ug.beta[ad], 
                          [1, device, loc, age]) 
                                 for age in ages]
                fig.add_trace(get_scatter(ages, 
                                          probs, 
                                          ad, 
                                          showlegend), 
                           row=device+1, 
                           col=loc+1)             
    fig.update_layout(template="presentation")
    import plotly.offline as pyo
    pyo.plot(fig)

ug = UserGenerator()
visualize_bandits(ug)

