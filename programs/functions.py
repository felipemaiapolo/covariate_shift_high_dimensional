import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error





def mse_norm(y_hay, y): return mean_squared_error(y_hay, y)/np.var(y)

def softplus(x): return .001*np.log(1+np.exp(x))

def sigmoid(x): return 1/(1+np.exp(-np.minimum(x,15)))

def relu(x): return np.maximum(0,x)

def auc(y, probs, sample_weight=None):
    return roc_auc_score(y,probs)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



