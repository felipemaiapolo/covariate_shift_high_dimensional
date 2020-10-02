import io
import requests
import os
import sys
import math
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import mixture


def MI_gmm_reg(X,y,gmm,feat): #,X0,X1,gmm0,gmm1,alpha=1
    
    eps=10**-50
    n,d=X.shape
    components=gmm.n_components
    Z=np.hstack((y.reshape((-1,1)),X))
    feat2=[0]+[f+1 for f in feat]
    clf=gmm

    #(X,Y)
    like=np.zeros(n)
    for c in range(components):
        like+=clf.weights_[c]*multivariate_normal.pdf(Z[:,feat2], clf.means_[c][feat2], clf.covariances_[c][feat2][:,feat2])

    log_like_xy=np.log(like + eps)

    #(X)
    like=np.zeros(n)
    for c in range(components):
        like+=clf.weights_[c]*multivariate_normal.pdf(Z[:,feat2[1:]], clf.means_[c][feat2[1:]], clf.covariances_[c][feat2[1:]][:,feat2[1:]])

    log_like_x=np.log(like + eps)

    #(Y)
    like=np.zeros(n)
    for c in range(components):
        like+=clf.weights_[c]*multivariate_normal.pdf(Z[:,0], clf.means_[c][0], clf.covariances_[c][0][0])

    log_like_y=np.log(like + eps)
    
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}

def MI_gmm_class(X,y,gmm,feat):

    eps=10**-50
    n,d=X.shape
    feat2=[f for f in feat]

    classes=list(set(y))
    p={}

    #Y
    like=np.zeros(n)
    for c in classes:
        p[c]=np.mean(y==c)
        like[y==c]=p[c]

    log_like_y=np.log(like + eps)

    #(X,Y)
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux[y==c]+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[y==c][:,feat2], gmm[c].means_[comp][feat2], gmm[c].covariances_[comp][feat2][:,feat2])

        #(X,Y)
        like[y==c]=p[c]*like_aux[y==c] 
    log_like_xy=np.log(like + eps)

    #X
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[:,feat2], gmm[c].means_[comp][feat2], gmm[c].covariances_[comp][feat2][:,feat2])

        #(X,Y)
        like+=p[c]*like_aux

    log_like_x=np.log(like + eps)
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}

def MI(cand, posic, feat, X, y, gmm):  #, X0, X1, gmm0, gmm1, alpha=1
    n,d=X.shape
    aux = copy.deepcopy(posic)
    aux[feat] = cand
    
    if type(gmm)==dict:
        dic=MI_gmm_class(X,y,gmm,aux)
    else:
        dic=MI_gmm_reg(X,y,gmm,aux) #,X0,X1,gmm0,gmm1,alpha
    
    return cand, dic
    
class select_vars:
    
    def __init__(self, gmm, d=10, stop=.01):
        self.d=d
        self.stop=stop
        self.gmm=gmm
    
    def fit(self, X, y, verbose=True): #,X0,X1,gmm0,gmm1,alpha=1
        
        if verbose: print("Let's begin the selection...") 
            
        n,D=X.shape
        d=self.d
        gmm=self.gmm
        self.n=n
        
        
        posic=[]
        Js=[]
        stds=[]
        lista = list(range(D))

        for feat in range(d):

            posic.append(None)
            J_best=-math.inf
            
            outputs = [MI(cand, posic, feat, X, y, gmm) for cand in lista] #, X0, X1, gmm0, gmm1, alpha

            for out in outputs:
                
                cand, dic = out
                J_current = dic['mi']
                
                if J_current > J_best:
                    J_best=J_current
                    std_best=dic['std']
                    posic[feat] = cand
        
            lista.remove(posic[feat])
             
            Js.append(J_best)
            stds.append(std_best)

            
            #Stop
            if self.stop==None: 
                pass
            else:
                if feat>=1:
                    if Js[-1]/Js[-2]-1 < self.stop: break
            
            #Verbose
            if verbose: print("- Round={:2d} --- Î={:.2f} --- Selected Features={}".format(feat,np.round(J_best,2),posic))

                
        #Saving outputs
        self.loss_list=Js[:-1]
        self.stds_list=stds[:-1]
        self.var_list=posic[:-1]
        
        self.var_bool=np.zeros(D, dtype=bool)
        
        for var in self.var_list: 
            self.var_bool[var]=True
        
    def transform(self,X): return X[:,self.var_bool]
    
    def get_loss(self): return self.loss_list, self.stds_list
    
    def get_vars(self): return self.var_list
    
    def plot_loss(self): 
        
        l,s=self.get_loss()
        plt.errorbar(list(range(len(l))), l, yerr=(s/np.sqrt(self.n)))
        #plt.title("Mutual Information")
        plt.xlabel("Features")
        plt.ylabel("Mutual Information")
        plt.show()