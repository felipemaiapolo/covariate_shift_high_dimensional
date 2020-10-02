import math
import numpy as np 
from numpy import linalg as LA
import pandas as pd
import random
import copy
import multiprocessing as mp

from sklearn import ensemble
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score, mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KernelDensity
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from skopt import gp_minimize
from skopt import dummy_minimize
from programs.functions import *


class LSPC:
    def __init__(self, B, sigma, rho):
        self.B=B
        self.sigma=sigma
        self.rho=rho
        self.centers=None
        self.theta=None
    
    def fit(self,X,y):
        index=[random.choice(range(np.shape(X)[0])) for _ in range(self.B)]
        self.centers=X[index]
        pi=np.array(pd.get_dummies(y))
        pi=pi.reshape((np.shape(X)[0],-1))
        PHI=rbf_kernel(X, self.centers, 1/(2*self.sigma**2))
        self.theta=np.linalg.inv(PHI.T@PHI+self.rho*np.identity(self.B))@PHI.T@pi
        
    def predict_proba(self,X):
        PHI=rbf_kernel(X, self.centers, 1/(2*self.sigma**2))           
        y_proba=relu(PHI@self.theta)/np.sum(relu(PHI@self.theta), axis=1).reshape((-1,1))
        #y_proba=softplus(PHI@self.theta)/np.sum(softplus(PHI@self.theta), axis=1).reshape((-1,1))
        return y_proba
    
    def get_auc(self,X,y):
        y_proba=self.predict_proba(X)
        return auc(y.reshape((-1,1)),y_proba[:,1].reshape((-1,1)))


class LSPC_w:
    def __init__(self,B):
        self.model=None
        self.B=B
    
    def cv(self,X,y):
        
        Xw_train,Xw_test,yw_train,yw_test=train_test_split(X,y, test_size=0.3, random_state=1)

        def treinar_modelo(params):
            sigma = params[0]
            rho = params[1]
            model = LSPC(B=self.B, sigma=sigma, rho=rho)
            model.fit(Xw_train,yw_train)
            return -model.get_auc(Xw_test,yw_test)
        
        space = [(.01, 25.), #sigma
                 (.01, 25.)] #rho
        
        resultados_gp = gp_minimize(treinar_modelo, space, random_state=1, verbose=0, n_calls=50, n_random_starts=25)
        return resultados_gp.x
    
    def fit(self,X0,X1):
        X=np.vstack((X0, X1))
        y=np.hstack((np.zeros(np.shape(X0)[0]),np.ones(np.shape(X1)[0])))
        pars=self.cv(X,y)
        model=LSPC(B=self.B, sigma=pars[0], rho=pars[1])
        model.fit(X,y)
        self.model=model
    
    def predict(self, X):
        return 1/(self.model.predict_proba(X)[:,1])
    
############################################################

#Definindo função
def evaluate_Logreg_w(c, X_train, y_train, X_test, y_test, reg):
    model = LogisticRegression(penalty=reg, C=c, random_state=42, solver='liblinear', n_jobs=-1)
    model.fit(X_train,y_train)
    return [c, log_loss(y_test, model.predict_proba(X_test)[:,1])]
    
def valid_Logreg_w(self, X, y, reg='l1', set_cv=[(10**-4,.5),(.501,2),(2.001,5)]): 
    #Arrumando dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

    #Fazendo testes
    for interval in set_cv:
       
        #pool = mp.Pool(mp.cpu_count())
        #output = pool.starmap(evaluate_Logreg_w, [(c, X_train, y_train, X_test, y_test, reg) for c in np.linspace(interval[0],interval[1],12)]) 
        #pool.close()
        
        output=[evaluate_Logreg_w(c, X_train, y_train, X_test, y_test, reg) for c in np.linspace(interval[0],interval[1],8)]
        
        #Output
        output=np.array(output)
        index=np.argmin(output[:,1])
        
        if output[index][0]==interval[1]: continue
        else: break
  
    return output[index][0], output[index][1]

class Logreg_w:
    def __init__(self):
        self.eps=10**-5
    
    def fit(self,X0 ,X1, reg='l1'):
        
        #Arrumando dados
        X=np.vstack((X0, X1))
        y=np.hstack((np.zeros(np.shape(X0)[0]),np.ones(np.shape(X1)[0])))
        
        #Pegando melhores hiper.
        self.best_c, self.log_loss = valid_Logreg_w(self, X, y, reg)
        
        #Treinando modelo final
        model=LogisticRegression(penalty=reg,C=self.best_c,random_state=0,solver='liblinear', n_jobs=-1)
        model.fit(X,y)
        self.model=model
    
    def predict(self, X):
        return (self.model.predict_proba(X)[:,0])/(self.model.predict_proba(X)[:,1] + self.eps)
    
    def predict_mar(self, X):
        return 1/(self.model.predict_proba(X)[:,1]+ self.eps)
    
    def get_log_loss(self): return self.log_loss
 

class pca_98:
    def __init__(self):
        pass
    
    def fit(self,X):
        pca = PCA()
        pca.fit(X)
        cumsum=np.cumsum(pca.explained_variance_ratio_)
        index=1
        while cumsum[index]<.99:
            index+=1
        
        pca = PCA(n_components=index+1)
        pca.fit(X)
        return pca

###
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unique_columns(a):
    return unique_rows(a.astype(float).T).T

class Poly_Logreg_w:
    def __init__(self, degree=2):
        self.degree=degree
        self.eps=10**-5
    
    def fit(self,X0 ,X1, reg='l1'):
        
        #Arrumando dados
        X=np.vstack((X0, X1))
        y=np.hstack((np.zeros(np.shape(X0)[0]),np.ones(np.shape(X1)[0])))
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #self.pca=pca_98().fit(X)
        #X = self.pca.transform(X)
        
        #Pegando melhores hiper.
        self.best_c, self.log_loss = valid_Logreg_w(self, X, y, reg)
        #print(self.best_c)
      
        #Treinando modelo final
        model=LogisticRegression(penalty=reg,C=self.best_c,random_state=0,solver='liblinear', n_jobs=-1)
        model.fit(X,y)
        self.model=model
    
    def predict(self, X):
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        return (self.model.predict_proba(X)[:,0])/(self.model.predict_proba(X)[:,1] + self.eps)
    
    def predict_mar(self, X):
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        return 1/(self.model.predict_proba(X)[:,1]+ self.eps)
    
    def calc_log_loss(self, X): 
        poly = PolynomialFeatures(self.degree)
        X = poly.fit_transform(X)
        #X = unique_columns(X)
        #X = self.pca.transform(X)
        y = np.zeros(np.shape(X)[0])
        y[0]=1 #gambs
        return log_loss(y, self.model.predict_proba(X)[:,1])
         
    def get_log_loss(self): return self.log_loss