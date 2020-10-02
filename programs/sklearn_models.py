import math
import numpy as np 
from numpy import linalg as LA
import random
import copy

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score, mean_squared_error, log_loss, r2_score, explained_variance_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.svm import LinearSVC

def class_error(x,y,sample_weight=None):
    return 1-accuracy_score(x,y,sample_weight=sample_weight)

class identity:
    def __init__(self):
        pass
    def transform(self,X):
        return(X)
    
class SVM_class:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = LinearSVC(penalty='l2', dual=False)
        param_grid = {'C':np.linspace(.0001,20,50)}
        best_c = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', n_jobs=-1).fit(X, y,sample_weight).best_params_['C']
        return best_c
    
    def fit(self,X,y,sample_weight=None):
        c=self.cv(X,y,sample_weight)
        model=LinearSVC(penalty='l1', dual=False,C=c,random_state=0).fit(X,y,sample_weight)
        return model
    
    def identity(self,X): 
        return identity()

class Logreg_class:
    def __init__(self,reg='l2'):
        self.reg=reg
    
    def cv(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = LogisticRegression()
        param_grid = {'C': list(np.linspace(.001,50,50))}
        self.best_c = GridSearchCV(model, param_grid, cv=2, scoring='neg_log_loss', n_jobs=-1).fit(X, y,sample_weight).best_params_['C']
    
    def fit(self,X,y,sample_weight=None):
        c=self.cv(X,y,sample_weight)
        model=LogisticRegression(penalty=self.reg,C=self.best_c,random_state=0,solver='liblinear', n_jobs=-1).fit(X,y,sample_weight)
        return model
    
    def identity(self,X): 
        return identity()
    
class KDE:
    def __init__(self):
        self.model=None
    
    def cv(self,X):
        model = KernelDensity()
        param_grid = {'bandwidth': np.linspace(0.01, 1.0, 30)}
        best_b = GridSearchCV(model, param_grid, cv=2, n_jobs=-1).fit(X).best_params_['bandwidth']
        return best_b
    
    def fit(self,X):
        b=self.cv(X)
        self.model=KernelDensity(bandwidth=b).fit(X)
        return self.model
    
class RF_class:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): 
        model = RandomForestClassifier(max_features='sqrt')
        param_grid = {'n_estimators': [200,300], 'min_samples_leaf':[15,25]} # 
        cv = GridSearchCV(model, param_grid, cv=2, scoring='roc_auc_ovr', n_jobs=-1).fit(X, y, sample_weight)
        return cv.best_params_['n_estimators'],cv.best_params_['min_samples_leaf']

    def fit(self,X,y,sample_weight):
        pars=self.cv(X,y,sample_weight)
        model=RandomForestClassifier(n_estimators=pars[0], min_samples_leaf=pars[1],random_state=0,max_features='sqrt').fit(X,y,sample_weight)
        return model

    
    
class DT_class:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): 
        d=np.shape(X)[1] #np.maximum(int(np.shape(X)[1]/3),2)
        
        model = DecisionTreeClassifier()
        param_grid = {'min_samples_leaf':[5,15,25,40,50]} # 
        cv = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', n_jobs=-1).fit(X, y, sample_weight)
        return cv.best_params_['min_samples_leaf']

    def fit(self,X,y,sample_weight=None):
        d=np.shape(X)[1]
        
        pars=self.cv(X,y,sample_weight)
        model=DecisionTreeClassifier(min_samples_leaf=pars,random_state=0).fit(X,y,sample_weight)
        return model
    
    def identity(self,X): 
        return identity()
    
    
class DT_reg:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): 
        d=np.shape(X)[1] #np.maximum(int(np.shape(X)[1]/3),2)
        
        model = DecisionTreeRegressor()
        param_grid = {'min_samples_leaf':[5,15,25,40,50]} # 
        cv = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1).fit(X, y, sample_weight)
        return cv.best_params_['min_samples_leaf']

    def fit(self,X,y,sample_weight=None):
        d=np.shape(X)[1]
        
        pars=self.cv(X,y,sample_weight)
        model=DecisionTreeRegressor(min_samples_leaf=pars,random_state=0).fit(X,y,sample_weight)
        return model
    
    def identity(self,X): 
        return identity()
    
class RF_reg:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): 
        d=np.shape(X)[1] #np.maximum(int(np.shape(X)[1]/3),2)
        
        model = RandomForestRegressor(max_features=d)
        param_grid = {'n_estimators': [100,300], 'min_samples_leaf':[15,25]} # 
        cv = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1).fit(X, y, sample_weight)
        return cv.best_params_['n_estimators'],cv.best_params_['min_samples_leaf']

    def fit(self,X,y,sample_weight):
        d=np.shape(X)[1]
        
        pars=self.cv(X,y,sample_weight)
        model=RandomForestRegressor(n_estimators=pars[0], min_samples_leaf=pars[1],random_state=0,max_features=d).fit(X,y,sample_weight)
        return model
    
    def fit_select(self,X,y,sample_weight):
        d=np.shape(X)[1]
        
        pars=self.cv(X,y,sample_weight)
        model=RandomForestRegressor(n_estimators=pars[0], min_samples_leaf=pars[1],random_state=0,max_features=d).fit(X,y,sample_weight)
        model_select=SelectFromModel(model, prefit=True)
        return model_select
    
    def identity(self,X): 
        return identity()
        
class Lasso_reg:
    def __init__(self):
        pass
    
    def cv_Ridge(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = Ridge()
        param_grid = {'alpha':  list(np.linspace(.01,25,50))}
        best_a = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error').fit(X, y,sample_weight).best_params_['alpha']
        return best_a
    
    def cv_Lasso(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = Lasso()
        param_grid = {'alpha':  list(np.linspace(.01,10,20))}
        best_a = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error').fit(X, y,sample_weight).best_params_['alpha']
        return best_a
    
    def fit(self,X,y,sample_weight):
        X,y=np.asfortranarray(X),np.asfortranarray(y)
        alpha=self.cv_Ridge(X,y,sample_weight)
        model=Ridge(alpha=alpha,random_state=0).fit(X,y,sample_weight)
        return model
    
    def fit_select(self,X,y,sample_weight=None):
        X,y=np.asfortranarray(X),np.asfortranarray(y)
        alpha=self.cv_Lasso(X,y,sample_weight)
        model=Lasso(alpha=alpha,random_state=0).fit(X,y,sample_weight)
        model_select=SelectFromModel(model, prefit=True)
        return model_select
    
    def identity(self,X): 
        return identity()

class Ridge_reg:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = Ridge()
        param_grid = {'alpha': list(np.linspace(.01,25,50))}
        best_a = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error').fit(X, y,sample_weight).best_params_['alpha']
        return best_a
    
    def fit(self,X,y,sample_weight):
        alpha=self.cv(X,y,sample_weight)
        model=Ridge(alpha=alpha,random_state=0).fit(X,y,sample_weight)
        return model
    
    def identity(self,X): 
        return identity()
    

class K_Ridge_reg:
    def __init__(self):
        pass
    
    def cv(self,X,y,sample_weight): ##CV to determine C l2-reg
        D=np.shape(X)[1]
        model = KernelRidge(kernel='rbf', gamma=1/D)
        param_grid = {'alpha':  list(range(1,15))}
        best_a = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error').fit(X, y,sample_weight).best_params_['alpha']
        return best_a
    
    def fit(self,X,y,sample_weight):
        alpha=self.cv(X,y,sample_weight)
        D=np.shape(X)[1]
        model=KernelRidge(alpha=alpha, kernel='rbf', gamma=1/D).fit(X,y,sample_weight)
        return model
    
    def cv_Lasso(self,X,y,sample_weight): ##CV to determine C l2-reg
        model = Lasso()
        param_grid = {'alpha':  list(np.linspace(.01,10,20))}
        best_a = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error').fit(X, y,sample_weight).best_params_['alpha']
        return best_a
    
    def fit_select(self,X,y,sample_weight=None):
        X,y=np.asfortranarray(X),np.asfortranarray(y)
        alpha=self.cv_Lasso(X,y,sample_weight)
        model=Lasso(alpha=alpha,random_state=0).fit(X,y,sample_weight)
        model_select=SelectFromModel(model, prefit=True)
        return model_select
    
    def identity(self,X): 
        return identity()