import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy
import acquire
import prepare
from scipy import stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")
# importing my personal wrangle module
import wrangle

# Ryan's code to spit out the list of features:

def show_features_rankings(X, rfe):
    """
    Takes in a dataframe and a fit RFE object in order to output the rank of all features
    """
    # rfe here is reference rfe from cell 15
    var_ranks = rfe.ranking_
    var_names = X.columns.tolist()
    ranks = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    ranks = ranks.sort_values(by="Rank", ascending=True)
    return ranks


def select_kbest(X,y,k):
    '''
    Function that returns top k features of a model using k best model,
    accepting X columns, y columns, and k number of top features.
    
    '''
    # K Best model here:

    from sklearn.feature_selection import SelectKBest, f_regression

    # parameters: f_regression stats test, give me 8 features
    f_selector = SelectKBest(f_regression, k=k)

    # find the top 8 X's correlated with y
    f_selector.fit(X, y)

    X_reduced = f_selector.transform(X)

    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = X.loc[:,feature_mask].columns.tolist()

    return f_feature


def select_rfe(X,y,k):
    '''
    Function that returns top k features of a model using rfe model,
    accepting X columns, y columns, and k number of top features.
    
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, k)

    # fit the data using RFE
    rfe.fit(X,y)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = X.iloc[:,feature_mask].columns.tolist()
    return rfe_feature