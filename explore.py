import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
# import acquire
# import prepare
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

categoricals = ['county']
continuous = ['bedrooms','bathrooms','sq_ft','tax_value','tax_amount','age']

def plot_variable_pairs(x,y, df):
    '''
    Accepts the string name of an x variable (one continuous feature) and y variable, as well as the df name
    and returns a joinplot for them, including regression line.
    
    '''
    sns.jointplot(x=x, y=y, data=df,  kind='reg', height=5)
    plt.show()

def months_to_years(df):
    '''
    Takes a dataframe with a tenure column (in months) and adds a tenure_years column
    
    '''
    df['tenure_years'] = round(df.tenure/12).astype(int)

def plot_categorical_and_continuous_vars(cat_vars,cont_vars,df):
# This cell is returning 4 graphs for each of the 10 categorical variables (one graph for each continuous)
    for i, colx in enumerate(cat_vars):
        for coly in cont_vars:
            plt.figure(figsize=(25, 5))
            # i starts at 0, but plot nos should start at 1
            plot_number = i + 1 
            # Create subplot.
            plt.subplot(1,len(cat_vars), plot_number)
            # Title with column name.
            plt.title(colx)
            # Display histogram for column.
            sns.barplot(x=df[colx],y=df[coly], data=df),
            #they're all being drawn on the same plot
    #         sns.swarmplot(x=telco[colx],y=telco[coly], data=telco),
    #         sns.stripplot(x=telco[colx],y=telco[coly], data=telco)
            # Hide gridlines.
            plt.grid(False)
            plt.show()
            plt.tight_layout()
            
    for i, colx in enumerate(cat_vars):
        for coly in cont_vars:
            plt.figure(figsize=(25, 5))
            # i starts at 0, but plot nos should start at 1
            plot_number = i + 1 
            # Create subplot.
            plt.subplot(1,len(cat_vars), plot_number)
            # Title with column name.
            plt.title(colx)
            # Display histogram for column.
            sns.swarmplot(x=df[colx],y=df[coly], data=df),
    #         sns.stripplot(x=telco[colx],y=telco[coly], data=telco)
            # Hide gridlines.
            plt.grid(False)
            plt.show()
            plt.tight_layout()

    for i, colx in enumerate(cat_vars):
        for coly in cont_vars:
            plt.figure(figsize=(25, 5))
            # i starts at 0, but plot nos should start at 1
            plot_number = i + 1 
            # Create subplot.
            plt.subplot(1,len(cat_vars), plot_number)
            # Title with column name.
            plt.title(colx)
            # Display histogram for column.
            sns.stripplot(x=df[colx],y=df[coly], data=df)
            # Hide gridlines.
            plt.grid(False)
            plt.show()
            plt.tight_layout()