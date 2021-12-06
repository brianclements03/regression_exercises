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

###################### Acquire Titanic Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT \
                bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, \
                taxamount, fips\
                FROM zillow.properties_2017 AS zp LEFT JOIN zillow.propertylandusetype AS plt USING (propertylandusetypeid) \
                WHERE plt.propertylandusetypeid = 261'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in the zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df

def remove_outliers(df, k, col_list):
    ''' 
    
    Here, we remove outliers from a list of columns in a dataframe and return that dataframe
    
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def clean_and_prep_data(df):
    '''
    This function will do some light cleaning and minimal other manipulation of the zillow data set, 
    to get it ready to be split in the next step.
    
    '''
        # renaming columns for ease of reading
    df = df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms','calculatedfinishedsquarefeet':'sq_ft','taxvaluedollarcnt':'tax_value','taxamount':'tax_amount','fips':'county'})
    cols = ['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount']
    # replace fips codes with county names
    df.county = df.county.replace({6037.0:'LA',6059.0: 'Orange',6111.0:'Ventura'})
    # dropping nulls
    df = df.dropna()
    #removing outliers--see the function elsewhere in this file
    df = remove_outliers(df, 1.5, cols)
    # accessing datetime info so as to create an 'age' variable
    from datetime import date
    df.yearbuilt =  df.yearbuilt.astype(int)
    year = date.today().year
    df['age'] = year - df.yearbuilt
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['yearbuilt'])

    return df


def split_zillow(df):
    '''
    Takes in the zillow dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # SPLIT
    # Test set is .2 of original dataframe
    train, test = train_test_split(df, test_size = .2, random_state=123)#, stratify= df.tax_value)
    # The remainder is here divided .7 to train and .3 to validate
    train, validate = train_test_split(train, test_size=.3, random_state=123)#, stratify= train.tax_value)

    return train, validate, test


def encode_zillow(df):
    '''
    This is encoding a few of the zillow columns for later modelling; it drops the original column 
    once it has been encoded
    
    '''
    # ordinal encoder? sklearn.OrdinalEncoder

    cols_to_dummy = df['county']
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis = 1)
    #df.columns = df.columns.astype(str)
    # I ended up renaming counties in an above function; the other encoded cols are renamed here:
    #df.rename(columns={'6037.0':'LA', '6059.0': 'Orange', '6111.0':'Ventura'}, inplace=True)
    # I have commented out the following code bc i think i might want to have the county column for exploration
    #df = df.drop(columns='county')
    return df




def scale_zillow(train,validate,test):
    '''
    Takes in the zillow dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # SCALE
    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    # 2. fit the object
    scaler.fit(train)
    # 3. use the object. Scale all columns for now
    train = pd.DataFrame(scaler.transform(train), columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount', 'age',
       'LA', 'Orange', 'Ventura'])
    test = pd.DataFrame(scaler.transform(test), columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount', 'age',
       'LA', 'Orange', 'Ventura'])
    validate = pd.DataFrame(scaler.transform(validate), columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount', 'age',
       'LA', 'Orange', 'Ventura'])

    return train, validate, test
