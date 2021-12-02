import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'taxamount']
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
    # renaming columns for ease of reading
    df = df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms','calculatedfinishedsquarefeet':'sq_ft','taxvaluedollarcnt':'tax_value','taxamount':'tax_amount'})
    return df


def split_zillow(df):
    '''
    Takes in the zillow dataframe and returns train, validate, test subset dataframes
    '''

    # Test set is .2 of original dataframe
    train, test = train_test_split(df, test_size = .2, random_state=123)#, stratify= df.tax_value)
    # The remainder is here divided .7 to train and .3 to validate
    train, validate = train_test_split(train, test_size=.3, random_state=123)#, stratify= train.tax_value)
    return train, validate, test


def encode_zillow(df):
    '''
    This is encoding a few of me zillow columns for later modelling; it drops the original column 
    once it has been encoded
    
    '''
    cols_to_dummy = df['fips']
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis = 1)
    df.columns = df.columns.astype(str)
    df.rename(columns={'6037.0':'LA', '6059.0': 'Orange', '6111.0':'Ventura'}, inplace=True)
    df = df.drop(columns='fips')
    return df

