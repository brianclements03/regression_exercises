import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Titanic Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    The env file includes the username, password and host address as seen
    in the curly brackets
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    


def new_telco_data():
    '''
    This function reads the telco data from the Codeup database into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    First, it checks if the .csv already exists; if not, it will pull it off
    the Codeup server using a SQL query
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df




def new_mall_data():
    '''
    This function reads the mall_customers data from the Codeup database into a df.
    '''
    sql_query = """
                select * from mall_customers.customers
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    
    return df

def get_mall_data():
    '''
    This function reads in mall_customers data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    First, it checks if the .csv already exists; if not, it will pull it off
    the Codeup server using a SQL query
    '''
    if os.path.isfile('mall.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('mall.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_mall_data()
        
        # Cache data
        df.to_csv('mall.csv')
        
    return df