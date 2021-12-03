import pandas as pd
import numpy as np
import matplotlib as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire




def prep_telco(df):
    """
    This function preps the telco data set be dropping several columns that I decided were
    not helpful for my analysis. The columnts dropped were:
        -payment_type_id, which was in reality a duplicate of the payment_type column
        -internet_service_type_id, which also duplicated info found in another column
        -contract_type_id, which also was duplicated info.
    After this step, I have replaced a small number of null values in the total_charges 
    field with 0, so they could be analyzed along with the rest of the total_charges
    (I decided this was the best course b/c I'm assuming they null values were new accounts
    with no charges yet; in any case, there were fewer than 20 of these nulls). The resulting
    values were converted to float datatype.

    Next, I dummied up a number of columns (please see list below) in order to encode them
    as 0 or 1 for analytical purposes; these dummy columns were finally concatenated to the
    original dataframe.
    
    """
    df = df.drop_duplicates()
    cols_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df = df.drop(columns = cols_to_drop)
    df.total_charges = df.total_charges.replace(' ',0)
    df.total_charges = df.total_charges.astype(float)

    # the following lines create bins for the respective columns
    df['monthly_charges_bins']= pd.qcut(df['monthly_charges'], 5)
    df['total_charges_bins']= pd.qcut(df['total_charges'], 8)
    #df.total_charges_bins = df.total_charges_bins.astype(float)
    cols_to_dummy = df[['gender','partner','dependents','phone_service','multiple_lines',
                        'online_security','online_backup','device_protection','tech_support','streaming_tv',
                        'streaming_movies','paperless_billing','churn','contract_type',
                        'internet_service_type','payment_type', 'monthly_charges_bins',
                        'total_charges_bins']]
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)

    return df

def cols_to_dummy(df):
    """
    This function dummies up all the string data columns of the dataframe into a new 
    dataframe, concatenates it on the existing dataframe, and then drops the original 
    columns. This will allow for easier modelling.
    
    """
    cols_to_drop = ['customer_id','gender', 'partner', 'dependents', 'phone_service', 
                    'multiple_lines', 'online_security','online_backup', 'device_protection',
                    'tech_support', 'streaming_tv', 'streaming_movies','paperless_billing',
                    'churn', 'contract_type', 'internet_service_type', 'payment_type'
                    ]
    dummy_df = pd.get_dummies(cols_to_drop, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)
    df = df.drop(columns = cols_to_drop)
    df = df.drop(columns = ['monthly_charges_bins','total_charges_bins'])

    return df


def split_telco_data(df):
    '''
    Takes in a the telco datafram and returns train, validate, test subsets. I have
    initially created a test_size of 20% for testing purposes (thus, leaving the 'train'
    set at 80% of the original); the 'train' set was split a second time into a 'validate'
    set (equalling 30% of the 80% 'train' left after the initial split) and a final 'train'
    which would be the remaining values.
    '''
    telco_train, telco_test = train_test_split(df, test_size = .2, stratify=df.churn_Yes)
    telco_train, telco_validate = train_test_split(telco_train, test_size=.3, stratify=telco_train.churn_Yes)
    return telco_train, telco_validate, telco_test

