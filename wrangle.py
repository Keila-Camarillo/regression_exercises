import env
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_data(directory=os.getcwd(), filename="zillow.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output zillow df
"""
    SQL_query = "select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017 where propertylandusetypeid = '261'"
    if os.path.exists(directory):
        df = pd.read_csv(filename) 
        return df
    
    else:
        url = env.get_db_url("zillow")
    
        df = pd.read_sql(SQL_query, url)
        
        #want to save to csv
        df.to_csv(filename)
        return df
       
def remove_outliers(df, exclude_column=[["bathroom_full", "ventura" , "orange"]], sd=4):
    """
    Remove outliers from a pandas DataFrame using the Z-score method.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame with outliers removed.
    """
    num_outliers_total = 0
    for column in df.columns:
        if column == exclude_column:
            continue
        series = df[column]
        z_scores = np.abs(stats.zscore(series))
        num_outliers = len(z_scores[z_scores > sd])
        num_outliers_total += num_outliers
        df = df[(z_scores <= sd) | pd.isnull(df[column])]
        print(f"{num_outliers} outliers removed from {column}.")
    print(f"\nTotal of {num_outliers_total} outliers removed.")
    return df

def clean_zillow():
    """
    Remove nulls froms DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The prepped DataFrame with nulls and outliers removed.
    """
    df = pd.read_csv("zillow.csv")
    df = df.dropna()
    df = df.drop(columns=["Unnamed: 0"])
    # rename columns 
    df = df.rename(columns={"bedroomcnt": "bedroom",
                            "bathroomcnt": "bathroom",
                            "calculatedfinishedsquarefeet": "area",
                            "taxvaluedollarcnt": "property_value",
                            "yearbuilt": "year",
                            "taxamount": "tax"})
    df = remove_outliers(df,exclude_column=("fips"))
    df["fips"] = df.fips.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})
    dummy_df = pd.get_dummies(df[["fips"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.rename(columns={"fips_Orange": "orange", "fips_Ventura": "ventura"})
    
    return df

def remove_encodes(df):
    '''
    In the Zillow DataFrame, will only keep continous variables

    Return:
    pandas.DataFrame
    '''
    return df[['bedroom', 'bathroom', 'area', 'property_value', 'year', 'tax']]

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

def plot_histograms(df):
    """
    Plots a histogram of each column in a pandas DataFrame using seaborn.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    """
    # Loop through each column in the DataFrame
    for col in df.columns:
        # Create a histogram using seaborn
        sns.histplot(data=df, x=col)
        plt.xticks(rotation=45, ha='right')

        # Show the plot
        plt.show()

def plot_boxplot(df):
    """
    Plots a histogram of each column in a pandas DataFrame using seaborn.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    ** will error on category variables 
    """
    # Loop through each column in the DataFrame
    for col in df.columns:
        # Create a histogram using seaborn
        sns.boxplot(data=df, x=col)
        
        # Show the plot
        plt.show()

def split_data(df):
    '''
    Takes in two arguments the dataframe name and the ("stratify_name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123)
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123)
    return train, validate, test

def categorize_bathrooms(bathrooms):
    """
    Categorizes the column bathrooms into 'full' and 'half'
    """
    if bathrooms.is_integer():
        return 1
    else:
        return 0
    
def rename_col(df, list_of_columns=[]): 
    '''
    Take df with incorrect names and will return a renamed df using the 'list_of_columns' which will contain a list of appropriate names for the columns  
    '''
    df = df.rename(columns=dict(zip(df.columns, list_of_columns)))
    return df

def mm_scale(x_train, x_validate, x_test):
    '''
    Scales data on x_train, x_validate, x_test

    Returns: 
    pandas.DataFrame: x_train_scaled, x_validate_scaled, x_test_scaled

    ** will error with category variables -- remove beforehand
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)


    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    col_name = list(x_train.columns)

    x_train_scaled, x_validate_scaled, x_test_scaled = pd.DataFrame(x_train_scaled), pd.DataFrame(x_validate_scaled), pd.DataFrame(x_test_scaled)
    x_train_scaled, x_validate_scaled, x_test_scaled  = rename_col(x_train_scaled, col_name), rename_col(x_validate_scaled, col_name), rename_col(x_test_scaled, col_name)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled