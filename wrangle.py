import env
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_df():
    """
    This function will:
    - read from local directory for csv file
        - return if exists
    - Output zillow df
"""
    df = pd.read_csv("zillow.csv")
    return df

def get_zillow_data(directory=os.getcwd(), filename="zillow.csv"):
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
    if os.path.exists(directory + filename):
        df = pd.read_csv(filename) 
        return df
    
    else:
        url = env.get_db_url('zillow')
    
        df = pd.read_sql(SQL_query, url)
        
        #want to save to csv
        df.to_csv(filename)
        return df


def remove_outliers(df, exclude_column=None, threshold=4):
    """
    This function removes outliers from a pandas dataframe, with an option to exclude a single column.
    
    Args:
    df: pandas dataframe
    exclude_column: string, optional column name to exclude from outlier detection
    threshold: float, optional number of standard deviations from the mean to consider a value an outlier
    
    Returns:
    pandas dataframe with outliers removed
    """
    if exclude_column is not None:
        # Copy dataframe and drop excluded column
        df_clean = df.drop(columns=exclude_column)
    else:
        df_clean = df.copy()
    
    # Calculate z-score for each value
    z_scores = np.abs((df_clean - df_clean.mean()) / df_clean.std())
    
    # Remove rows with any value above threshold
    df_clean = df.loc[(z_scores < threshold).all(axis=1)]
    
    return df_clean

def remove_outliers(df, exclude_column=None, sd=3):
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
        
        # Show the plot
        plt.show()

def plot_boxplot(df):
    """
    Plots a histogram of each column in a pandas DataFrame using seaborn.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
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

def split_clean_zillow():
    """
    Remove nulls froms DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame split into train, validate, test with nulls and outliers removed.
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
    train, validate, test = split_data(df)
    return train, validate, test

def split_scale(df):
    """
    Remove nulls froms DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame split into train, validate, test with nulls and outliers removed.
    """
    df = df.dropna()
    df = df.drop(columns=["Unnamed: 0"])
    # rename columns 
    df = df.rename(columns={"bedroomcnt": "bedroom",
                        	"bathroomcnt": "bathroom",
                            "calculatedfinishedsquarefeet": "area",
                            "taxvaluedollarcnt": "property_value",
                            "yearbuilt": "year",
                            "taxamount": "tax"})
    df["fips"] = df.fips.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})
    dummy_df = pd.get_dummies(df[["fips"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.rename(columns={"fips_Orange": "orange", "fips_Ventura": "ventura"})
    df = df.drop(columns=["fips", "property_value"])
    df = remove_outliers(df)
    train, validate, test = split_data(df["bedroom", "bathroom", "area", "property_value", "year", "tax", "fips"])
    return train, validate, test

def mm_scale(df):
    train, validate, test = split_scale(df)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)


    x_train_scaled = scaler.transform(train)
    x_validate_scaled = scaler.transform(validate)
    x_test_scaled = scaler.transform(test)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled