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

def categorize_bathrooms(bathrooms):
    """
    Categorizes the column bathrooms into 'full' and 'half'
    """
    if bathrooms.is_integer():
        return 1
    else:
        return 0

def remove_outliers3(df, columns_to_skip=[], z_score_threshold=3):
    """
    Remove outliers from a dataframe while skipping specified columns.

    Args:
        df (pandas.DataFrame): The input dataframe.
        columns_to_skip (list): A list of column names to skip while removing outliers. Default is an empty list.
        z_score_threshold (float): The threshold value for the z-score. Values with a z-score higher than the threshold will be considered outliers. Default is 3.

    Returns:
        pandas.DataFrame: The dataframe with outliers removed.
    """
    df_cleaned = df.copy()

    for column in df.columns:
        if column in columns_to_skip:
            continue

        # Calculate z-scores for the column
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())

        # Filter out rows with z-scores higher than the threshold
        df_cleaned = df_cleaned[z_scores <= z_score_threshold]

    return df_cleaned

def split_scale(df):
    """
    Remove nulls froms DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame split into train, validate, test with nulls, outliers removed, and scaled on minmax.
    """
    df = df.dropna()
    df = df.drop(columns=["Unnamed: 0"])
    # rename columns 
    df = df.rename(columns={"bedroomcnt": "bedroom",
                        	"bathroomcnt": "bathroom",
                            "calculatedfinishedsquarefeet": "area",
                            "yearbuilt": "year",
                            "taxamount": "tax",
                            "taxvaluedollarcnt": "value"})
    df["fips"] = df.fips.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})

    # Add new column bathroom_type: full or half baths
    df['bathroom_full'] = df['bathroom'].apply(categorize_bathrooms)

    dummy_df = pd.get_dummies(df[["fips"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.rename(columns={"fips_Orange": "orange", "fips_Ventura": "ventura"})
    
    
    df = df.drop(columns=["fips"])
    df = remove_outliers(df)
    df = pd.DataFrame(df)

    df = df[["bedroom", "bathroom", "bathroom_full", "area", "year", "tax", "orange", "ventura", "value"]]
    train, validate, test = split_data(df)

    # Scale train, validate, and test
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)


    x_train_scaled = scaler.transform(train)
    x_validate_scaled = scaler.transform(validate)
    x_test_scaled = scaler.transform(test)
    # Convert the array to a DataFrame
    df_x_train_scaled = pd.DataFrame(x_train_scaled)
    df_train = df_x_train_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})
    x_train_scaled = df_train[["bedroom", "bathroom", "bathroom_full", "area", "year", "tax", "orange", "ventura"]]
    
    df_x_validate_scaled = pd.DataFrame(x_validate_scaled)
    df_validate = df_x_validate_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})
    x_validate_scaled= df_train[["bedroom", "bathroom", "bathroom_full", "area", "year", "tax", "orange", "ventura"]]
    
    df_x_test_scaled = pd.DataFrame(x_test_scaled)
    df_test = df_x_test_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})
    x_test_scaled = df_train[["bedroom", "bathroom", "bathroom_full", "area", "year", "tax", "orange", "ventura"]]
    # Y 
    
    y_train_scaled = df_train[["value"]]
    y_validate_scaled = df_validate[["value"]]
    y_test_scaled = df_test[["value"]]

    return x_train_scaled, x_validate_scaled, x_test_scaled, y_train_scaled, y_validate_scaled, y_test_scaled
    


def mm_scale(df):
    train, validate, test = split_scale(df)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)


    x_train_scaled = scaler.transform(train)
    x_validate_scaled = scaler.transform(validate)
    x_test_scaled = scaler.transform(test)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled


def split_scale1(df):
    """
    Remove nulls froms DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame split into train, validate, test with nulls, outliers removed, and scaled on minmax.
    """
    df = df.dropna()
    df = df.drop(columns=["Unnamed: 0"])
    # rename columns 
    df = df.rename(columns={"bedroomcnt": "bedroom",
                        	"bathroomcnt": "bathroom",
                            "calculatedfinishedsquarefeet": "area",
                            "yearbuilt": "year",
                            "taxamount": "tax",
                            "taxvaluedollarcnt": "value"})
    df["fips"] = df.fips.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})

    # Add new column bathroom_type: full or half baths
    df['bathroom_full'] = df['bathroom'].apply(categorize_bathrooms)

    dummy_df = pd.get_dummies(df[["fips"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.rename(columns={"fips_Orange": "orange", "fips_Ventura": "ventura"})
    
    
    df = df.drop(columns=["fips"])
    df = remove_outliers(df)
    df = pd.DataFrame(df)

    df = df[["bedroom", "bathroom", "bathroom_full", "area", "year", "tax", "orange", "ventura", "value"]]
    train, validate, test = split_data(df)

    # Scale train, validate, and test
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)


    x_train_scaled = scaler.transform(train)
    x_validate_scaled = scaler.transform(validate)
    x_test_scaled = scaler.transform(test)
    # Convert the array to a DataFrame
    df_x_train_scaled = pd.DataFrame(x_train_scaled)
    df_train = df_x_train_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})
   
    df_x_validate_scaled = pd.DataFrame(x_validate_scaled)
    df_validate = df_x_validate_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})

    df_x_test_scaled = pd.DataFrame(x_test_scaled)
    df_test = df_x_test_scaled.rename(columns={0: 'bedroom', 1: 'bathroom', 2: 'bathroom_full', 3: 'area', 4: 'year', 5: 'tax', 6: 'orange', 7:'ventura', 8: "value"})

    return df_train, df_validate, df_test
    