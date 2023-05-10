import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, stats

import env
import wrangle as w

def plot_variable_pairs(df):
    sns.pairplot(df, kind="reg")
    plt.show()

def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 6))

    # Bar plot
    sns.barplot(x=categorical_var, y=continuous_var, data=df, ax=ax1)
    
    ax1.tick_params(axis='x', rotation=45)

    # Box plot
    sns.boxplot(x=categorical_var, y=continuous_var, data=df, ax=ax2)
    ax2.tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(x=categorical_var, y=continuous_var, data=df, ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.show()