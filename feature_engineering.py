import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

def select_kbest(x, y, k):
    '''
    Takes in the predictors (X), the target (y), and the number of features to select (k)
    
    Returns the names of the top k selected features based on the SelectKBest class
    '''
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(x, y)
    selected_features_mask = selector.get_support()
    selected_features = x.columns[selected_features_mask].tolist()
    return selected_features

def rfe(x, y, k):
    estimator = LinearRegression()  # Estimator used for feature ranking
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(x, y)
    selected_features_mask = selector.support_
    selected_features = x.columns[selected_features_mask].tolist()
    return selected_features

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def metric_df(baseline, rmse, r2):
    df = pd.DataFrame(data=[
        {
            'model':'baseline',
            'rmse':rmse,
            'r2':r2
        }

    ])
    return df