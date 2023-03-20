# initializing

import numpy as np
import pandas as pd
import statsmodels.api as sm


# create toy model
# define variable X and Y

n = 1000

np.random.seed(42)

X1 = np.random.normal(10, 1, n)
X2 = np.random.normal(20, 1, n)
X3 = np.random.normal(3, 1, n)
error = np.random.normal(0, 0.1, n)

Y = (0.2 * X1) + (0.3 * X2) - ((0.1 + error) * X3) + error

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

df['Y'].describe()
df['Y'] = df['Y'].apply(lambda x: 1 if x >= 7.7 else 0)

X = df[['X1', 'X2', 'X3']]
Y = df[['Y']]

def calc_contrib_ols(df, Y_col):
    X = df.drop(Y_col, axis=1)
    Y = df[Y_col]
    model = sm.OLS(Y, sm.add_constant(X)).fit(disp=0)
    r2 = model.rsquared
    
    # prepare some DataFrames as placeholder
    covariance = pd.DataFrame(columns=X.columns)
    coefficient = pd.DataFrame(columns=X.columns)
    R_squared = pd.DataFrame(columns=X.columns)
    y_variance = Y.var()
    
    # retrieve coefficient for each X variable from the full model
    for i, col in enumerate(X.columns):
        coef = model.params[i+1]
        coefficient[col] = [coef]
        
        # calculate covariance, r-squared, and weight
        for col in X.columns:

            cov_XY = np.cov(X[col], Y, bias=False, rowvar=False)[1, 0]
            cov_XY = 0 if np.isnan(cov_XY) else cov_XY
            covariance[col] = [cov_XY]
            
            R_squared[col] = covariance[col] /  y_variance * coefficient[col]
        
        # prepare dictionary for the result
        contrib = {}
        for col in X.columns:
            contrib[col] = R_squared[col][0]
        
        # check if decomposition equals r2
        contrib['Total'] = sum(contrib.values())
        contrib['R2'] = r2
    
    # create DataFrame for the final result
    contrib_df = pd.DataFrame.from_dict(contrib, orient='index', columns=['Contribution'])
            
    return contrib_df
    

def calc_contrib_logit(df, Y_col):
    X = df.drop(Y_col, axis=1)
    Y = df[Y_col]
    
    # perform logit regression for the full model, retrieve its McFadden's pseudo-R2
    model = sm.Logit(Y, sm.add_constant(X)).fit(disp=0)
    loglike_null = model.llnull
    pseudo_r2 = model.prsquared
    
    # prepare some DataFrames as placeholder
    covariance = pd.DataFrame(columns=X.columns)
    coefficient = pd.DataFrame(columns=X.columns)
    weights = pd.Series(index=X.columns, dtype=np.float64)
    R_squared = pd.DataFrame(columns=X.columns)
    y_variance = Y.var()
    
    # retrieve coefficient for each X variable from the full model
    for i, col in enumerate(X.columns):
        coef = model.params[i+1]
        coefficient[col] = [coef]
        
        # perform logit regression for each X variable and Y, then calculate covariance, "r-squared", and weight
        for col in X.columns:
            model2 = sm.Logit(Y, X[col]).fit(disp=0)

            cov_XY = np.cov(X[col], Y, bias=False, rowvar=False)[1, 0]
            cov_XY = 0 if np.isnan(cov_XY) else cov_XY
            covariance[col] = [cov_XY]
            
            R_squared[col] = covariance[col] /  (y_variance * coefficient[col])
            
            weights[col] = 1 - (model2.llf / loglike_null)
            
    # calculate total contribution 
    total_contrib = sum(weights[col] * R_squared[col][0] for col in X.columns)

    # prepare dictionary for the result
    contrib = {}
    
    # calculate the contribution of each X variable
    for col in X.columns:
        contrib[col] = weights[col] * R_squared[col][0] / total_contrib * pseudo_r2
    
    # check if decomposition equals McFadden's pseudo-r2
    contrib['Total'] = sum(contrib.values())
    contrib['Pseudo R2'] = pseudo_r2
    
    # create DataFrame for the final result
    contrib_df = pd.DataFrame.from_dict(contrib, orient='index', columns=['Contribution'])

    return contrib_df


# test
contributions = calc_contrib_ols(df, 'Y')
contributions
contributions2 = calc_contrib_logit(df, 'Y')
contributions2