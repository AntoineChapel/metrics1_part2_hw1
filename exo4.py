import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

np.random.seed(6411)

N = 100
P = 30
b1_2 = np.array([1, 2])
b3_p = np.zeros((P-2))
beta = np.hstack((b1_2, b3_p)).reshape(-1, 1)

X = np.random.normal(0, 1, size=(N, P))
e = np.random.normal(0, 1, size=(N, 1))
y = X@beta + e

#OLS %*%
model1 = sm.OLS(y, X[:, 0:2]).fit(cov_type='HC1')
print(model1.summary())


#Lasso regression
lambda_vec = np.array([0.0, 0.05, 0.1, 0.2])

lasso_results = np.empty((P, lambda_vec.shape[0]))
for i, lambdaval in enumerate(lambda_vec):
    model2 = sm.OLS(y, X).fit_regularized(method='elastic_net', alpha=lambdaval, L1_wt=1.0)
    lasso_results[:, i] = (model2.params)

lasso_results_df = pd.DataFrame(np.round(lasso_results, 3), columns=["l = 0", "l = 0.05", "l = 0.1", "l=0.2"])
print(lasso_results_df)


#Lasso Cross-validation:
lasso_crossval = LassoCV(alphas=np.linspace(0.01, 5.01, 1000), cv=5)
lasso_crossval.fit(X, y.flatten())

print("\nBest lambda suggested by CV:", lasso_crossval.alpha_)
