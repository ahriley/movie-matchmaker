import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def cosine_sim(df):
    vals = df.values
    sim = vals @ vals.T
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim / norms / norms.T

def mse(pred, actual):
    pred = pred.values[actual.values > 0]
    actual = actual.values[actual.values > 0]
    finite = np.isfinite(pred)
    return mean_squared_error(pred[finite], actual[finite])

def predict_full(rates, weights):
    weights = np.nan_to_num(weights)
    pred = weights.dot(rates) / np.abs(weights).dot(rates.mask(rates>0, other=1.0))
    return pd.DataFrame(data=pred, index=rates.index, columns=rates.columns)

# TODO: don't go through numpy, just use pandas
def train_test_split(df, empty, testfrac=0.2):
    if np.isnan(empty):
        df = df.fillna(value=0.0)
    elif empty != 0.0:
        raise NotImplementedError("'empty' must be 0.0 or np.nan")
        
    vals = df.values
    test = np.zeros(vals.shape)
    train = vals.copy()
    for user in range(vals.shape[0]):
        nonzero = vals[user, :].nonzero()[0]
        test_ratings = np.random.choice(nonzero, size=int(testfrac*len(nonzero)), replace=False)
        train[user, test_ratings] = 0.0
        test[user, test_ratings] = vals[user, test_ratings]
        
    # Test and training are actually disjoint
    assert(np.all((train * test) == 0))
    
    train = pd.DataFrame(data=train, index=df.index, columns=df.columns)
    test = pd.DataFrame(data=test, index=df.index, columns=df.columns)
    if np.isnan(empty):
        test.replace(to_replace=0.0, value=np.nan, inplace=True)
        train.replace(to_replace=0.0, value=np.nan, inplace=True)
    
    return train, test

if __name__ == "__main__":
    print("Everything looks good!")