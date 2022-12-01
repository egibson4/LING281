from scipy.stats import norm
from sklearn.model_selection import KFold
from IPython.lib.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# sklearn has accuracy, precision, recall functions
# his baseline for the lectures was a most frequent classifier: mine will be the basic feldman one

# train-test split using this package:
# for train_idx, test_idx in KFold(n_splits=10, shuffle=True).split(vowel_data):
#     train = vowel_data.iloc[train_idx]
#     test = vowel_data.iloc[test_idx]

# lecture slides are modeling vowels in F1 and F2 space, which means they are multivariate gaussians
# the feldman paper only models vowels in one of those formant spaces, so it uses univariate gaussians
# need to use multiple category case: Appendix A?
# need to fit MUc and SIGMAc?


vot_data = pd.read_csv("cues1_vot.csv")
vot_data = vot_data[['stop', 'vot', 'gender']]
# vot_data = vot_data[vot_data['stop'].isin(['P', 'B'])]
vot_data = vot_data[vot_data['gender'] == 'F']
# calculate mean and cov for p and b
vot_means = vot_data.groupby('stop')[['vot']].mean()
vot_covs = vot_data.groupby('stop')[['vot']].cov()
print(vot_means)
# print(vot_covs)

# x = vot_data['vot'].sample().values[0]

# just the mean of the P(t | x, c) distribution: (cat_cov*x + noise_cov*cat_mean) / (cat_cov + noise_cov)
def vot_expected_percept_given_c(X, c, noise_cov=100*np.eye(1)):
    cat_cov = vot_covs.loc[c].values
    cat_mean = vot_means.loc[c].values
    
    result = (cat_cov*X + noise_cov*cat_mean) / (cat_cov + noise_cov)
    return result

def vot_expected_percept(X, noise_cov=100*np.eye(1)):
     # we should have the same number of likelihoods as the number of categories
    likelihoods = np.array([
        norm(
            mu, vot_covs.loc[stop] + noise_cov
        ).pdf(X)
        for stop, mu in vot_means.iterrows()
    ])

    # print("Likelihoods:", likelihoods)

    # if likelihoods is a one-dimensional array, then add one empty dimension to it?
    if len(likelihoods.shape) == 1:
        print("enter if")
        likelihoods = likelihoods[:, None]

    # for each likelihood, divide it by the total of all likelihoods (normalizing?)
    posteriors = np.array(likelihoods)/np.array(likelihoods).sum(0)
    # so what does the [None, :] do? looks like nothing bc the array dimensions are the same without it
    # print("Posteriors:", posteriors)
   
    percepts = np.array([
        vot_expected_percept_given_c(X, stop, noise_cov=noise_cov)
        for stop in vot_means.index
    ])

    # print("Raw Percepts:", percepts)
    # print("Adjusted posteriors:", posteriors[:, :, None])

    return np.sum(posteriors*percepts, axis=0)

# for _ in range(10):
#     x = vot_data[['vot']].sample().values
#     # print(x)
#     print("Sample:", x)
#     print("Percept:", vot_expected_percept(x))
#     print()


warping = vot_data[['stop', 'vot']].copy()
warping["noise"] = 0

for noise_var in np.concatenate([np.arange(100, 1000, 100), np.arange(1000, 10000, 1000), np.arange(10000, 110000, 10000)]):
  target = vot_expected_percept(
      vot_data[['vot']].values, 
      noise_cov=noise_var*np.eye(1)
  )
  target = pd.DataFrame(target, columns=['vot'])
  target['stop'] = vot_data['stop'].values
  target["noise"] = noise_var

  warping = pd.concat([warping, target], axis=0)

print(warping)