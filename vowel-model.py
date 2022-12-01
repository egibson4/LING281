from scipy.stats import norm
import numpy as np
import pandas as pd

# using means and covariances for /i/ and /e/ 
# from Feldman et al 2009, using data from Iverson & Kuhl 1995

# make a dataframe for the vowel means
means = {'f1': [224, 423], 'f2': [2413, 1936]}
vowel_means = pd.DataFrame(data=means, index=['i', 'e'])

cat_cov = 5873
noise_cov = 4443
# cat_cov + noise_cov
cov_sum = 10316

def vowel_expected_percept_given_c(X, c, noise_cov=100*np.eye(1)):
    cat_mean = vowel_means.loc[c].values
    
    result = (cat_cov*X + noise_cov*cat_mean) / (cat_cov + noise_cov)
    return result

def vowel_expected_percept(X, noise_cov=100*np.eye(1)):
    likelihoods = np.array([
        norm(
            mu, cat_cov + noise_cov
        ).pdf(X)
        for vowel, mu in vowel_means.iterrows()
    ])

    if len(likelihoods.shape) == 1:
        print("enter if")
        likelihoods = likelihoods[:, None]

    posteriors = np.array(likelihoods)/np.array(likelihoods).sum(0)
   
    percepts = np.array([
        vowel_expected_percept_given_c(X, vowel, noise_cov=noise_cov)
        for vowel in vowel_means.index
    ])

    return np.sum(posteriors*percepts, axis=0)