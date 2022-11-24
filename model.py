import requests
import io, zipfile, scipy.io.wavfile
from IPython.lib.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# just using the data from lecture notes to figure out how to set up the model
url = 'https://web.archive.org/web/20200918131933/http://homepages.wmich.edu/~hillenbr/voweldata/bigdata.dat'
req = requests.get(url)

vowel_data_raw = [l.strip(' \r').split() 
                  for l in req.content.decode().split('\n') 
                  if len(l.split()) == 30]
# put F1 and F2 into pandas dataframe 
# (probably would be VOT for stops, spectral stuff for fricatives)
vowel_data = pd.DataFrame([l[3:5] for l in vowel_data_raw], index=[l[0] for l in vowel_data_raw], columns=['f1', 'f2']).astype(int)

vowel_data['speakertype'] = vowel_data.index.map(lambda x: x[0])
vowel_data['speakersex'] = vowel_data.index.map(lambda x: 'male' if x[0] in ['b', 'm'] else 'female')
vowel_data['speakerage'] = vowel_data.index.map(lambda x: 'adult' if x[0] in ['w', 'm'] else 'child')
vowel_data['speakerid'] = vowel_data.index.map(lambda x: x[1:3])
vowel_data['vowel'] = vowel_data.index.map(lambda x: x[3:])

vowel_data = vowel_data[(vowel_data.f1>0)&(vowel_data.f2>0)]

# print(vowel_data.head())

# might need to do some fourier transforms to get the data in a good format?
# sklearn has accuracy, precision, recall functions
# his baseline for the lectures was a most frequent classifier: mine will be the basic feldman one

# TODO: train-test split using this package:
from sklearn.model_selection import KFold
# for train_idx, test_idx in KFold(n_splits=10, shuffle=True).split(vowel_data):
#     train = vowel_data.iloc[train_idx]
#     test = vowel_data.iloc[test_idx]

# lecture slides are modeling vowels in F1 and F2 space, which means they are multivariate gaussians
# the feldman paper only models vowels in one of those formant spaces, so it uses univariate gaussians
# need to use multiple category case: Appendix A?
from scipy.stats import norm
# need to fit MUc and SIGMAc?
vowel_means = vowel_data.groupby('vowel')[['f1', 'f2']].mean()
vowel_means.sort_values('f2')
vowel_covs = vowel_data.groupby('vowel')[['f1', 'f2']].cov()

def expected_percept(X, noise_cov=10000*np.eye(2)):
    # calculates likelihoods for each vowel: need to edit this to use my data
    likelihoods = np.array([
        norm(
            mu, vowel_covs.loc[vowel] + noise_cov
        )
        for vowel, mu in vowel_means.iterrows()
    ])


vot_data = pd.read_csv("cues1_vot.csv")
vot_data = vot_data[['stop', 'vot', 'gender']]
vot_data = vot_data[vot_data['stop'].isin(['P', 'B'])]
vot_data = vot_data[vot_data['gender'] == 'F']
print(vot_data)

