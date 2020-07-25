import pickle
import pandas as pd
import pdb

with open('../data/multi_dim/multi_dim_1979.pickle', 'rb') as file:
    test =pickle.load(file)

test2 = pd.read_csv('../data/remake/remake_1979.csv', index_col=0)
pdb.set_trace()

