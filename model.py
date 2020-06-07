import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('hiring.csv')

data['experience'].fillna(0, inplace=True)

data['test_score'].fillna(data['test_score'].mean(), inplace=True)

X = data.iloc[:, :3]


def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'eleven': 11, 'zero': 0, 0: 0,
                 'twelve': 12, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    return word_dict[word]


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = data.iloc[:, -1]

from sklearn.linear_model import LinearRegression

linReg = LinearRegression()

linReg.fit(X, y)

pickle.dump(linReg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[2, 9, 6]]))
