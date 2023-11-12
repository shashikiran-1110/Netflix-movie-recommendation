import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Movie Recommendation.csv', header = None)
transactions = []
for i in range(0, 7466):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

results = list(rules)

results

def inspect(results):
    movie_1         = [tuple(result[2][0][0])[0] for result in results]
    movie_2         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(movie_1, movie_2, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Movie 1', 'Movie 2', 'Support'])

resultsinDataFrame

resultsinDataFrame.nlargest(n = 10, columns = 'Support')
