import numpy as np
import pandas as pd
import datetime
import argparse
from utils import *

# to supress divide by zero or NaN errors that are handled
ignore = np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--userId")
parser.add_argument("--cutoff")
parser.add_argument("--output")
parser.add_argument("--method")
args = parser.parse_args()

userId = args.userId
cutoff = int(args.cutoff)
output = args.output
method = args.method

# Read in the movies, timestamp baseline
movies = pd.read_csv('data/movies.csv',header=0)
t0 = datetime.datetime(year=1970, month=1, day=1)

# Sample function that records your answers
def randMovie(moviedf, num=25):
    rate_array, movie_index, tstamps = [], [], []
    print("For each movie, type a numeric rating (0-5) or <Enter> if you haven't seen it.")
    
    while len(rate_array) < num:
        rand = moviedf.sample()
        randtitle = rand.title.values[0]
        try:
            rating_ = input('What is your rating for "'+randtitle+'": ')
            rate_array.append(float(rating_))
            movie_index.append(int(rand.movieId.values))
            t1 = datetime.datetime.utcnow()
            tstamps.append(int((t1-t0).total_seconds()))
        except:
            pass
    
    return movie_index, rate_array, tstamps

# call function to record ratings
ratemovies, ratings, tstamp = randMovie(movies, num=cutoff)
user = [userId for a in ratemovies]

# format to dataframe, save info
df = {'userId': user, 'movieId': ratemovies, 'rating': ratings, 'timestamp': tstamp}
df = pd.DataFrame(df)
df = df[['userId', 'movieId', 'rating', 'timestamp']]
df.to_csv(output,index=False,header=True)
print()
print("Ratings saved to "+output)

# import new user ratings
mine = pd.read_csv(output,header=0)
mine['movieId'] = [int(a) for a in mine['movieId']]
mine['userId'] = np.inf
mine = mine.loc[mine['movieId']>0]

# import all ratings, concatenate new ratings
ratings = pd.read_csv('data/ratings.csv',header=0)
ratings = pd.concat([ratings,mine],ignore_index=True)
ratings = ratings.pivot_table(index='movieId',columns='userId',values='rating')
ratings.fillna(0, inplace=True)

print("Computing weights...")
# compute weights
if method == 'pearson':
    similarity = ratings.T.corr()
elif method == 'cosine':
    similarity = cosine_sim(ratings)
print("Finished computing weights")

# predict ratings
prediction = predict_full(ratings, similarity)
alex_pred = prediction[np.inf].sort_values(ascending=False)

movies = pd.read_csv('data/movies.csv', index_col='movieId')

# print out top/worst predictions
print("Highest predicted rating (highest on top)")
print(movies.loc[alex_pred.index[:5]])
print()
print("Lowest predicted rating (lowest on top)")
print(movies.loc[alex_pred.index[:-5-1:-1]])
