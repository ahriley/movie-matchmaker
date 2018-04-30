import pandas as pd
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--userId")
parser.add_argument("--cutoff")
parser.add_argument("--output")
args = parser.parse_args()

userId = args.userId
cutoff = int(args.cutoff)
output = args.output

# Read in the movies, timestamp baseline
movies = pd.read_csv('data/movies.csv',header=0)
t0 = datetime.datetime(year=1970, month=1, day=1)

# Sample function that records your answers
def randMovie(moviedf, num=25):
    rate_array, movie_index, tstamps = [], [], []
    print("For each movie, type a numeric rating (0-5) or 'n' if you haven't seen it.")
    
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

ratemovies, ratings, tstamp = randMovie(movies, num=cutoff)
user = [userId for a in ratemovies]

df = {'userId':user, 'movieId': ratemovies, 'rating': ratings, 'timestamp': tstamp}
df = pd.DataFrame(df)
df = df[['userId', 'movieId', 'rating', 'timestamp']]
df.to_csv(output,index=False,header=True)
