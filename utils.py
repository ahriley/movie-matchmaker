import pandas as pd
import numpy as np

DATA_DIR = 'data/'

# ratings.csv -- 'userId' gave 'movieId' a 'rating' at 'timestamp'
# 20,000,263 individual ratings
def load_ratings(filename=None,tstamp=False):
    if filename is None:
        filename = DATA_DIR+'ratings.csv'

    df = pd.read_csv(filename)
    if not tstamp:
        df.drop(columns=['timestamp'], inplace=True)

    return df

# movies.csv -- 'movieId' has 'title' and 'genres', 27,278 movies
# note: flat=False reads processed pickle file with genre data as cols
def load_movies(filename=None,flat=False):
    if filename is None:
        filename = DATA_DIR+'movies.csv'

    df = pd.read_csv(filename,index_col='movieId')
    """
    # code used to generate processed/movies_unflattened.pkl
    if not flat:
        # figure out list of possible genres
        genreset = set()
        for index, row in df.iterrows():
            genrelist = row['genres'].split('|')
            genreset = genreset.union(set(genrelist))

        # create columns (0 means genre doesn't apply)
        for genre in genreset:
            movies[genre] = 0

        for index, row in df.iterrows():
            genrelist = row['genres'].split('|')
            for genre in genrelist:
                df.loc[index, genre] = 1

        df.drop(columns=['genres'], inplace=True)
    """
    if not flat:
        df = pd.read_pickle('processed/movies_unflattened.pkl')
        df.set_index('movieId',inplace=True)

    return df

# maps 10,381 movieIds to 1,128 genome tags via a relevance score
def load_genome(filename=None, flat=False):
    if filename is None:
        filename = DATA_DIR+'genome-scores.csv'

    df = pd.read_csv(filename)

    if not flat:
        movies = sorted(list(set(df['movieId'])))
        labels = np.arange(1,1128+1,1)

        rel = df.as_matrix(columns=['relevance'])
        rel = np.reshape(rel, (10381, 1128))
        matr = np.insert(rel, 0, movies, axis=1)

        df = pd.DataFrame(matr)
        df.set_index(0, inplace=True)
        df.index.names = ['movieId']
        df.index = df.index.map(int)

    return df

# maps tagId to a tag: e.g. "zombies", "slavery"; 1,128 different tags
def load_genometags(filename=None):
    if filename is None:
        filename = DATA_DIR+'genome-tags.csv'

    return pd.read_csv(filename, index_col='tagId')

# 'userId' gave 'movieId' a 'tag' at 'timestamp', 465,564 rows
# NOTE: probably not useful, since it's user-specific input
def load_tags(filename=None,tstamp=False):
    if filename is None:
        filename = DATA_DIR+'tags.csv'

    df = pd.read_csv(filename)
    if not tstamp:
        df.drop(columns=['timestamp'], inplace=True)

    return df

# links 'movieId' to 'imdbId' and 'tmdbId'
# NOTE: probably not useful, only connects different databases
def load_links(filename=None):
    if filename is None:
        filename = DATA_DIR+'links.csv'

    return pd.read_csv(filename)
