# movie-matchmaker

An application of collaborative filtering methods to the [MovieLens dataset](https://grouplens.org/datasets/movielens/). See the accompanying [report](docs/report.md) for more detail on the methodology used.

## Getting started

These instructions should get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You will need to install the (mostly normal) array of Python packages `numpy`, `pandas`, `matplotlib`, and `sklearn`.  All of these should be installable from your preferred package installer, for example with `conda`:
```
conda install <package>
```

### Installing
To install, simply navigate to where you want the project located on your machine and perform a `git clone`:
```
git clone https://github.com/stringkm/movie-matchmaker.git
```
To check that the project is speaking with your Python packages, run `utils.py`
```
cd movie-matchmaker
python utils.py
```
This will import all of the required packages and define some functions useful for the project. If you get message `Everything looks good!` then you can assume things are working.

### Testing
After this, you should be able to move into the directory
```
cd path/to/movie-matchmaker
```
and run `ratemovies.py` in the following manner
```
python ratemovies.py --userId=test --cutoff=5 --output=test.csv --method=cosine
```
You should see a prompt similar to the one below (the movie will very likely be different)
```
For each movie, type a numeric rating (0-5) or <Enter> if you haven't seen it.
What is your rating for "Toy Story (1995)":
```
Simply exit the process with either `exit` or `^C`:
```
For each movie, type a numeric rating (0-5) or <Enter> if you haven't seen it.
What is your rating for "Toy Story (1995)": exit

Exiting recommendation program
```
If you've made it to this point everything is probably set up correctly.

### Optional download
This project contains the latest [small development version](https://grouplens.org/datasets/movielens/latest/) of the MovieLens dataset as of May 2018, containing ~100,000 ratings. The interested developer might wish to apply this package to the [full stable benchmark](https://grouplens.org/datasets/movielens/20m/) version of ~20 million ratings. To do so, download the dataset from the linked website, unzip it, and modify the `FULL_DATA` parameter in `0_explore_data.ipynb` to point to the folder containing the data. To apply this to any other point in the analysis you will need to modify the files to point to that version of the `ratings.csv` and `movies.csv` files (see the `0_*.ipynb` files for more information on the contents of the dataset).

## Contents

For further detail on methodology, read the [project report](docs/report.md).

### Code
* `0_explore_data.ipynb`: data exploration of the [full stable benchmark](https://grouplens.org/datasets/movielens/20m/) version
* `0_explore_ratings.ipynb`: further data exploration focused on the ratings of the [small development version](https://grouplens.org/datasets/movielens/latest/)
* `1_pearson.ipynb`: implementation of collaborative filtering with Pearson correlation coefficient weights
* `2_cosine.ipynb`: implementation of collaborative filtering with vector cosine similarity weights
* `3_top_k.ipynb`: implementation of top-k collaborative filtering
* `ratemovies.py`: API to rate randomly selected movies, save those ratings, and compute (using either weight method) the top 5 and bottom 5 predicted rated movies
* `utils.py`: defines useful functions used throughout the project

### Folders
* `data/`: contents of the [small development version](https://grouplens.org/datasets/movielens/latest/) of the MovieLens dataset as of May 2018. See `0_explore_data.ipynb` for an exploration of the [full stable benchmark](https://grouplens.org/datasets/movielens/20m/) version of these files, which is quite similar to the small version. The actual analysis uses the small dataset throughout
* `docs/`: this README and other project documentation ([proposal](docs/proposal.pdf), [presentation slides](docs/presentation.pdf), and final [report](docs/report.md)
* `figures/`: figures from the analysis included in the project [report](docs/report.md)
* `processed/`: saved files created in `0_explore_data.ipynb` and practice ratings generated by the authors

## Authors
* Katelyn Stringer - Pearson correlation - [stringkm](https://github.com/stringkm)
* Alex Riley - Cosine similarity, top-k filtering - [ahriley](https://github.com/ahriley)

## Acknowledgements
This project was created as part of the Spring 2018 course [STAT 689: Statistical Computing with R and Python](https://longjp.github.io/statcomp/) taught by [Dr. James Long](https://github.com/longjp) at Texas A&M University.

We acknowledge the helpful advice contained in the following sources that helped us design and implement our algorithms:
* Michael Ekstrand, ["Similarity Functions for User-User Collaborative Filtering,"](https://grouplens.org/blog/similarity-functions-for-user-user-collaborative-filtering/) Grouplens (blog), October 24, 2013.
* Suresh Kumar Gorakala, [_Building Recommendation Engines_](https://www.packtpub.com/big-data-and-business-intelligence/building-recommendation-engines) (Birmingham, UK: Packt Publishing Ltd), 2016.
* James Long, ["Netflix Prize and Collaborative Filtering"](https://longjp.github.io/statcomp/lectures/collab_filter.pdf) (lecture, Statistical Computing in R and Python, Texas A&M University, College Station, TX), March 8, 2018.
* ["Netflix Prize"](https://www.netflixprize.com/), _Netflix, Inc._, accessed April 30, 2018.
* Ethan Rosenthal, ["Intro to Recommender Systems: Collaborative Filtering,"](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/) Data Piques (blog), November 2, 2015.