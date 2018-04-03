# movie-matchmaker

## Requirements
* assumes the [MovieLens dataset](https://grouplens.org/datasets/movielens/20m/) is downloaded and its contents are set up in the same directory as the code in a folder named `data/` (this can be renamed but to do so the `DATA_DIR` needs to be set appropriately in `utils.py`)
* `numpy`, `pandas`, `matplotlib`

## Contents

#### Demos
* `read_data.ipynb`: example showing how to use the API built in `utils.py` to interact with the dataset with `pandas` commands

#### Processed Data
This folder is used to hold processed data that took awhile to compute (iterating over 10,000s of rows) but isn't that big storage-wise
* `movies_unflattened.pkl` is the contents of `movies.csv` but with each genre as a column and a 1 (0) if the genre applies (does not apply) to the movie
* `movies_withstats.pkl` is the same as `movies_unflattened.pkl` with two added columns for the total number of reviews each movie received and the average of those reviews
