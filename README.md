# Self organizing net

## Requirements
- python3.x (where x > 6)

## Download dataset
By default a filename seeds_dataset.txt is specified in main.py file. You can change this to other datasets with similar text file structure.

## How to run
- `git clone https://github.com/Wicwik/nn_som.git`
- `cd nn_som`
- `wget https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt`
- `python -m venv venv_nn_som`
- `. venv_nn_som/bin/activate`
- `pip install -r requirements.txt`
- `python main.py`