
import numpy as np
import matplotlib.pyplot as plt
from implementations import *

# Load Data
DATA_TRAIN_PATH = './data/train.csv'
y, tX2, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# Clean Data
tX = clean_data(tX2)


# Build polynomial feature matrix
degree = 7
tx = build_poly(tX, degree)


## Run method, here least_squares
weights, loss = least_squares(y, tx)


# Generate prediction-file
DATA_TEST_PATH = './data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test = clean_data(tX_test)
tX_test = build_poly(tX_test, degree)

OUTPUT_PATH = './least_squares_final.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)



