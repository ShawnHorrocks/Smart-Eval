import os
import chess
import numpy as np
import pandas as pd
import shared as sh


def main():
    X, y = sh.load_train_data()

    idx = np.arange(len(X))
    folds = np.array_split(idx, 4)
    test_idx = folds[0]
    train_idx = np.concatenate(folds[:0] + folds[1:])
    
    x_train, y_train, x_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    w = get_best_w(x_train, y_train)

    pred_y = predict(x_test, w)

    mae, r2 = compute_accuracy(pred_y, y_test)

    print("MAE: {:.4f} -- R2: {:.4f}".format(mae, r2))



def get_best_w(X, y):
    w, *_ = np.linalg.lstsq(X.astype(np.float32), y, rcond=None)
    return w

def predict(X, w):
    return X @ w

def compute_accuracy(pred_y, true_y):
    mae = np.mean(np.abs(pred_y - true_y))
    r2 = 1 - np.mean((pred_y - true_y) ** 2) / np.var(true_y)

    return mae, r2

if __name__ == "__main__":
    main()