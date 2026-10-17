import os
import chess
import numpy as np
import pandas as pd

N_COLS = 769

def main():
    X, y = load_train_data()

    idx = np.arange(len(X))
    folds = np.array_split(idx, 4)
    test_idx = folds[0]
    train_idx = np.concatenate(folds[:0] + folds[1:])
    
    x_train, y_train, x_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    w = get_best_w(x_train, y_train)

    pred_y = predict(x_test, w)

    mae, r2 = compute_accuracy(pred_y, y_test)

    print("MAE: {:.4f} -- R2: {:.4f}".format(mae, r2))

def translate_fen(fen):
    board = chess.Board(fen)
    x = np.zeros(N_COLS, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        piece_index = piece.piece_type - 1
        if piece.color == chess.BLACK: piece_index += 6

        x[piece_index * 64 + square] = 1
    if board.turn == chess.WHITE: x[768] = 1
        
    return x

def load_train_data():
    if os.path.exists("data/train.npz"):
        if not os.path.getmtime("data/train.csv") > os.path.getmtime("data/train.npz"):
            data = np.load("data/train.npz")
            return data["X"], data["y"] 

    df = pd.read_csv("data/train.csv")

    X = np.stack([translate_fen(f) for f in df["fen"]])
    y = df["eval_cp"].to_numpy(dtype=np.float32)

    np.savez("data/train.npz", X=X, y=y)

    return X, y

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