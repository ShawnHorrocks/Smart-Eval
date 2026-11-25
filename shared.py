import os
import chess
import numpy as np
import pandas as pd

N_COLS = 769
SEED = 14

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

def split_data(n=None, seed=SEED):
    X, y = load_train_data()
    random = np.random.default_rng(seed)
    idx = random.permutation(len(X))
    if n is not None:
        idx = idx[:n]
    cut = int(len(idx) * .8)
    train = idx[:cut]
    val = idx[cut:]
    return X[train], y[train], X[val], y[val]

def compute_accuracy(pred_y, true_y):
    mae = np.mean(np.abs(pred_y - true_y))
    r2 = 1 - np.mean((pred_y - true_y) ** 2) / np.var(true_y)

    return mae, r2