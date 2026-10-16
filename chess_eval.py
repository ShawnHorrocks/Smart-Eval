import chess
import numpy as np
import pandas as pd

N_COLS = 769

def main():
    X, y = load_train_data()

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
    df = pd.read_csv("data/train.csv")

    X = np.stack([translate_fen(f) for f in df["fen"]])
    y = df["eval_cp"].to_numpy(dtype=np.float32)

    np.savez("data/train.npz", X=X, y=y)

    return X, y

if __name__ == "__main__":
    main()