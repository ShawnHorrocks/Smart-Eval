from datasets import load_dataset as ld
import csv

ds = ld("Lichess/chess-position-evaluations", split="train", streaming=True)

max_rows = 150000 
best = {}

print("Loading data...")

for i, row in enumerate(ds):
    if len(best) >= max_rows: break

    fen, depth = row["fen"], row["depth"]
    if fen in best and best[fen][0] >= depth:
        continue
    if row["cp"] is not None:
        cp = max(-1500, min(1500, row["cp"]))
    else:
        if row["mate"] > 0: cp = 1500
        else: cp = -1500
    best[fen] = (depth, cp)

with open("data/positions.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["fen", "eval_cp"])
    for fen, (_, cp) in best.items():
        w.writerow([fen, cp])

print(len(best), " unique positions")