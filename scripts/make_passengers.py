"""
Write kx/data/passengers.csv: 500 synthetic passengers with NaNs, categories and long labels, for the table kinds.
"""
import os

import numpy as np
import pandas as pd

N = 500
SEED = 4
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kx", "data", "passengers.csv")
TICKETS = ("standard single journey", "family return ticket", "first class saloon pass", "staff concession fare")


def make(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Passengers whose survival depends on sex, class and age, as in the real story."""
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.2, 0.55])
    sex = rng.choice(["female", "male"], n, p=[0.35, 0.65])
    age = np.clip(rng.normal(30 + 8 * (pclass == 1), 13, n), 0.5, 75).round(1)
    fare = rng.lognormal(np.log([60, 20, 9])[pclass - 1], 0.6).round(2)
    port = rng.choice(["Southampton", "Cherbourg", "Queenstown"], n, p=[0.7, 0.2, 0.1]).astype(object)
    deck = rng.choice(list("ABCDEFG"), n).astype(object)
    ticket = np.where(pclass == 1, TICKETS[2], rng.choice(TICKETS, n, p=[0.5, 0.3, 0.05, 0.15]))
    odds = 0.4 + 2.2 * (sex == "female") - 0.9 * (pclass - 1) - 0.02 * (age - 30)
    survived = (rng.random(n) < 1 / (1 + np.exp(-odds))).astype(int)
    deck[rng.random(n) < np.array([0.15, 0.8, 0.9])[pclass - 1]] = np.nan
    port[rng.random(n) < 0.015] = np.nan
    age = np.where(rng.random(n) < 0.2, np.nan, age)
    return pd.DataFrame({"age": age, "fare": fare, "sex": sex, "pclass": pclass, "port": port, "deck": deck,
                         "ticket_type": ticket, "survived": survived})


if __name__ == "__main__":
    make().to_csv(OUT, index=False)
    print(f"wrote {OUT}")
