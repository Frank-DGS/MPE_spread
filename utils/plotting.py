import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, weight=0.9):
    if len(x) == 0:
        return x
    sm = [x[0]]
    for v in x[1:]:
        sm.append(sm[-1] * weight + (1 - weight) * v)
    return np.array(sm)


def plot_training_curve(np_file: str, out_png: str, title: str = "MAPPO on MPE simple_spread"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    data = np.load(np_file)
    ep_returns = data if isinstance(data, np.ndarray) else data[()]
    if isinstance(ep_returns, np.lib.npyio.NpzFile):
        ep_returns = ep_returns["arr_0"]

    x = np.arange(len(ep_returns))
    y = np.array(ep_returns)
    ys = smooth(y, 0.9)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, alpha=0.3, label="raw")
    plt.plot(x, ys, label="ema(0.9)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


