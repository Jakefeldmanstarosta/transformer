import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import *


RESULTS_PATH = r"C:\Users\jakef\Projects\next-token-prediction-control\results\stability"
os.makedirs(RESULTS_PATH, exist_ok=True)

def plot_loss_grid(delta_T_values, delta_Q_values, grid, PATH):
    nx = len(delta_T_values)
    ny = len(delta_Q_values)

    plt.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), grid)
    for i in range(ny):
        for j in range(nx):
            plt.text(j + 0.5, i + 0.5, f"{grid[i, j]:.2f}", ha='center', va='center')

    plt.xticks(np.arange(nx) + 0.5, [f"{v:.2f}" for v in delta_T_values])
    plt.yticks(np.arange(ny) + 0.5, [f"{v:.2f}" for v in delta_Q_values])
    plt.xlabel("delta(T)")
    plt.ylabel("delta(Q)")
    plt.savefig(os.path.join(PATH, "colour-gird.png"), bbox_inches='tight')
    plt.close()
    
def generate_pairs(n, K_tilde, N, transition_kernel, observation_kernel, mode = "filter", **kwargs):
    # discrete state process X_{t+1} ~ T_kernel[X_t] and observation Y_t ~ Q_kernel[X_t]
    # mode == "filter": predict hidden state X_t from Y_[t - N + 1, t]
    # mode == "predict": predict next observation Y_{t + 1} from Y_[t - N + 1, t]

    X = []
    Y = []
    n_samples = K_tilde + N - 1
    #prediction requires one extra sample
    if mode == "predict":
        n_samples += 1

    X.append(0)
    for r in range(1, n_samples):
        X.append(int(np.random.choice(n, p = transition_kernel[X[-1]])))

    for r in range(n_samples):
        Y.append(int(np.random.choice(n, p = observation_kernel[X[r]])))

    x = []
    y_tilde = []
    for r in range(K_tilde):
        x.append(Y[r: r+N])
        if mode == "predict":
            y_tilde.append(Y[r + N])
        elif mode == "filter":
            y_tilde.append(X[r + N - 1])
        else:
            print("Incorrect mode")

    #D_tilde = list(zip(x, y_tilde))
    return x, y_tilde
    

if __name__ == "__main__":
    # Q_kernel = {0: [[0, 0, 1], [1/2, 1/2, 0], [1/2, 0, 1/2]],
    #      1/6: [[0, 1/6, 5/6], [1/2, 1/2, 0], [1/2, 0, 1/2]],
    #      1/3: [[0, 1/3, 2/3], [1/2, 1/2, 0], [1/2, 0, 1/2]],
    #      1/2: [[1/4, 1/4, 1/2], [1/2, 1/2, 0], [1/2, 0, 1/2]],
    #      2/3: [[1/3, 1/3, 1/3], [1/2, 1/2, 0], [1/3, 1/3, 1/3]],
    #      1: [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]}

    T_kernel = Q_kernel = {0: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        1/6: [[1/18, 8/9, 1/18], [1/18, 1/18, 8/9], [8/9, 1/18, 1/18]],
        1/3: [[1/9, 7/9, 1/9], [1/9, 1/9, 7/9], [7/9, 1/9, 1/9]],
        1/2: [[1/6, 2/3, 1/6], [1/6, 1/6, 2/3], [2/3, 1/6, 1/6]],
        2/3: [[2/9, 5/9, 2/9], [2/9, 2/9, 5/9], [5/9, 2/9, 2/9]],
        1: [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]}

    num_trials = 1

    losses = {}  # (delta_t, delta_q) -> avg loss

    total = len(T_kernel) * len(Q_kernel)
    with tqdm(total = total, desc="sweep") as pbar:
        for delta_t, t_matrix in T_kernel.items():
            transition_kernel = np.array(t_matrix)
            losses[delta_t] = []

            for delta_q, q_matrix in Q_kernel.items():
                observation_kernel = np.array(q_matrix)

                loss_sum = 0
                for trial in range(num_trials):
                    actions, mu, y = train(generate_pairs, transition_kernel, observation_kernel)
                    train_loss = C_T(mu[T], y, LOSS, actions[-1])
                    #train_name = f"train-T{delta_t:.4g}-Q{delta_q:.4g}-trial{trial}"
                    #plot_state_predictions(mu, y, RESULTS_PATH, train_name)
                    #plot_promptwise_loss(mu, y, train_loss, RESULTS_PATH, train_name)

                    mu_test, y_test = test(actions, generate_pairs, transition_kernel, observation_kernel)
                    test_loss = C_T(mu_test[T], y_test, LOSS, actions[-1])
                    #test_name = f"test-T{delta_t:.4g}-Q{delta_q:.4g}-trial{trial}"
                    #plot_state_predictions(mu_test, y_test, RESULTS_PATH, test_name)
                    #plot_promptwise_loss(mu_test, y_test, loss, RESULTS_PATH, test_name)

                    loss_sum += test_loss

                losses[delta_t].append(loss_sum / num_trials)
                pbar.update(1)

    print(losses)

    delta_T_values = list(T_kernel.keys())
    delta_Q_values = list(Q_kernel.keys())
    grid = np.array([losses[c] for c in T_kernel]).T 

    plot_loss_grid(delta_T_values, delta_Q_values, grid, RESULTS_PATH)
