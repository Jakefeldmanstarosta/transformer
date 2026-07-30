import os
import numpy as np
import matplotlib.pyplot as plt

from common import *

# discrete state process X_{t+1} ~ T_kernel[X_t] and observation Y_t ~ Q_kernel[X_t].
# Both kernels are keyed by their own Dobrushin coefficient, so filter stability
# can be swept independently over the process's mixing rate (T_kernel) and the
# observation channel's genuineness (Q_kernel).

CHAIN = "dobrushin"

RESULTS_PATH = r"C:\Users\jakef\Projects\next-token-prediction-control\results\stability"
os.makedirs(RESULTS_PATH, exist_ok=True)

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

def plot_loss_grid(delta_T_values, delta_Q_values, grid):
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
    plt.show()


def generate_process(n, K_tilde, N, transition_kernel, observation_kernel, chain = CHAIN):
    #generates a process of indices (for the states)
    X = []
    Y = []
    n_samples = K_tilde + N - 1

    X.append(0)
    for r in range(1, n_samples):
        X.append(int(np.random.choice(n, p = transition_kernel[X[-1]])))

    for r in range(n_samples):
        Y.append(int(np.random.choice(n, p = observation_kernel[X[r]])))

    return Y


if __name__ == "__main__":

    num_trials = 1

    losses = {}  # (t_coeff, q_coeff) -> avg loss

    for t_coeff, t_matrix in T_kernel.items():
        transition_kernel = np.array(t_matrix)
        losses[t_coeff] = []

        for q_coeff, q_matrix in Q_kernel.items():
            observation_kernel = np.array(q_matrix)

            loss_sum = 0
            for trial in range(num_trials):
                actions, mu, y = train(generate_process, CHAIN, transition_kernel, observation_kernel)
                train_name = f"train-T{t_coeff:.4g}-Q{q_coeff:.4g}-trial{trial}"
                train_loss = C_T(mu[T], y, LOSS, actions[-1])
                plot_state_predictions(mu, y, RESULTS_PATH, train_name)
                #plot_promptwise_loss(mu, y, train_loss, RESULTS_PATH, train_name)

                mu_test, y_test = test(actions, generate_process, CHAIN, transition_kernel, observation_kernel)
                test_name = f"test-T{t_coeff:.4g}-Q{q_coeff:.4g}-trial{trial}"
                loss = C_T(mu_test[T], y_test, LOSS, actions[-1])
                plot_state_predictions(mu_test, y_test, RESULTS_PATH, test_name)
                #plot_promptwise_loss(mu_test, y_test, loss, RESULTS_PATH, test_name)

                loss_sum += loss

            losses[t_coeff].append(loss_sum / num_trials)

    print(losses)

    delta_T_values = list(T_kernel.keys())
    delta_Q_values = list(Q_kernel.keys())
    grid = np.array([losses[c] for c in T_kernel]).T  # (num_Q, num_T), matching plot_loss_grid's (ny, nx)

    plot_loss_grid(delta_T_values, delta_Q_values, grid)