import os
from math import erf, sqrt
import numpy as np
import matplotlib.pyplot as plt

from common import *

# continuous state process X_{t+1} = f(X_t) + W_t (dobrushin coefficient
# controlled by xi = sigma_t/t). Tests filter stability under the process's
# mixing rate, subject to quantization error from projecting onto S_n.

CHAIN = "dobrushin"

RESULTS_PATH = r"C:\Users\jakef\Projects\next-token-prediction-control\results\stability"
os.makedirs(RESULTS_PATH, exist_ok=True)


K = {0: [[0, 0, 1], [1/2, 1/2, 0], [1/3, 1/3, 1/3]],
     1/3: [[0, 1/3, 2/3], [1/2, 1/2, 0], [1/3, 1/3, 1/3]],
     1/2: [[1/4, 1/4, 1/2], [1/2, 1/2, 0], [1/3, 1/3, 1/3]],
     2/3: [[1/3, 1/3, 1/3], [1/2, 1/2, 0], [1/3, 1/3, 1/3]],
     1: [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]}


def create_transition(n, N, chain = CHAIN):
    return None


def delta_T(xi):
    # delta(T) = 2 * Phi(-t/sigma_t) = 1-erf(1/(xi*sqrt(2))), since sigma_t = xi * t
    if xi == 0:
        return 0.0
    return 1 - erf(1 / (xi * sqrt(2)))


def plot_loss_grid(delta_T_values, delta_K_values, grid):
    nx = len(delta_T_values)
    ny = len(delta_K_values)

    plt.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), grid)
    for i in range(ny):
        for j in range(nx):
            plt.text(j + 0.5, i + 0.5, f"{grid[i, j]:.2f}", ha='center', va='center')

    plt.xticks(np.arange(nx) + 0.5, [f"{v:.2f}" for v in delta_T_values])
    plt.yticks(np.arange(ny) + 0.5, [f"{v:.2f}" for v in delta_K_values])
    plt.xlabel("delta(T)")
    plt.ylabel("delta(K)")
    plt.show()


def generate_process(n, K_tilde, N, xi, transition, kernel, chain = CHAIN):
    #generates a process of indices (for the states)
    X = []
    Y = []
    n_samples = K_tilde + N - 1

    t = S[1]  # bound of the state space, f(X) clips into [-t, t]
    sigma_t = xi * t

    X.append(0.0)
    for r in range(1, n_samples):
        # f(X_{t-1}) = clip(X_{t-1}, -t, t)
        X.append(min(max(X[-1], -t), t) + np.random.normal(0, sigma_t))

    for r in range(n_samples):
        row = np.where(S_n == Q_n(X[r], S_n))[0][0]
        Y.append(int(np.random.choice(n, p = kernel[row])))

    return Y


# TRANSFORMER TRAINING

def train(xi, transition, kernel):
    #construct dataset
    X = generate_process(n, K_tilde, N, xi, transition, kernel, chain = CHAIN)
    x, y_tilde = create_pairs(X, S_n, N, K_tilde)

    #lift and dedup the dataset
    mu_0, y = construct_empirical_distribution(x, y_tilde, N, S_n)
    mu = np.zeros((T + 1, *mu_0.shape))
    mu[0] = mu_0

    #define terminal cost
    C = {}
    for t in range(T+1):
        C[t] = {}
    for i, mu_T in enumerate(create_reachable_ensembles(mu[0], target_depth=T)):
        C[T][ensemble_to_index(mu_T)] = C_T(mu_T, y, LOSS)

    gamma = {}
    for t in range(T):
        gamma[t] = {}
    actions = list(create_actions(U_m))

    #solve dp
    for t in range(T-1, -1, -1): #goes from T-1 to 0
        for mu_t in create_reachable_ensembles(mu[0], target_depth=t):

            i = ensemble_to_index(mu_t)
            costs = [C[t+1][ensemble_to_index(phi(u, mu_t))] for u in actions] #costs indexed by each action
            best_idx = np.argmin(costs)
            gamma[t][i] = actions[best_idx]
            C[t][i] = np.min(costs)

    #forward pass
    U_t = []
    for t in range(T):
        optimal_u = gamma[t][ensemble_to_index(mu[t])]
        mu[t + 1] = phi(optimal_u, mu[t])
        U_t.append(optimal_u)

    return U_t, mu, y


# TESTING
def test(U_t, xi, transition, kernel):
    X_test = generate_process(n, K_tilde, N, xi, transition, kernel, chain = CHAIN)
    x_test, y_tilde_test = create_pairs(X_test, S_n, N, K_tilde)
    mu_0_test, y_test = construct_empirical_distribution(x_test, y_tilde_test, N, S_n)

    mu_0_test = np.array(mu_0_test)
    mu_test = np.zeros((T + 1, *mu_0_test.shape))
    mu_test[0] = mu_0_test

    #Forward pass
    for t in range(T):
        mu_test[t + 1] = phi(U_t[t], mu_test[t])

    return mu_test, y_test


if __name__ == "__main__":

    xi_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    num_trials = 3
    transition = create_transition(n, N, chain = CHAIN)

    losses = {}  # coefficient: losses per xi

    for coeff, matrix in K.items():
        kernel = np.array(matrix)
        losses[coeff] = []

        for xi in xi_values:
            loss_sum = 0
            for trial in range(num_trials):
                actions, mu, y = train(xi, transition, kernel)
                train_name = f"train-K{coeff:.4g}-xi{xi}-trial{trial}"
                train_loss = C_T(mu[T], y, LOSS)
                #plot_state_predictions(mu, y, RESULTS_PATH, train_name)
                #plot_promptwise_loss(mu, y, train_loss, RESULTS_PATH, train_name)

                mu_test, y_test = test(actions, xi, transition, kernel)
                test_name = f"test-K{coeff:.4g}-xi{xi}-trial{trial}"
                loss = C_T(mu_test[T], y_test, LOSS)
                #plot_state_predictions(mu_test, y_test, RESULTS_PATH, test_name)
                #plot_promptwise_loss(mu_test, y_test, loss, RESULTS_PATH, test_name)

                loss_sum += loss

            losses[coeff].append(loss_sum / num_trials)

    print(losses, xi_values)

    delta_T_values = [delta_T(xi) for xi in xi_values]
    delta_K_values = list(K.keys())
    grid = np.array([losses[c] for c in K])

    plot_loss_grid(delta_T_values, delta_K_values, grid)