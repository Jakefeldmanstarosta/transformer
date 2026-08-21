import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import *


RESULTS_PATH = r"C:\Users\jakef\Projects\next-token-prediction-control\results\stability"
os.makedirs(RESULTS_PATH, exist_ok=True)

def create_kernel(delta, n, offset):

    result = []

    first_row = [round(delta / n, 2) for i in range(n)]
    first_row[offset] += 1.0 - sum(first_row)

    result.append(first_row)

    for i in range(n - 1):
        shifted = [result[-1][-1]] + result[-1][:-1]
        result.append(shifted)

    return result


def plot_loss_grid(delta_T_values, delta_Q_values, grid, PATH, filename = "colour-gird.png", cbar_label = "test loss"):
    nx = len(delta_T_values)
    ny = len(delta_Q_values)

    plt.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), grid)
    plt.colorbar(label = cbar_label)
    for i in range(ny):
        for j in range(nx):
            plt.text(j + 0.5, i + 0.5, f"{grid[i, j]:.2f}", ha='center', va='center')

    plt.xticks(np.arange(nx) + 0.5, [f"{v:.2f}" for v in delta_T_values])
    plt.yticks(np.arange(ny) + 0.5, [f"{v:.2f}" for v in delta_Q_values])
    plt.xlabel("delta(T)")
    plt.ylabel("delta(Q)")
    plt.savefig(os.path.join(PATH, filename), bbox_inches='tight')
    plt.close()


def plot_loss_line(x_values, losses, PATH, xlabel, filename, ylabel = "test loss"):
    plt.plot(x_values, losses, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(PATH, filename), bbox_inches='tight')
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
    

def TQ_sweep(T_kernel, Q_kernel, num_trials = 3, results_path = RESULTS_PATH):
    # holds the action space quantizer m and state space quantizer n fixed and sweeps delta(T), delta(Q)
    losses = {}  # delta_t -> [avg loss per delta_q]
    runtimes = {}  # delta_t -> [avg runtime per delta_q]
    weights = []

    total = len(T_kernel) * len(Q_kernel)
    with tqdm(total = total, desc="T/Q sweep") as pbar:
        for delta_t, t_matrix in T_kernel.items():
            transition_kernel = np.array(t_matrix)
            losses[delta_t] = []
            runtimes[delta_t] = []

            for delta_q, q_matrix in Q_kernel.items():
                observation_kernel = np.array(q_matrix)

                loss_sum = 0
                runtime_sum = 0
                for trial in range(num_trials):
                    start_time = time.perf_counter()

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

                    runtime = time.perf_counter() - start_time

                    loss_sum += test_loss
                    runtime_sum += runtime

                    weights.append({
                        "delta_t": delta_t,
                        "delta_q": delta_q,
                        "trial": trial,
                        "train_loss": float(train_loss),
                        "test_loss": float(test_loss),
                        "runtime": runtime,
                        "weights": [action_to_dict(u) for u in actions]
                    })

                losses[delta_t].append(loss_sum / num_trials)
                runtimes[delta_t].append(runtime_sum / num_trials)
                pbar.update(1)

    print(losses)
    print(runtimes)

    delta_T_values = list(T_kernel.keys())
    delta_Q_values = list(Q_kernel.keys())
    grid = np.array([losses[c] for c in T_kernel]).T
    runtime_grid = np.array([runtimes[c] for c in T_kernel]).T

    plot_loss_grid(delta_T_values, delta_Q_values, grid, results_path)
    plot_loss_grid(delta_T_values, delta_Q_values, runtime_grid, results_path,
                    filename = "runtime-grid.png", cbar_label = "runtime (s)")

    with open(os.path.join(results_path, "tq-weights.json"), "w") as file_handle:
        json.dump(weights, file_handle, indent = 2)

    return losses


def m_sweep(m_values, transition_kernel, observation_kernel, num_trials = 3, results_path = RESULTS_PATH):
    # holds T, Q, n fixed and sweeps the number of actions (m)
    transition_kernel = np.array(transition_kernel)
    observation_kernel = np.array(observation_kernel)

    losses = []
    runtimes = []
    weights = []
    with tqdm(total = len(m_values), desc="m-actions sweep") as pbar:
        for m_val in m_values:
            set_action_quantization(m_val)

            loss_sum = 0
            runtime_sum = 0
            for trial in range(num_trials):
                start_time = time.perf_counter()

                actions, mu, y = train(generate_pairs, transition_kernel, observation_kernel)
                train_loss = C_T(mu[T], y, LOSS, actions[-1])

                mu_test, y_test = test(actions, generate_pairs, transition_kernel, observation_kernel)
                test_loss = C_T(mu_test[T], y_test, LOSS, actions[-1])

                runtime = time.perf_counter() - start_time

                loss_sum += test_loss
                runtime_sum += runtime

                weights.append({
                    "m": m_val,
                    "trial": trial,
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss),
                    "runtime": runtime,
                    "weights": [action_to_dict(u) for u in actions]
                })

            losses.append(loss_sum / num_trials)
            runtimes.append(runtime_sum / num_trials)
            pbar.update(1)

    print(dict(zip(m_values, losses)))
    print(dict(zip(m_values, runtimes)))

    plot_loss_line(m_values, losses, results_path, "m (actions quantization)", "actions-line.png")
    plot_loss_line(m_values, runtimes, results_path, "m (actions quantization)", "actions-runtime.png",
                    ylabel = "runtime (s)")

    with open(os.path.join(results_path, "m-weights.json"), "w") as file_handle:
        json.dump(weights, file_handle, indent = 2)

    return losses


def n_sweep(n_values, delta_t, delta_q, num_trials = 3, results_path = RESULTS_PATH):
    # holds delta(T), delta(Q), and m fixed and sweeps the state quantization (n)
    losses = []
    runtimes = []
    weights = []
    with tqdm(total = len(n_values), desc="n-state sweep") as pbar:
        for n_val in n_values:
            set_state_quantization(n_val)
            transition_kernel = np.array(create_kernel(delta_t, n_val, 1))
            observation_kernel = np.array(create_kernel(delta_q, n_val, -1))

            loss_sum = 0
            runtime_sum = 0
            for trial in range(num_trials):
                start_time = time.perf_counter()

                actions, mu, y = train(generate_pairs, transition_kernel, observation_kernel)
                train_loss = C_T(mu[T], y, LOSS, actions[-1])

                mu_test, y_test = test(actions, generate_pairs, transition_kernel, observation_kernel)
                test_loss = C_T(mu_test[T], y_test, LOSS, actions[-1])

                runtime = time.perf_counter() - start_time

                loss_sum += test_loss
                runtime_sum += runtime

                weights.append({
                    "n": n_val,
                    "trial": trial,
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss),
                    "runtime": runtime,
                    "weights": [action_to_dict(u) for u in actions]
                })

            losses.append(loss_sum / num_trials)
            runtimes.append(runtime_sum / num_trials)
            pbar.update(1)

    print(dict(zip(n_values, losses)))
    print(dict(zip(n_values, runtimes)))

    plot_loss_line(n_values, losses, results_path, "n (state quantization)", "states-line.png")
    plot_loss_line(n_values, runtimes, results_path, "n (state quantization)", "states-runtime.png",
                    ylabel = "runtime (s)")

    with open(os.path.join(results_path, "n-weights.json"), "w") as file_handle:
        json.dump(weights, file_handle, indent = 2)

    return losses


if __name__ == "__main__":

    num_trials = 1

    delta_values = [0, 1/6, 1/3, 1/2, 2/3, 1]
    T_kernel = {delta: create_kernel(delta, n, 1) for delta in delta_values}
    Q_kernel = {delta: create_kernel(delta, n, -1) for delta in delta_values}

    # holds number of actions (m) fixed and sweeps T, Q for varying dobrushin coeffecients
    #TQ_sweep(T_kernel, Q_kernel, num_trials = num_trials)

    # holds T, Q fixed at a single kernel each and sweeps the number of actions (m)
    # m_values = [1, 2, 3, 4, 5]
    # m_sweep(m_values, T_kernel[1/3], Q_kernel[1/3], num_trials = num_trials)

    # holds delta(T), delta(Q), and m fixed and sweeps the state quantization (n)
    n_values = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    n_sweep(n_values, 1/3, 1/3, num_trials = num_trials)
