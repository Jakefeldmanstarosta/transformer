import os
import numpy as np
import matplotlib.pyplot as plt

from common import *

# finite state/measurement models (deterministic or probabilistic transition
# kernels over the quantized state space). No continuous state -> no
# quantization error, so xi's effect on Markovianness shows up cleanly.

CHAIN = "probabilistic"

RESULTS_PATH = r"C:\Users\jakef\Projects\next-token-prediction-control\results\markovianness"
os.makedirs(RESULTS_PATH, exist_ok=True)


def create_transition(n, N, chain = CHAIN):
    if chain == 'deterministic':
        transition_dict = np.zeros(n ** N, dtype=int)
        for i in range(n ** N):
            transition_dict[i] = np.random.randint(0, n)
        return transition_dict

    if chain == 'probabilistic':
        transition_matrix = np.random.rand(N * n, n)

        clustering_constant = 1000
        transition_matrix = transition_matrix ** clustering_constant

        for i in range(N * n):
            transition_matrix[i] /= np.sum(transition_matrix[i])
        return transition_matrix


def generate_process(n, K_tilde, N, xi, transition, chain = CHAIN):
    #generates a process of indices (for the states)
    X = []
    V = []
    Y = []
    n_samples = K_tilde + N - 1

    if chain == 'deterministic':

        transition_dict = transition

        for r in range(N):
            X.append(0)
            V.append(int(np.random.normal(0, scale = (1-xi)*n)))
            Y.append(int(reflect(X[r]+V[r], 0, n-1)))


        for r in range(N, n_samples):
            idx = int(n * X[r-2] + X[r-1])
            X.append(transition_dict[idx])
            V.append(int(np.random.normal(0, scale = (1 - xi)*n)))
            Y.append(int(reflect(X[r]+V[r], 0, n-1)))

        return Y

    if chain == 'probabilistic':

        transition_matrix = transition

        for r in range(N):
            X.append(one_hot(0, n))
            V.append(np.ones(n)/n)

        for r in range(N, n_samples):
            X.append(np.concatenate(X[-N:])/N @ transition_matrix) #Markovian
            V.append(np.ones(n)/n) #uniform non-Markovian

        for r in range(n_samples):
            dist = xi * X[r] + (1.0 - xi) * V[r]
            dist /= np.sum(dist)
            Y.append(np.random.choice(n, p= dist))

        return Y


if __name__ == "__main__":

    xi_values = [0.01, 0.25, 0.5, 0.75, 1.0]  # 0 = non-Markovian, 1 = Markovian

    losses = []
    num_trials = 3
    transition = create_transition(n, N, chain = CHAIN)

    for xi in xi_values:
        loss_sum = 0
        for trial in range(num_trials):
            actions, mu, y = train(generate_process, CHAIN, xi, transition)
            train_name = "train" + str(xi) + "-trial" + str(trial)
            train_loss = C_T(mu[T], y, LOSS, actions[-1])
            plot_state_predictions(mu, y, RESULTS_PATH, train_name)
            plot_promptwise_loss(mu, y, train_loss, RESULTS_PATH, train_name)

            mu_test, y_test = test(actions, generate_process, CHAIN, xi, transition)
            test_name = "test" + str(xi) + "-trial" + str(trial)
            loss = C_T(mu_test[T], y_test, LOSS, actions[-1])
            plot_state_predictions(mu_test, y_test, RESULTS_PATH, test_name)
            plot_promptwise_loss(mu_test, y_test, loss, RESULTS_PATH, test_name)

            loss_sum += loss

        losses.append(loss_sum / num_trials)

    print(losses, xi_values)
    plt.plot(xi_values, losses)
    plt.xlabel("xi value")
    plt.ylabel("Loss")
    plt.show()
