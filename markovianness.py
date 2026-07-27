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


# TRANSFORMER TRAINING

def train(xi, transition):
    #construct dataset
    X = generate_process(n, K_tilde, N, xi, transition, chain = CHAIN)
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
def test(U_t, xi, transition):
    X_test = generate_process(n, K_tilde, N, xi, transition, chain = CHAIN)
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

    xi_values = [0.01, 0.25, 0.5, 0.75, 1.0]  # 0 = non-Markovian, 1 = Markovian

    losses = []
    num_trials = 3
    transition = create_transition(n, N, chain = CHAIN)

    for xi in xi_values:
        loss_sum = 0
        for trial in range(num_trials):
            actions, mu, y = train(xi, transition)
            visualize(mu, y, RESULTS_PATH, name = "train" + str(xi) + "-trial" + str(trial))
            mu_test, y_test = test(actions, xi, transition)
            loss = visualize(mu_test, y_test, RESULTS_PATH, "test" + str(xi) + "-trial" + str(trial))

            loss_sum += loss

        losses.append(loss_sum / num_trials)

    print(losses, xi_values)
    plt.plot(xi_values, losses)
    plt.xlabel("xi value")
    plt.ylabel("Loss")
    plt.show()
