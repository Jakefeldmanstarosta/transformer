"""Quantized-MDP formulation of next-token prediction as a control problem.

Data is lifted to probability measures on a quantized token-embedding space
S_n, evolved through a quantized dynamics operator Phi, and the optimal
policy gamma is found by backward induction over the ensembles reachable
from the initial data, mirroring the algorithm steps described in the
accompanying paper (main.tex).

Variable names follow the paper's notation: S_n (quantized state space),
U_m (quantized action space), mu (lifted probability measures), phi
(quantized dynamics), R_l / Q_n (probability / state quantizers), W2
(Wasserstein-2 cost), gamma (policy), C (value function).
"""

import numpy as np
from scipy.optimize import linprog


# =============================================================================
# Quantization primitives
# =============================================================================

def quantize_state(s, S_n):
    """Q_n: snap a continuous state s to the nearest point in the grid S_n."""
    diffs = np.abs(s - S_n)
    return S_n[np.argmin(diffs)]


# TODO: implement Reznik's discrete probability distribution quantization algorithm
def quantize_probability(p, l):
    """R_l (entrywise): snap a probability mass p to the nearest multiple of 1/(l-1)."""
    return round(p * (l - 1)) / (l - 1)


def dirac(s):
    """Dirac/one-hot measure on S_n placing all mass on state s."""
    mu_i = np.zeros(n)
    idx = np.where(S_n == s)[0][0]
    mu_i[idx] = 1
    return mu_i


def measure_to_state(mu):
    """Recover the state value (not index) from a measure mu (works for mu_i/mu_k/mu_t shapes)."""
    return S_n[np.argmax(mu, axis=-1)]


def W2(p, q):
    """Squared Wasserstein-2 distance between two measures p, q on S_n, via linprog."""
    c = cost_matrix.flatten()

    p_len = len(p)
    q_len = len(q)

    if c.shape[0] != p_len * q_len:
        print("Ensure that the shape of the inputs match the cost matrix/S_n shape")

    A_eq = np.zeros((p_len + q_len, p_len * q_len))

    # isolate matrix rows
    for i in range(p_len):
        A_eq[i, i * q_len:(i + 1) * q_len] = 1.0

    # isolate matrix cols
    for j in range(q_len):
        A_eq[p_len + j, j::q_len] = 1.0

    # rows must equal p, cols must equal q
    b_eq = np.concatenate((p, q))

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    return result.fun


# =============================================================================
# Problem data: generate an ergodic process and split into data-label pairs
# =============================================================================

# S is the interval [0, 5]
floor = 0
ceil = 5


def P(X_curr, std=0.5):
    """Transition kernel: normal distribution centered one step ahead of X_curr."""
    bell = np.random.normal(loc=X_curr + 1, scale=std)
    return np.clip(bell, floor, ceil)


M = 8

X = [0]
for r in range(M - 1):
    X.append(P(X[r]))

# split up the process into data-label pairs
N = 2

x = []
y_tilde = []

# start at index N
for r in range(M - N):
    x.append(X[r:r + N])
    y_tilde.append(X[r + N])

K = len(x)

#1
D_tilde = list(zip(x, y_tilde))

T = 2   # time horizon = number of layers
l = 9   # probability measure quantization parameter (l levels = {0, 1/(l-1), ..., 1})
n = 10  # state space quantization parameter (number of states = number of bins + 1)
m = 5   # action space quantization parameter (unused directly; len(U_m) gives m)


# =============================================================================
# Quantized state, probability and action spaces
# =============================================================================

#2
S_n = []
for i in range(n):
    S_n.append(i / 2)
S_n = np.array(S_n)


def Q_n(x):
    return quantize_state(x, S_n)


cost_matrix = np.array([[np.linalg.norm(s1 - s2) ** 2 for s1 in S_n] for s2 in S_n])  # used by W2

#3
U_m = [-2, -1, 0, 1, 2]
# for our example, we add the weights

#4
# assume l > 0
P_l = {i / (l - 1) for i in range(0, l)}


def R_l(mu_k):
    """Quantize every entry of every measure mu_i in the ensemble mu_k to P_l."""
    return [[quantize_probability(x, l) for x in mu_i] for mu_i in mu_k]
# we have to choose a suitable l given our number of states n, or else probabilities
# could technically disappear; taking l > n is believed to suffice (TODO: verify)


# =============================================================================
# Build initial lifted measures from the data (steps 5-9)
# =============================================================================

#5-9
mu_0 = []
y = []
k_unique = 0

# create empirical distributions, remove duplicates, update K to reflect the number of unique labels
for k in range(K):

    candidate_x = np.array([dirac(Q_n(x[k][i])) for i in range(N)])
    candidate_y = dirac(Q_n(y_tilde[k]))

    if k > 0:
        matches = np.all(np.all(np.array(mu_0) == candidate_x, axis=2), axis=1)
    else:
        matches = []

    if np.any(matches):
        idx = np.where(matches == True)[0][0]  # we choose the first occurrence
        y[idx] += candidate_y

    else:
        y.append(candidate_y)
        mu_0.append(candidate_x)
        k_unique += 1

K = k_unique

for k in range(K):
    denom = np.sum(y[k])
    y[k] /= denom

# initialize lifted probability measures
mu_0 = np.array(mu_0)
mu = np.zeros((T + 1, *mu_0.shape))
mu[0] = mu_0
# mu has shape (T + 1, K, N, len(S_n))


# =============================================================================
# Quantized dynamics (step 10)
# =============================================================================

#10
def f(s, mu_i, u, S=S_n):  # arbitrary example function right now; TODO implement attention
    # dynamics here are done at the state level; can equally be done at the individual measure level
    if (s + u) in S:
        return s + u
    return s


def phi_n(u, mu_k):
    """Apply action u to every token measure in a single ensemble member mu_k."""
    result = []
    for x in mu_k:
        s = measure_to_state(x)
        s = Q_n(f(s, x, u))
        result.append(dirac(s))
    return np.array(result)


def phi(u, mu_t):
    """Apply action u to an entire ensemble of measures mu_t (the quantized dynamics Phi)."""
    result = []
    for mu_k in mu_t:
        result.append(R_l(phi_n(u, mu_k)))
    return np.array(result)


# =============================================================================
# Terminal cost and reachable-ensemble enumeration (step 11)
# =============================================================================

#11
def C_T(mu_T, y=y):
    """Average squared-W2 cost between each ensemble member's final token measure and its label."""
    total = 0
    for k in range(K):
        total += W2(mu_T[k][N - 1], y[k])
    return total / K


def ensemble_to_index(mu_t):
    """Map an ensemble of measures to its integer index in the flattened state space."""
    state = np.argmax(mu_t, axis=-1).flatten()  # look at the S_n axis
    return sum(state[i] * (n ** i) for i in range(N * K))


def create_reachable_ensembles(target_depth=None):
    """Breadth-first search (capped at depth T) over the ensembles reachable from mu[0]
    under the dynamics phi and actions U_m, instead of brute-force enumerating all of
    P^l(X_n)^K (which explodes as n ^ (K * N))."""
    start_ens = mu[0]
    visited = {0: {ensemble_to_index(start_ens)}}
    queue = [(start_ens, 0)]

    head = 0
    while head < len(queue):
        curr_ens, depth = queue[head]
        head += 1

        if target_depth is None or depth == target_depth:
            yield curr_ens

        if depth < T:
            visited.setdefault(depth + 1, set())
            for u in U_m:
                next_ens = phi(u, curr_ens)
                next_idx = ensemble_to_index(next_ens)
                if next_idx not in visited[depth + 1]:
                    visited[depth + 1].add(next_idx)
                    queue.append((next_ens, depth + 1))


# =============================================================================
# Backward induction: value function and optimal policy (steps 12-17)
# =============================================================================

def backward_induction(T, U_m):
    """Compute the value function C and optimal policy gamma via backward induction,
    restricted to ensembles reachable from mu[0].

    C[t][i]     = optimal cost-to-go from reachable ensemble index i at time t
    gamma[t][i] = optimal action from reachable ensemble index i at time t
    """
    C = {t: {} for t in range(T + 1)}
    gamma = {t: {} for t in range(T)}

    for mu_T in create_reachable_ensembles(target_depth=T):
        C[T][ensemble_to_index(mu_T)] = C_T(mu_T)

    for t in range(T - 1, -1, -1):  # goes from T-1 down to 0
        for mu_t in create_reachable_ensembles(target_depth=t):
            i = ensemble_to_index(mu_t)
            costs = [C[t + 1][ensemble_to_index(phi(u, mu_t))] for u in U_m]  # indexed by action
            gamma[t][i] = U_m[np.argmin(costs)]
            C[t][i] = np.min(costs)

    return C, gamma


# =============================================================================
# Forward pass: roll out the optimal policy from the data (steps 18-25)
# =============================================================================

def forward_pass(T, mu, gamma):
    """Roll mu forward under gamma, recording the optimal action taken at each time step."""
    U_t = []
    for t in range(T):
        optimal_u = gamma[t][ensemble_to_index(mu[t])]
        mu[t + 1] = phi(optimal_u, mu[t])
        U_t.append(optimal_u)
    return mu, U_t


def main():
    #11-17
    C, gamma = backward_induction(T, U_m)

    #18-25
    mu_final, U_t = forward_pass(T, mu, gamma)
    print(U_t)
    return U_t


if __name__ == "__main__":
    main()
