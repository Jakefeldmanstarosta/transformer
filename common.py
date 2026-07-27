import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os


# QUANTIZERS

#state
def Q_n(s, S_n):
    diffs = np.abs(s - S_n)
    return S_n[np.argmin(diffs)]

def quantize_probability(p, l):
    return round(p * (l - 1))/ (l-1)

def R_l(mu_k):
    quantized = [[quantize_probability(x, l) for x in mu_i] for mu_i in mu_k]
    return quantized


# LOSSES

def W2(p, q):
    # note the cost is defined for P(S_n) x P(S_n)
    # p,q have shape mu_i

    c = cost_matrix.flatten()

    p_len = len(p)
    q_len = len(q)

    if(c.shape[0] != p_len * q_len):
        print("Ensure that the shape of the inputs match the cost matrix/S_n shape")

    A_eq = np.zeros((p_len + q_len, p_len * q_len))

    #we isolate matrix rows
    for i in range(p_len):
        A_eq[i, i * q_len : (i+1) * q_len] = 1.0

    #we isolate matrix cols
    for j in range(q_len):
        A_eq[p_len + j, j::q_len] = 1.0

    #rows must equal p, cols must equal q
    b_eq = np.concatenate((p, q))

    result = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = (0, None), method ='highs')
    return result.fun

def cross_entropy(p, q):
    # note the cost is defined for P(S_n) x P(S_n)
    # p,q have shape mu_i
    q = np.clip(q, 1e-10, 1)
    return -np.sum(p * np.log(q))


# ACTIONS

class Action:
    def __init__(self, W, A, b, Q, K, V, m = None):
        if m is not None:
            W = np.unique(np.linspace(*W, m))
            A = np.unique(np.linspace(*A, m))
            b = np.unique(np.linspace(*b, m))
            Q = np.unique(np.linspace(*Q, m))
            K = np.unique(np.linspace(*K, m))
            V = np.unique(np.linspace(*V, m))
        self.W = W
        self.A = A
        self.b = b
        self.Q = Q
        self.K = K
        self.V = V

    def __repr__(self):
        return f"W={self.W}, A={self.A}, b={self.b}, Q={self.Q}, K={self.K}, V={self.V}"

def create_actions(U):
    for w_i in U.W:
        for a_i in U.A:
            for b_i in U.b:
                for q_i in U.Q:
                    for k_i in U.K:
                        for v_i in U.V:
                            yield Action(w_i, a_i, b_i, q_i, k_i, v_i)


# TRANSITIONS

def f(s, mu_k, u):

    ff = u.W * relu(u.A * s + u.b)

    scores = np.array([s * u.Q * top_measure_to_state(mu_i) * u.K for mu_i in mu_k])
    weights = softmax(beta * scores)
    attn = np.sum(weights * u.V * np.array([top_measure_to_state(mu_i) for mu_i in mu_k]))

    return np.clip(attn + ff, *S)

def phi_n(u, mu_k):
    result = []
    for mu_i in mu_k:
        s = top_measure_to_state(mu_i)
        s = Q_n(f(s, mu_k, u), S_n)
        result.append(dirac(s))
    return np.array(result)

def phi(u, mu_t):
    result = []
    for mu_k in mu_t:
        result.append(R_l(phi_n(u, mu_k)))
    return np.array(result)

# HELPER FUNCTIONS

def dirac(s):
    mu_i = np.zeros(n)
    idx = np.where(S_n == s)[0][0]
    mu_i[idx] = 1
    return mu_i

def top_measure_to_state(mu):  #works for any shape of mu (mu_i/mu_k/mu_t)
    return S_n[np.argmax(mu, axis = -1)]

def measure_to_state(mu_t): #works for shape mu_t
    states = []
    probs = []
    for mu_k in mu_t:
        order = np.argsort(mu_k)[::-1]
        nonzero = mu_k[order] > 0
        idxs = order[nonzero]
        states.append(S_n[idxs])
        probs.append(mu_k[idxs])
    return states, probs

def one_hot(x, n):
    v = np.zeros(n)
    v[x] = 1
    return v

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x)
    return e / e.sum()

def reflect(x, lo, hi):
    T = 2 * (hi - lo)
    y = (x - lo) % T
    return lo + min(T - y, y)

def C_T(mu_T, y, loss):
    K_local = len(y)
    total = 0
    for k in range(K_local):
        if loss == 'CE' or loss == 'Cross Entropy':
            total += cross_entropy(mu_T[k][N-1], y[k])
        if loss == 'W2' or loss == 'Wasserstein':
            total += W2(mu_T[k][N-1], y[k])
    return total / K_local

def create_reachable_ensembles(mu_0, target_depth = None):
    #breadth first search with capped depth of T
    count = 0
    visited = {}
    queue = []
    #keep track of ensemble, depth

    start_ens = mu_0
    start_idx = ensemble_to_index(start_ens)

    visited[0] = {start_idx}
    queue.append((start_ens, 0))

    head = 0

    while head < len(queue):
        curr_ens, depth = queue[head]
        head += 1

        if target_depth is None or depth == target_depth:
            yield curr_ens

        if depth < T:
            if depth + 1 not in visited:
                visited[depth + 1] = set()
            for u in create_actions(U_m):
                next_ens = phi(u, curr_ens)
                next_idx = ensemble_to_index(next_ens)
                count += 1
                if next_idx not in visited[depth + 1]:
                    visited[depth + 1].add(next_idx)
                    queue.append((next_ens, depth + 1))

def ensemble_to_index(mu_t):
    state = np.argmax(mu_t, axis = -1).flatten() #look at the S_n axis
    index = sum(int(state[i]) * (n ** i) for i in range(N * len(mu_t)))
    return index


# DATASET CONSTRUCTION

def create_pairs(X, S_n, N, K_tilde):
    x = []
    y_tilde = []

    #start at index N
    for r in range(K_tilde - N):
        x.append(S_n[X[r:r+N]])
        y_tilde.append(S_n[X[r + N]])

    #D_tilde = list(zip(x, y_tilde))
    return x, y_tilde

def construct_empirical_distribution(x, y_tilde, N, S_n):
    #create empirical distributions, remove duplicates
    K_tilde = len(x)
    mu_0 = []
    y = []

    for k in range(K_tilde):

        candidate_x = np.array([dirac(Q_n(x[k][i], S_n)) for i in range(N)])
        candidate_y = dirac(Q_n(y_tilde[k], S_n))

        if k > 0:
            matches = np.all(np.all(np.array(mu_0) == candidate_x, axis = 2), axis=1)
        else:
            matches = []

        if np.any(matches):
            idx = np.where(matches == True)[0][0] # we choose the first occurance
            y[idx] += candidate_y

        else:
            y.append(candidate_y)
            mu_0.append(candidate_x)

    for k in range(len(y)):
        denom = np.sum(y[k])
        y[k] /= denom

    return np.array(mu_0), y


# VISUALIZATION

def visualize(mu_arr, y_labels, RESULTS_PATH, name):
    #first figure
    filename = name + "-1.png"
    predicted = top_measure_to_state(mu_arr[T])[:, -1]
    label_states, label_probs = measure_to_state(np.array(y_labels))

    plt.plot(predicted, 's', label=f'after', color='orange')
    K_viz = len(y_labels)
    for k in range(K_viz):
        xs = np.full(len(label_states[k]), k)
        plt.scatter(xs, label_states[k], s=np.array(label_probs[k]) * 300,
                    color='C0', alpha=0.6, label='label' if k == 0 else None)


    plt.xlabel(r"$\mathcal{K}$ index")
    plt.ylabel("State")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.close()

    #second figure
    filename = name + "-2.png"

    loss_before = C_T(mu_arr[0], y_labels, LOSS)
    loss_after = C_T(mu_arr[T], y_labels, LOSS)
    promptwise_loss_before = [cross_entropy(mu_arr[0][k][-1], y_labels[k]) for k in range(K_viz)]
    promptwise_loss_after = [cross_entropy(mu_arr[T][k][-1], y_labels[k]) for k in range(K_viz)]

    #plt.fill_between(np.arange(K_viz), promptwise_loss_before, alpha=0.25, label=f'before (loss = {loss_before:.2f})', color='blue')
    plt.fill_between(np.arange(K_viz), promptwise_loss_after, alpha=0.25, label=f'after (loss = {loss_after:.2f})', color='orange')

    plt.xlabel(r"$\mathcal{K}$ index")
    plt.ylabel("Prompt-level loss between predicted and actual labels")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.close()

    return loss_after


# PARAMETERS

T = 2       # time horizon = number of layers

l = 9       # probability measure quantizations

n = 5       # state space quantizations

m = 3       # action space quantization

N = 1       # length of prompt/order of markov chain

K_tilde = 98 #number of training pairs (state labels)

beta = 0.3  # attention temperature

LOSS = 'Cross Entropy'


#STATE

# S is the interval [0, 5]
S = (-2.5, 2.5)
#d = 1


#QUANTIZERS

#create example quantized state space
S_n = np.linspace(*S, n)

#used for W2
cost_matrix = np.array([[np.linalg.norm(s1 - s2)**2 for s1 in S_n] for s2 in S_n])

P_l = {i/(l -1) for i in range(0, l)}
U_m = Action((1, 1), (-2, 2), (-2 , 2), (1,1), (-2,2), (-2, 2), m = m)
