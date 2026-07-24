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


# BEEFY FUNCTIONS

def create_transition(n, N, chain = 'deterministic'):
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

    if chain == 'dobrushin':
        return None

def generate_process(n, K_tilde, N, xi, transition, chain = 'deterministic'):
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

    if chain == 'dobrushin':

        q = t = n/2
        simga_q = 0
        # we take xi = sigma_t/t \in {1, 1.5, 2}
        sigma_t = xi * t

        #as a technicality, we select the following +/-t
        t_pos = np.ceil((n-1)/2)
        t_neg = np.floor((1-n)/2)


        X.append(0)
        Y.append(int(X[-1] + t))
        for r in range(1, n_samples):
            X.append(reflect(np.random.normal(X[-1], sigma_t), t_neg, t_pos))
            Y.append(int(X[-1] + t))

        return Y

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

m = 6       # action space quantization 

N = 1       # length of prompt/order of markov chain 

K_tilde = 98 #number of training pairs (state labels) 

#xi = 1.0    # 0 = non-Markovian, 1 = Markovian

beta = 0.3  # attention temperature  

LOSS = 'Cross Entropy'

CHAIN = "dobrushin"


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

    markovian_xi = [0.01, 0.5, 1.0]
    stability_xi = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    if CHAIN == "dobrushin":
        xi_values = stability_xi
    else:
        xi_values = markovian_xi

    losses = []
    num_trials = 3
    transition = create_transition(n, N, chain = CHAIN)

    for xi in xi_values:
        loss_sum = 0
        for trial in range(num_trials):
            actions, mu, y = train(xi, transition)
            visualize(mu, y, r"C:\Users\jakef\Projects\next-token-prediction-control\results", name = "train" + str(xi) + "-trial" + str(trial))
            mu_test, y_test = test(actions, xi, transition)
            loss = visualize(mu_test, y_test, r"C:\Users\jakef\Projects\next-token-prediction-control\results", "test" + str(xi) + "-trial" + str(trial))

            loss_sum += loss

        losses.append(loss_sum / num_trials)
    
    print(losses, xi_values)
    plt.plot(xi_values, losses)
    plt.xlabel("xi value")
    plt.ylabel("Loss")
    plt.show()