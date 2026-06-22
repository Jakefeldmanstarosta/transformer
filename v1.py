
import numpy as np
from scipy.optimize import linprog


def quantize_state(s, S_n):
    diffs = np.abs(s - S_n)
    return S_n[np.argmin(diffs)]


# to impliment: Reznik's discrete probability distribution quantization algorithm  
def quantize_probability(p, l):
    return round(p * (l - 1))/ (l-1) 



def dirac(s):
    mu_i = np.zeros(n)
    idx = np.where(S_n == s)[0][0]
    mu_i[idx] = 1
    return mu_i


def measure_to_state(mu):  #works for any shape of mu (mu_i/mu_k/mu_t)
    return np.argmax(mu, axis = -1)


def W2(p, q):
    #note the cost is defined for P(S_n) x P(S_n)
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


#data-label pairs
x = [[1,1],
     [1,1],
     [2,2],
     [3,3],
     [3,3],]
y_tilde = [1, 1, 1, 2, 2]
# S is the interval [0, 5]

K = len(x)
N = len(x[0])


#1
D_tilde = list(zip(x, y_tilde))

# time horizon = number of layers 
T = 2

# probability measure quantization parameter
# (number of quantization levels = number of bins + 1) (ie l = 2 gives {0, 1})
l = 9

#state space quantization parameter
# (number of states = number of bins + 1)
n = 6

#action space quantization parameter
m = 5


#2
#create example quantized state space
S_n = []
for i in range(n):
    S_n.append(i)
S_n = np.array(S_n)


def Q_n(x):
    return (quantize_state(x, S_n))
S_n


cost_matrix = np.array([[np.linalg.norm(s1 - s2)**2 for s1 in S_n] for s2 in S_n]) #used for W2


#3
U_m = [-2, -1, 0, 1, 2]
#for our example, we add the weights


#4
#assume l > 0
P_l = {i/(l -1) for i in range(0, l)}

def R_l(mu_k):
    quantized = [[quantize_probability(x, l) for x in mu_i] for mu_i in mu_k]
    return quantized
# we have to choose a suitible l given our number of states n or else the probabilities could technically disapear. 
# i think taking l > n suffices, but i have to check


#5-9
mu_0 = []
y = []
k_unique = 0

#create empirical distributions, remove duplicates, update K to reflect the number of unique labels 
for k in range(K):
    
    candidate_x = np.array([dirac(Q_n(x[k][i])) for i in range(N)])
    candidate_y = dirac(Q_n(y_tilde[k]))

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
        k_unique += 1

K = k_unique

for k in range(K):
    denom = np.sum(y[k])
    y[k] /= denom


#initialize lifted probability measures
mu_0 = np.array(mu_0)
mu = np.zeros((T + 1, *mu_0.shape))
mu[0] = mu_0
#mu has shape (T + 1, K, N, len(S_n))


#10
def f(s, mu_i, u, S = S_n): #arbitrary example function right now. to impliment attention.
    #dynamics here are done at the state level. can equally be done at the individual measure level 
    if (s + u) in S:
        return s + u
    return s

def phi_n(u, mu_k):
    result = []
    for x in mu_k:
        s = measure_to_state(x)
        s = Q_n(f(s, x, u))
        result.append(dirac(s))
    return np.array(result)

def phi(u, mu_t):
    result = []
    for mu_k in mu_t:
        result.append(R_l(phi_n(u, mu_k)))
    return np.array(result)


#11 
def C_T(mu_T, y = y):
    sum = 0
    for k in range(K):
        sum += W2(mu_T[k][N-1], y[k])
    return sum / K

def create_ensembles(num_toks = n, toks_per_prompt = N, S = S_n): 
    #return all the ways to assign each state in S_n to each K x N slots to create the set P^l(X_n)^K
    #unfortunately, this explodes computationally with n ^ (K * N) iterations
    indices = [0] * K * N

    while True:
        yield np.array([dirac(i) for i in indices]).reshape(K, N, n)

        #odometer counting
        pos = 0
        while pos < N * K:
            indices[pos] += 1 
            if indices[pos] > n - 1:
                indices[pos] = 0
                pos += 1 
            else: break

        if pos >= N * K:
            return
        
def ensemble_to_index(mu_t): 
    state = np.argmax(mu_t, axis = -1).flatten() #look at the S_n axis
    index = sum(state[i] * (n ** i) for i in range(N * K))
    return index

num_states = (n) ** (K * N)
C = np.zeros((T + 1, num_states)) #for each time step and state, we find the optimal cost 

for i, mu_T in enumerate(create_ensembles()):
    C[T][i] = C_T(mu_T)
    if i % 1000 == 0:
        print(i)


#12 - 17
gamma = np.zeros((T, num_states))  #for each time step and state, we find the optimal action

for t in range(T-1, -1, -1): #goes from T-1 to 0
    for i, P in enumerate(create_ensembles()):

        costs = [C[t+1][ensemble_to_index(phi(u, P))] for u in U_m] #costs indexed by each action

        gamma[t][i] = U_m[np.argmin(costs)]
        C[t][i] = np.min(costs)

    print(t)

#18 - 24
#Forward pass
U_t = []
for t in range(T): 
    optimal_u = gamma[t][ensemble_to_index(mu[t])]
    mu[t + 1] = phi(optimal_u, mu[t])
    U_t.append(optimal_u)


#25
U_t

#should be [-2.0, 1.0]


