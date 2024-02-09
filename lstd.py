import numpy as np

import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng()

def _random_unit_vector(d):
    v = rng.standard_normal(d) # See https://mathworld.wolfram.com/SpherePointPicking.html
    return v/np.linalg.norm(v)
# End fn _random_unit_vector

def _random_simplex_vector(d):
    u = np.hstack((rng.uniform(0., 1., d-1), np.array([0., 1.]))) # Rubinstein and Kroese (2007). See Algorithm 1 in https://iew.technion.ac.il/~onn/Selected/AOR09.pdf
    u.sort()
    v = np.roll(u, -1) - u
    return v[:-1]
# End fn _random_simplex_vector

class LinearMDP:
    def __init__(self, S, A, d, H, s_0=0):
        self._state = s_0 # current state
        self._h = 0       # current step index (within episode)
        self._trajectory = ['E', 'S%d' % s_0] # the trajectory of the MDP (s_0, a_0, r_0, s_1, a_1, r_1, ...)

        assert (isinstance(S, int) and S >= 1)
        self.S = S      # num. of states
        assert (isinstance(A, int) and A >= 1)
        self.A = A      # num. of actions
        assert (isinstance(d, int) and d >= 1)
        self.d = d      # feature mapping dimension
        assert (isinstance(H, int) and H >= 1)
        self.H = H      # planning horizon
        assert (s_0 in range(S))
        self.s_0 = s_0  # fixed initial state (for each episode)
        
        self.phi_map = LinearMDP.init_random_feature_map(S, A, d)  # a dict, with phi_map[(s,a)] = \phi(s,a)
        self.reward_weights = LinearMDP.init_random_reward_weights(d, H) # a list, with reward_weights[h] = \theta_h \in \RR^d
        self.transition_measures = LinearMDP.init_random_transition_measures(S, d, H) # a list, with transition_measures[h] = \mu_h \in \RR^{S \times d} with each column giving a measure
    # End fn __init__
    
    def sample_reward(self, expected_reward):
        r_sigma = 0.02
        return np.clip(rng.normal(expected_reward, r_sigma), 0., 1.) # Assume clipped normal reward distribution around expected reward
    # End fn sample_reward
    
    def init_random_feature_map(S, A, d):
        phi_map = {}
        for s in range(S):
            for a in range(A):
                phi_map[(s, a)] = _random_simplex_vector(d)
        return phi_map
     # End fn init_random_feature_map
     
    def init_random_reward_weights(d, H):
        reward_weights = []
        for h in range(H):
            reward_weights.append(_random_simplex_vector(d)) # reward weights in random simplex => rewards in [0, 1]
        return reward_weights
    # End fn init_random_reward_weights
    
    def init_random_transition_measures(S, d, H):
        transition_measures = []
        for h in range(H):
            mu_h = np.hstack([_random_simplex_vector(S)[:,np.newaxis] for i in range(d)])
            transition_measures.append(mu_h)
        return transition_measures
    # End fn init_random_transition_measures
    
    # Query phi(current_state, a)
    def query_phi(self, a):
        return self.phi_map[(self._state, a)]
    # End fn query_phi

    def take_action(self, a):
        phi = self.phi_map[(self._state, a)]
        
        # Compute the expected reward and sample from the reward distribution
        expected_reward = np.dot(self.reward_weights[self._h], phi)
        reward = self.sample_reward(expected_reward)

        self._trajectory.append('A%d' % a)
        self._trajectory.append('R%f' % reward)
        
        # Compute the transition distribution and sample the next state
        transition_distr = np.matmul(self.transition_measures[self._h], phi[:,np.newaxis])
        next_state = rng.choice(S, p=transition_distr.ravel())
        
        # Update the episode index (modulo H)
        self._h = self._h + 1
        if self._h == H: # Begin the new episode (reset the MDP)
            self._h = 0
            next_state = self.s_0
            self._trajectory.append('E')

        self._state = next_state
        self._trajectory.append('S%d' % next_state)
        
        return reward, phi
    # End fn take_action
# End class LinearMDP

# Compute (A + uv^T)^{-1} using the Sherman-Morrison formula (https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
def _sherman_morrison_update(A_inv, u, v):
    u, v = u.reshape(-1,1), v.reshape(-1,1) # Reshape to column vectors
    Z = 1. + np.linalg.multi_dot([v.T, A_inv, u])
    R = A_inv - np.linalg.multi_dot([A_inv, u, v.T, A_inv]) / Z

    return R
# End fn _sherman_morrison_update

if __name__ == '__main__':
    K = 15000
    S = 100
    A = 8
    d = 10
    H = 10
    lambbda = 0.01 # Since 'lambda' is a Python keyword
    beta = 0.01
    mdp = LinearMDP(S, A, d, H)
    
    Phi_acts = [[np.zeros(shape=(K,d)) for a in range(A)] for h in range(H)]
    
    Lambda_inv = [None for h in range(H)]
    w = [[] for h in range(H)]
    Phi_alg = [np.zeros(shape=(K,d)) for h in range(H)]
    rewards = [np.zeros(shape=(K,)) for h in range(H)]
    q_vec = [np.zeros(shape=(K,)) for h in range(H)] # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm", page 10, eqn no 22 in notes)
    
    for k in range(K):
        if k % 500 == 0: print('k = %d' % k)
        # Policy formulation
        for h in range(H-1,-1,-1):
            if k > 0:
                Lambda_inv[h] = _sherman_morrison_update(Lambda_inv[h], Phi_alg[h][k-1,:], Phi_alg[h][k-1,:])
            else:
                Lambda_inv[h] = (1./lambbda)*np.eye(d, dtype=np.float64)
            # End if
            if h < H-1:
                q_vec[h] = np.maximum.reduce([np.clip(Phi_acts[h+1][a] @ w[h+1][-1], a_min=None, a_max=H) for a in range(A)])
            w[h].append(np.linalg.multi_dot([Lambda_inv[h], Phi_alg[h].T, rewards[h] + q_vec[h]]))
        # End for
        # Learning
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                Phi_acts[h][a][k,:] = mdp.query_phi(a)
            # End for
            
            opt_a = np.argmax(np.clip(np.array([np.dot(Phi_acts[h][a][k,:], w[h][-1]) for a in range(A)]), a_min=None, a_max=H))
            reward, phi = mdp.take_action(opt_a)
            rewards[h][k] = reward
            Phi_alg[h][k,:] = phi
    # End for
    
    w_adj_dists = np.array([np.array([np.linalg.norm(w[h][k] - w[h][k-1], ord=np.inf) for k in range(1,K)]) for h in range(H)])
    w_adj_dists_max = np.linalg.norm(w_adj_dists.T, ord=np.inf, axis=-1)
    w_adj_dists_bound = np.array([1.0 if w_adj_dists_max[t] > 1.0/np.sqrt(K) else 0.0 for t in range(w_adj_dists_max.shape[0])])
    print('\n')
    for h in range(H):
        plt.plot(np.arange(1,K), w_adj_dists[h], label=('h = %d' % h))
    plt.plot(np.arange(1,K), w_adj_dists_bound, label='bound')
    plt.title('adjacent w l_infty distances')
    plt.xlabel('k')
    plt.ylabel('||w_k - w_{k-1}||_infty')
    plt.legend()
    plt.show()

#    print('\n\nTRAJECTORY\n-------------')
#    print(mdp._trajectory)
# End if
