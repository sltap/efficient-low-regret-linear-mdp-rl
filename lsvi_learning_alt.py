import numpy as np
import time

from lsrl_utils import sherman_morrison_update

from scipy.spatial.distance import pdist

from sys import getsizeof

class LSVIHistory:
    def __init__(self, int_size, threshold, H, d):
        self.int_size = int_size
        self.H = H
        self.d = d
        if threshold is None:
            self.threshold = 0.1
        else:
            self.threshold = threshold
        self.buffer = np.zeros(shape=(H, int_size, d*d))
        self.idxs = [0 for _ in range(H)]
        self.sizes = [0 for _ in range(H)]
    # End fn __init__
    
    def add(self, h, M):
        self.buffer[h, self.idxs[h]] = M.flatten()
        self.idxs[h] += 1
        self.sizes[h] = min(self.int_size, self.sizes[h] + 1)
        if self.idxs[h] >= self.int_size:
            self.idxs[h] = 0
    # End fn add
    
    def size(self, h):
        return self.sizes[h]
    # End fn size
    
    def mpwdist(self, h):
        pwdists = pdist(self.buffer[h], metric='euclidean') # Effectively, computes the Frobenius norm
        return np.max(pwdists)
     # End fn mpwdist
     
    def learning_cond(self, h):
        return self.mpwdist(h) > self.threshold*self.d*self.d
    # End fn learning_cond
    
    def __sizeof__(self):
        return getsizeof(self.int_size)*4 + self.buffer.nbytes + getsizeof(self.idxs)*2
    # End fn __sizeof__
# End class LSVIHistory

# Vectorized LSVI-UCB implementation with fixed learning alternation
def lsvi_ucb_alt_learning_fixed(mdp, K, lambbda, beta, V_opt_zero, learn_iters_base, scale_factor, total_learn_iters):
    A = mdp.A
    d = mdp.d
    H = mdp.H

    r = total_learn_iters
    arr_index = 0
    cur_learn_iters = learn_iters_base
    is_learning = True
    learning_finished = False
    cur_iter_ctr = 0

    if K % 100 == 0:
        print('lsvi_ucb_alt_learning :: K = %d, r = %d, start = %d' % (K, r, cur_learn_iters))
    t1 = time.process_time_ns()

    Lambda_inv = np.zeros(shape=(H, d, d))
    phi_alg_prev = np.zeros(shape=(H, d))

    Phi_acts = np.zeros(shape=(H, A, r, d))
    cur_Phi_acts = np.zeros(shape=(H, A, d))

    Phi_alg = np.zeros(shape=(H, r, d))
    rewards = np.zeros(shape=(H, r))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    q_vec = np.zeros(shape=(H, r))

    w = np.zeros(shape=(H, d))
    TR = 0.

    for k in range(K):
        if is_learning:
            cur_iter_ctr += 1
            if cur_iter_ctr > cur_learn_iters:
                is_learning = False
                if cur_learn_iters <= 1:
                    learning_finished = True
            # End if cur_iter_ctr > cur_learn_iters
        # End if is_learning

        if is_learning is False:
            cur_iter_ctr -= 1
            if cur_iter_ctr <= 0 and learning_finished is False:
                cur_learn_iters = int(cur_learn_iters * scale_factor)
                cur_iter_ctr = 0
                if cur_learn_iters < 1:
                    learning_finished = True
                else:
                    is_learning = True
            # End if cur_iter_ctr <= 0 and not learning_finished
        # End if not is_learning

        # Inverse Covariance update
        if is_learning:
            for h in range(0, H - 1, 1):
                if k > 0:
                    Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
                else:
                    Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
                # End if
            # End for (h)

        # Policy formulation
        if is_learning:
            for h in range(H - 1, -1, -1):
                if h < H - 1:
                    q_vec[h] = np.maximum.reduce(np.clip(np.array([np.dot(Phi_acts[h + 1, a], w[h + 1]) + beta * np.sqrt(
                        np.diagonal(Phi_acts[h + 1, a] @ Lambda_inv[h] @ Phi_acts[h + 1, a].T))
                                              for a in range(A)]), a_min=None, a_max=H))
                w[h] = np.linalg.multi_dot([Lambda_inv[h], Phi_alg[h].T, rewards[h] + q_vec[h]])
            # End for (h)
        # End if (is_learning)

        # Learning
        episode_total_reward = 0.
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                cur_Phi_acts[h, a] = mdp.query_phi(a)
            # End for
            # Choose action from UCB-regularized greedy policy
            opt_a = np.argmax(np.clip(np.array([np.dot(cur_Phi_acts[h, a], w[h]) + beta * np.sqrt(
                np.dot(cur_Phi_acts[h, a], Lambda_inv[h] @ cur_Phi_acts[h, a])) for a in range(A)]), a_min=None,
                                      a_max=H))
            # Take action and aggregate reward
            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            phi_alg_prev[h] = phi

            if is_learning:
                for a in range(A):
                    Phi_acts[h, a, arr_index, :] = cur_Phi_acts[h, a]
                rewards[h, arr_index] = reward
                Phi_alg[h, arr_index, :] = phi
            # End if is_learning
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)

        if is_learning and not learning_finished:
            arr_index += 1
            assert arr_index < r
        # End if
    # End for (k)

    apx_space_usage = Lambda_inv.nbytes + phi_alg_prev.nbytes + Phi_acts.nbytes + cur_Phi_acts.nbytes + Phi_alg.nbytes + rewards.nbytes + q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_alt_learning_fixed

# Vectorized LSVI-UCB implementation with adaptive learning alternation
def lsvi_ucb_alt_learning_adaptive(mdp, K, lambbda, beta, V_opt_zero, total_learn_iters, min_phase_len, lookback_period, alt_threshold):
    A = mdp.A
    d = mdp.d
    H = mdp.H

    r = total_learn_iters
    arr_index = [0 for _ in range(H)]
    is_learning = [True for _ in range(H)]
    learning_finished = [False for _ in range(H)]
    cur_learn_iter_ctr = [0 for _ in range(H)]
    history = LSVIHistory(int_size=lookback_period, threshold=alt_threshold, H=H, d=d)

    if K % 100 == 0:
        print('lsvi_ucb_alt_learning :: K = %d, r = %d' % (K, r))
    t1 = time.process_time_ns()

    Lambda_inv = np.zeros(shape=(H, d, d)) # Only including the learning iterations
    phi_alg_prev = np.zeros(shape=(H, d))
    temp_lambda_inv = np.zeros(shape=(H, d, d)) # Including all iterations

    Phi_acts = np.zeros(shape=(H, A, r, d))
    cur_Phi_acts = np.zeros(shape=(H, A, d))

    Phi_alg = np.zeros(shape=(H, r, d))
    rewards = np.zeros(shape=(H, r))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    q_vec = np.zeros(shape=(H, r))

    w = np.zeros(shape=(H, d))
    TR = 0.

    for k in range(K):
        # Inverse Covariance update
        for h in range(0, H - 1, 1):
            if k > 0:
                temp_lambda_inv[h] = sherman_morrison_update(temp_lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
                if is_learning[h]:
                    Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h]) 
            else:
                temp_lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
                Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
            # End if
            history.add(h, temp_lambda_inv[h])
        # End for (h)
        
        # Update is_learning and learning_finished
        if is_learning[h]:
            cur_learn_iter_ctr[h] += 1
            if cur_learn_iter_ctr[h] > min_phase_len:
                if not history.learning_cond(h):
                    is_learning[h] = False
        else:
            if history.learning_cond(h) and not learning_finished[h]:
                cur_learn_iter_ctr[h] = 0
                is_learning[h] = True
        # End if
        
        # Policy formulation
        for h in range(H - 1, -1, -1):
            if is_learning[h]:
                if h < H - 1:
                    q_vec[h] = np.maximum.reduce(np.clip(np.array([np.dot(Phi_acts[h + 1, a], w[h + 1]) + beta * np.sqrt(
                        np.diagonal(Phi_acts[h + 1, a] @ Lambda_inv[h] @ Phi_acts[h + 1, a].T))
                                              for a in range(A)]), a_min=None, a_max=H))
                w[h] = np.linalg.multi_dot([Lambda_inv[h], Phi_alg[h].T, rewards[h] + q_vec[h]])
            # End if is_learning[h]
        # End for (h)

        # Learning
        episode_total_reward = 0.
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                cur_Phi_acts[h, a] = mdp.query_phi(a)
            # End for
            # Choose action from UCB-regularized greedy policy
            opt_a = np.argmax(np.clip(np.array([np.dot(cur_Phi_acts[h, a], w[h]) + beta * np.sqrt(
                np.dot(cur_Phi_acts[h, a], Lambda_inv[h] @ cur_Phi_acts[h, a])) for a in range(A)]), a_min=None,
                                      a_max=H))
            # Take action and aggregate reward
            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            phi_alg_prev[h] = phi

            if is_learning[h]:
                for a in range(A):
                    Phi_acts[h, a, arr_index[h], :] = cur_Phi_acts[h, a]
                rewards[h, arr_index[h]] = reward
                Phi_alg[h, arr_index[h], :] = phi
                if arr_index[h] < r-1:
                    arr_index[h] += 1
                else: # Spent the learning space budget
                    is_learning[h] = False
                    learning_finished[h] = True
                # End if
            # End if is_learning[h]
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)
    # End for (k)

    apx_space_usage = history.__sizeof__() + 18*H + Lambda_inv.nbytes + temp_lambda_inv.nbytes + phi_alg_prev.nbytes + Phi_acts.nbytes + cur_Phi_acts.nbytes + Phi_alg.nbytes + rewards.nbytes + q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_alt_learning_adaptive
