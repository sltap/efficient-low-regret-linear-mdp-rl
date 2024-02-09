import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import time

from transform import CountSketchTransform
from scipy.sparse import csc_matrix

from lsrl_utils import sherman_morrison_update

def lsvi_ucb_sketched_learning(mdp, K, r, lambbda, beta, V_opt_zero):
    A = mdp.A
    d = mdp.d
    H = mdp.H

    if K % 100 == 0:
        print('lsvi_ucb_sketched_learning(%d) :: K = %d' % (r, K))
    t1 = time.process_time_ns()

    sketch = CountSketchTransform(K, d+1, r=r)

    Lambda_inv = np.zeros(shape=(H, d, d))
    phi_alg_prev = np.zeros(shape=(H, d))

    s_Phi_acts = np.zeros(shape=(H, A, r, d))
    s_Phi_alg = np.zeros(shape=(H, r, d))
    s_rewards = np.zeros(shape=(H, r, 1))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    s_q_vec = np.zeros(shape=(H, r, 1))

    w = np.zeros(shape=(H, d, 1))
    TR = 0.

    for k in range(K):
        # Policy formulation
        for h in range(H - 1, -1, -1):
            if k > 0:
                Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
            else:
                Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
            # End if
            if h < H - 1:
                s_q_vec[h] = np.maximum.reduce([np.clip(s_Phi_acts[h + 1,a] @ w[h + 1] + beta * np.sqrt(
                    np.diagonal(s_Phi_acts[h + 1,a] @ Lambda_inv[h] @ s_Phi_acts[h + 1,a].T)).reshape(-1, 1),
                                                        a_min=None, a_max=H)
                                                for a in range(A)])
            w[h] = np.linalg.multi_dot([Lambda_inv[h], s_Phi_alg[h].T, s_rewards[h] + s_q_vec[h]])
        # End for (h)

        # Learning
        episode_total_reward = 0.
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                phi_a = mdp.query_phi(a)
                Phi_a_tensored = csc_matrix((phi_a.ravel(),
                                             k*np.ones(shape=(d,), dtype=np.int32),
                                             np.arange(d + 1)), shape=(K, d))
                s_Phi_acts[h][a] = s_Phi_acts[h][a] + sketch.transform(Phi_a_tensored).todense()
            # End for

            Q_est = np.clip(np.array([np.dot(mdp.query_phi(a), w[h]) + beta * np.sqrt(
                np.dot(mdp.query_phi(a), Lambda_inv[h] @ mdp.query_phi(a))) for a in range(A)]), a_min=None,
                                      a_max=H)
            opt_a = np.argmax(Q_est)

            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            s_rewards[h] = s_rewards[h] + sketch.transform(csc_matrix((np.array([reward]),
                                                                       np.array([k]),
                                                                       np.array([0, 1])),
                                                                      shape=(K, 1))).todense()
            Phi_tensored = csc_matrix((phi.ravel(), k * np.ones(shape=(d,), dtype=np.int32), np.arange(d + 1)),
                                      shape=(K, d))
            s_Phi_alg[h] = s_Phi_alg[h] + sketch.transform(Phi_tensored).todense()
            phi_alg_prev[h] = phi
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)
    # End for (k)

    apx_space_usage = sketch.__sizeof__() + Lambda_inv.nbytes + phi_alg_prev.nbytes + s_Phi_acts.nbytes + s_Phi_alg.nbytes + s_rewards.nbytes + s_q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_sketched_learning
