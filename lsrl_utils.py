import numpy as np
#import matplotlib.pyplot as plt


# Compute (A + uv^T)^{-1} from A^{-1}, u, and v, using the Sherman-Morrison formula
# (https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
def sherman_morrison_update(a_inv, u, v):
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)  # Reshape to column vectors
    norm = 1. + np.linalg.multi_dot([v.T, a_inv, u])
    res = a_inv - np.linalg.multi_dot([a_inv, u, v.T, a_inv]) / norm

    return res
# End fn sherman_morrison_update


#def plot_weights_convergence(w):
#    w_adj_dists = []
#    horizon = w.shape[0]
#    n_episodes = w.shape[1]
#
#    for h in range(horizon):
#        print('h = %d' % h)
#        w_adj_dists.append([])
#        for k in range(n_episodes-1):
#            w_adj_dists[h].append(max((np.linalg.norm(w[h][k] - w[h][k1], ord=np.inf) for k1 in range(k, n_episodes))))
#        # End for k
#    # End for h
#
#    w_adj_dists = np.array(w_adj_dists)
#    print('Done\n')
#    w_adj_dists_max = np.linalg.norm(w_adj_dists.T, ord=np.inf, axis=-1)
#    w_adj_dists_bound = np.array([1.0 if w_adj_dists_max[t] > 10.0/(np.sqrt(n_episodes)+1) else 0.0
#                                  for t in range(w_adj_dists_max.shape[0])])
#    print('\n')
#    for h in range(horizon):
#        plt.plot(np.arange(1, n_episodes), w_adj_dists[h], label=('h = %d' % h))
#    plt.plot(np.arange(1, n_episodes), w_adj_dists_bound, label='bound indicator')
#    plt.title('max subsequent w l_infty distances')
#    plt.xlabel('k')
#    plt.ylabel('||w_k - w_{k-1}||_infty')
#    plt.legend()
#    plt.show()
## End fn plot_weights_convergence
