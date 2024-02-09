import numpy as np

from numpy.random import default_rng

from abc import ABC, abstractmethod

random_seed = 1542973613
rng = default_rng(random_seed)


def _random_unit_vector(d: int):
    v = rng.standard_normal(d)  # See https://mathworld.wolfram.com/SpherePointPicking.html
    return v/np.linalg.norm(v)
# End fn _random_unit_vector


def _random_simplex_vector(d: int):
    # Rubinstein and Kroese (2007). See Algorithm 1 in https://iew.technion.ac.il/~onn/Selected/AOR09.pdf
    u = np.hstack((rng.uniform(0., 1., d-1), np.array([0., 1.])))
    u.sort()
    v = np.roll(u, -1) - u
    return v[:-1]
# End fn _random_simplex_vector

class LinearMDP(ABC):
    def __init__(self, A: int, d: int, H: int, seed: int, reward_variance: float):
        self._state = self.reset(seed)  # current state
        self._h = 0  # current step index (within episode)
        self.reward_variance = reward_variance

        assert (A >= 1)
        self.A = A  # num. of actions
        assert (d >= 1)
        self.d = d  # feature mapping dimension
        assert (H >= 1)
        self.H = H  # planning horizon
    # End fn __init__
    
    def reset(self, seed=None):
        pass
    
    def sample_reward(self, expected_reward: float):
        # Assume clipped normal reward distribution around expected reward
        return np.clip(rng.normal(expected_reward, self.reward_variance), 0., 1.)
    # End fn sample_reward
    
    # Return phi(current_state, a)
    @abstractmethod
    def query_phi(self, a):
        pass
    
    # Return phi(s, a) - note that this is beyond the RL paradigm.
    @abstractmethod
    def _get_phi(self, s, a):
        pass
    
    # Return r_h(current_state, a)
    @abstractmethod
    def get_expected_reward(self, h, a):
        pass
    
    # Return state s' sampled from P_h(current_state, a)
    @abstractmethod
    def get_next_state(self, h, a):
        pass
    
    def take_action(self, a):
        phi = self.query_phi(a)
        
        # Compute the expected reward and sample from the reward distribution
        expected_reward = self.get_expected_reward(self._h, a)
        reward = self.sample_reward(float(expected_reward))
        
        # Compute the transition distribution and sample the next state
        next_state = self.get_next_state(self._h, a)
        
        # Update the episode index (modulo H)
        self._h = self._h + 1
        if self._h == self.H:  # Begin the new episode (reset the MDP)
            self._h = 0
            next_state = self.reset()
        
        self._state = next_state
        
        return reward, phi
    # End fn take_action
# End class LinearMDP

class TabLinearMDP(LinearMDP):
    def __init__(self, S: int, A: int, d: int, H: int, s_0: int = 0, seed: int = 0, reward_variance: float = 0.02):
        self.s_0 = s_0
        assert (S >= 1)
        self.S = S
        super().__init__(A, d, H, seed, reward_variance)
    # End fn __init__
    
    def reset(self, seed=None):
        return self.s_0
    # End fn reset
    
    def init_random_mdp(self):
        # a dict, with phi_map[(s,a)] = \phi(s,a)
        self.phi_map = TabLinearMDP.init_random_feature_map(self.S, self.A, self.d)
        # a list, with reward_weights[h] = \theta_h \in \RR^d
        self.reward_weights = TabLinearMDP.init_random_reward_weights(self.d, self.H)
        # a list, with transition_measures[h] = \mu_h \in \RR^{S \times d} with each column giving a measure
        self.transition_measures = TabLinearMDP.init_random_transition_measures(self.S, self.d, self.H)
    # End fn init_random_mdp
    
    @staticmethod
    def init_random_feature_map(S: int, A: int, d: int):
        phi_map = {}
        for s in range(S):
            for a in range(A):
                phi_map[(s, a)] = _random_simplex_vector(d)
        return phi_map
    # End fn init_random_feature_map

    @staticmethod
    def init_random_reward_weights(d: int, H: int):
        reward_weights = []
        for h in range(H):
            reward_weights.append(_random_simplex_vector(d))  # reward weights in random simplex => rewards in [0, 1]
        return reward_weights
    # End fn init_random_reward_weights

    @staticmethod
    def init_random_transition_measures(S: int, d: int, H: int):
        transition_measures = []
        for h in range(H):
            mu_h = np.hstack([_random_simplex_vector(S)[:, np.newaxis] for _ in range(d)])
            transition_measures.append(mu_h)
        return transition_measures
    # End fn init_random_transition_measures

    # Return phi(current_state, a)
    def query_phi(self, a):
        return self.phi_map[(self._state, a)]
    # End fn query_phi
    
    # Return phi(s, a) - note that this is beyond the RL paradigm.
    def _get_phi(self, s, a):
        return self.phi_map[(s, a)]
    # End fn _get_phi
    
    # Return r_h(current_state, a)
    def get_expected_reward(self, h, a):
        phi = self.query_phi(a)
        return np.dot(self.reward_weights[h], phi)
    # End fn get_expected_reward
    
    # Return state s' sampled from P_h(current_state, a)
    def get_next_state(self, h, a):
        phi = self.query_phi(a)
        transition_distr = np.matmul(self.transition_measures[h], phi[:, np.newaxis])
        return rng.choice(range(self.S), p=transition_distr.ravel())
    # End fn get_next_state
    
    ######################################################################################################
    # THE FOLLOWING FNS GIVE INFO BEYOND RL PARADIGM. REQUIRED FOR MDP LEARNING (VI) TO GET TRUE OPTIMAL.
    ######################################################################################################
    
    # Return P_h(s,a) \in \Delta(S)
    def P(self, h: int, s, a):
        return np.matmul(self.transition_measures[h], self._get_phi(s, a)[:, np.newaxis]).ravel()
    # End fn P
    
    # Return r_h(s, a)
    def r(self, h: int, s, a):
        return np.dot(self.reward_weights[h], self._get_phi(s, a))
    # end fn r
    
    # Uses Q-Value iteration to learn an (approximately) optimal policy
    def get_optimal_policy(self, conv_dist: float = 1e-4):
        # Compute the optimal policy using Q-value iteration
        Q = [np.zeros(shape=(self.S, self.A)) for _ in range(self.H+1)]
        delta = np.infty

        while delta > conv_dist:
            delta = 0.
            for h in range(self.H-1, -1, -1):
                for s in range(self.S):
                    for a in range(self.A):
                        temp = Q[h][s, a]
                        Q[h][s, a] = self.r(h, s, a) + np.dot(self.P(h, s, a), np.max(Q[h+1], axis=1))
                        delta = max(delta, np.abs(temp - Q[h][s, a]))
                    # End for a
                # End for s
            # End for h
        # End while (delta > conv_dist)

        # Return the greedy policy corresponding to the converged Q function
        return [np.argmax(Q[h], axis=1) for h in range(self.H)]
    # End fn get_optimal_policy

    # Compute V^pi from a given policy pi
    def evaluate_policy(self, pi):
        V = [np.zeros(shape=(self.S,)) for _ in range(self.H+1)]
        for h in range(self.H-1, -1, -1):
            for s in range(self.S):
                V[h][s] = self.r(h, s, pi[h][s]) + np.dot(self.P(h, s, pi[h][s]), V[h+1])
            # End for s
        # End for h
        return V
    # End fn evaluate_policy
# End class TabLinearMDP


# Test
if __name__ == '__main__':
    from argparse import ArgumentParser
    import pickle
    from pathlib import Path
    
    parser = ArgumentParser(prog="linear_mdp", description="Create and store a random linear MDP environment as a LinearMDP object.")
    
    parser.add_argument("num_states", type=int, help="The number of states in the linear MDP (S)")
    parser.add_argument("num_actions", type=int, help="The number of actions in the linear MDP (A)")
    parser.add_argument("embedding_dim", type=int, help="The feature embedding dimension (d), such that phi(s,a) |-> R^d")
    parser.add_argument("planning_horizon", type=int, help="The planning horizon (H), i.e. the length of each episode.")
    parser.add_argument("-o", "--output-file", default="linear_mdp.dat", help="File to store the LinearMDP object.")
    
    args = parser.parse_args()
    print('Arguments:', args)
    
    mdp = TabLinearMDP(S=args.num_states, A=args.num_actions, d=args.embedding_dim, H=args.planning_horizon)
    mdp.init_random_mdp()
    pi_opt = mdp.get_optimal_policy()
    V_opt = mdp.evaluate_policy(pi_opt)
    
    print('Value fn[h=0]', V_opt[0])
    print('Value fn[h=0,s=s_0]', V_opt[0][mdp.s_0])
    
    output_file_path = Path(args.output_file)
    with output_file_path.open(mode='wb') as out_file_obj:
        pickle.dump({'mdp': mdp, 'pi_opt': pi_opt, 'V_opt':V_opt}, out_file_obj)
    print('Wrote to file: ', output_file_path)
    print('END')
# End if
