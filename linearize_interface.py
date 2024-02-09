import numpy as np
import gym
import torch

import sys
sys.path.insert(0, './thirdparty/lvrep-rl/')

from agent.sac.sac_agent import SACAgent
from agent.ctrlsac.ctrlsac_agent import CTRLSACAgent

from linear_mdp import LinearMDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralLinearMDP(LinearMDP):
    def __init__(self, representation, env, H, seed=0):
        self.representation = representation
        self.env = env
        self.s_0 = 0 # For simplifying regret calculation between NeuralLinearMDP and TabLinearMDP
        super().__init__(A=env.action_space.n, d=representation.feature_dim, H=H, seed=seed, reward_variance=0.02)
    # End fn __init__
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)[0]
    # End fn reset
    
    # Return phi(current_state, a)
    def query_phi(self, a):
        return self.representation.phi(torch.from_numpy(self._state.astype(np.float32)).to(device), torch.from_numpy(np.array([a], dtype=np.float32)).to(device)).detach().cpu().numpy()
    # End fn query_phi
    
    # Return phi(s, a) - note that this is beyond the RL paradigm.
    def _get_phi(self, s, a):
        return self.representation.phi(torch.from_numpy(self._state.astype(np.float32)).to(device), torch.from_numpy(np.array([a], dtype=np.float32)).to(device)).detach().cpu().numpy()
    # End fn _get_phi
    
    # Return r_h(current_state, a)
    def get_expected_reward(self, h, a):
        phi = self.query_phi(a)
        return self.representation.theta(torch.from_numpy(phi.astype(np.float32)).to(device)).detach().cpu().numpy()
    # End fn get_expected_reward
    
    # Return state s' sampled from P_h(current_state, a)
    def get_next_state(self, h, a):
        return self.env.step(a)[0]
    # End fn get_next_state
# End class NeuralLinearMDP


# Test
if __name__ == '__main__':
    from argparse import ArgumentParser
    import pickle
    from pathlib import Path
    
    parser = ArgumentParser(prog="linearize_interface")
    
    parser.add_argument("planning_horizon", type=int, help="The planning horizon (H), i.e. the length of each episode.")
    parser.add_argument("-d", "--data-file", default="ctrlsac_agent.dat", help="Input file with representation data (ctrlsac)")
    parser.add_argument("-o", "--output-file", default="linearized_mdp.dat", help="File to store the LinearMDP object.")
    
    args = parser.parse_args()
    print('Arguments:', args)

    data_file_path = Path(args.data_file)
    with data_file_path.open(mode='rb') as data_file_obj:
        data = pickle.load(data_file_obj)
        env_name = data['env_name']
        agent = data['agent']
    print(agent.phi)
    print(agent.mu)
    print(agent.theta)
    
    env = gym.make(env_name)
    
    mdp = NeuralLinearMDP(representation=agent, env=env, H=args.planning_horizon)
    
    output_file_path = Path(args.output_file)
    with output_file_path.open(mode='wb') as out_file_obj:
        # This V_opt helps simplify regret calculation in the learning algorithm since the (approx) optimal V_opt cannot
        # really be computed for non-tabular MDP.
        pickle.dump({'mdp': mdp, 'pi_opt': np.zeros(1), 'V_opt': (mdp.H+1.)*np.ones(shape=(mdp.H,1))}, out_file_obj)
    print('Wrote to file: ', output_file_path)
    print('END')
# End if
