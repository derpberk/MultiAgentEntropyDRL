from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

nav = np.genfromtxt('/home/azken/Samuel/MultiAgentEntropyDRL/Environment/example_map.csv', delimiter=',')

n_agents = 4
init_pos = np.array([[22,24],
                     [16,16],
                     [10,13],
                     [40,28],
                     [31,11],
                     [33,26],
                     [47,30],
                     [39,18],
                     [22,6],
                     [6,17]])

init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             movement_length=1,
                             distance_budget=100,
                             random_initial_positions=False,
                             initial_meas_locs=None)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E5),
                                       batch_size=64,
                                       target_update=1,
                                       soft_update=True,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=0,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=True,
                                       safe_actions=False)


def mean_parameters(model):

    weight_mu = [np.mean(np.abs(param.data.detach().cpu().numpy())) for name, param in model.named_parameters() if 'weight_mu' in name]
    weight_sigma = [np.mean(np.abs(param.data.detach().cpu().numpy())) for name, param in model.named_parameters() if 'weight_sigma' in name]
    bias_mu = [np.mean(np.abs(param.data.detach().cpu().numpy())) for name, param in model.named_parameters() if 'bias_mu' in name]
    bias_sigma = [np.mean(np.abs(param.data.detach().cpu().numpy())) for name, param in model.named_parameters() if 'bias_sigma' in name]

    return [np.mean(weight_mu), np.mean(weight_sigma), np.mean(bias_mu), np.mean(bias_sigma)]

def mean_parameters_per_layer(model):

    layers = {}
    params_names = ['weight_mu', 'weight_sigma', 'bias_mu', 'bias_sigma']

    for name, param in model.named_parameters():

        if not any([params_name in name for params_name in params_names]):
            continue

        name = name.split('.')
        layer_name = name[0]
        weight_name = name[1]

        if not layer_name in layers:
            layers[layer_name] = {'weight_mu': 0, 'weight_sigma': 0, 'bias_mu':0, 'bias_sigma':0}

        dict_weight_name = [param_name for param_name in params_names if weight_name in param_name][0]

        layers[layer_name][dict_weight_name] = np.mean(np.abs(param.data.detach().cpu().numpy()))

    return layers

def load_and_process_values(path):

    multiagent.load_model(path)
    return mean_parameters_per_layer(multiagent.dqn)


episode_nums = np.arange(1,10) * 5000

path = '/home/azken/Samuel/MultiAgentEntropyDRL/Learning/runs/Noisy_Decoupled_Reward/Episode_{}_Policy.pth'

vals = [load_and_process_values(path.format(num)) for num in episode_nums]

x = list(vals[0].keys())

y = [list(vals[0]['common_layer_1'].values()) for ]