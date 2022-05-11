from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.MultiAgentEnvironment import UncertaintyReductionMA
import numpy as np
import matplotlib.pyplot as plt

nav = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
n_agents = 4
init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]])/3
init_pos = init_pos.astype(int)

env = UncertaintyReductionMA(navigation_map=nav,
                             number_of_agents=n_agents,
                             initial_positions=init_pos,
                             movement_length=1,
                             distance_budget=100,
                             random_initial_positions=True,
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

env.return_individual_rewards = True

multiagent.load_model('/home/azken/Samuel/MultiAgentEntropyDRL/Learning/runs/May11_14-09-47_M3009R21854/BestPolicy.pth')

multiagent.epsilon = 0


done = False
s = env.reset()

#env.render()
R = []
Unc = []
Dist = []
Colls = []
Regr = []



while not done:

    if multiagent.noisy:
        multiagent.dqn.reset_noise()

    a = multiagent.select_action(s)
    s,r,done,i = env.step(a)
    #print(env.get_metrics())
    #env.render()
    R.append(r[0])
    Unc.append(r[1])
    Dist.append(r[2])
    Colls.append(r[3])

    env.render(pauseint=0.001)

plt.show(block=True)

fig, axs = plt.subplots(4, 1, sharex=True)

axs[0].plot(np.cumsum(R, axis=0))
axs[0].set_title('Reward')
axs[0].legend(['Agent {}'.format(i) for i in range(n_agents)])
axs[1].plot(np.asarray(Unc))
axs[1].set_title('Uncertainty')
axs[2].plot(np.asarray(Dist))
axs[2].set_title('Distance')
axs[3].plot(np.asarray(Colls))
axs[3].set_title('Collisions')
plt.show(block=True)

print(np.sum(R)/4)
