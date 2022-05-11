import gym
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel as W, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error as MSE
from GroundTruths import ShekelGT
from scipy.spatial.distance import cdist
from OilSpillEnvironment import OilSpillEnv


class DiscreteVehicle:

    def __init__(self, initial_position, n_actions, movement_length, navigation_map):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        # Static navigation map #
        self.navigation_map = navigation_map
        # Binary map for the agent positioning #
        self.position_map = np.copy(self.navigation_map)
        # Travelled distance #
        self.distance = 0.0
        self.num_of_collisions = 0
        self.action_space = gym.spaces.Discrete(n_actions)
        self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
        self.movement_length = movement_length

    def move(self, action):

        self.distance += self.movement_length
        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])

        next_position = np.clip(self.position + movement, (0, 0), np.array(self.navigation_map.shape) - 1)

        if self.check_collision(next_position):
            collide = True
            self.num_of_collisions += 1
        else:
            collide = False
            self.position = next_position
            self.waypoints = np.vstack((self.waypoints, [self.position]))

        return collide

    def check_collision(self, next_position):

        if any(next_position > np.array(self.navigation_map.shape) - 1) or any(next_position < 0):
            return True

        elif self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
            return True

        return False

    def reset(self, initial_position):

        # Reset the initial positions and compute the new position maps #
        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.position_map = np.copy(self.navigation_map)
        self.position_map[self.position[0], self.position[1]] = 1.0
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.distance = 0.0
        self.num_of_collisions = 0

    def check_action(self, action):

        angle = self.angle_set[action]
        movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
        next_position = self.position + movement

        return self.check_collision(next_position)

    def move_to_position(self, goal_position):

        """ Add the distance """
        assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
        self.distance += np.linalg.norm(goal_position - self.position)
        """ Update the position """
        self.position = goal_position


class DiscreteFleet:

    def __init__(self, number_of_vehicles, n_actions, initial_positions, movement_length, navigation_map):

        self.number_of_vehicles = number_of_vehicles
        self.initial_positions = initial_positions
        self.n_actions = n_actions
        self.movement_length = movement_length
        self.vehicles = [DiscreteVehicle(initial_position=initial_positions[k],
                                         n_actions=n_actions,
                                         movement_length=movement_length,
                                         navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def move(self, fleet_actions):

        collision_array = [self.vehicles[k].move(fleet_actions[k]) for k in range(self.number_of_vehicles)]

        self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

        return collision_array

    def measure(self, gt, extra_positions=None):

        """
        Take a measurement in the given N positions
        :param gt_field:
        :return: An numpy array with dims (N,2)
        """

        if extra_positions is not None:
            positions = extra_positions
        else:
            positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

        values = []
        for pos in positions:
            values.append(gt.evaluate(pos))

        if self.measured_locations is None:
            self.measured_locations = positions
            self.measured_values = values
        else:
            self.measured_locations = np.vstack((self.measured_locations, positions))
            self.measured_values = np.hstack((self.measured_values, values))

        """
        # Check only non redundant values #
        non_redundant_locs, non_redundant_idxs = np.unique(self.measured_locations, axis=0, return_index=True)
        redundant_flags = len(non_redundant_locs) != len(self.measured_locations)
        self.measured_values = np.asarray(self.measured_values)[non_redundant_idxs]
        self.measured_locations = non_redundant_locs

        return self.measured_values, self.measured_locations, redundant_flags
        """
        return self.measured_values, self.measured_locations

    def reset(self, initial_positions):

        for k in range(self.number_of_vehicles):
            self.vehicles[k].reset(initial_position=initial_positions[k])

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def get_distances(self):

        return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

    def get_positions(self):

        positions = [self.vehicles[i].position for i in range(self.number_of_vehicles)]

        return np.asarray(positions)

    def get_agent_position(self, i):
        return self.vehicles[i].position

    def get_distance_matrix(self):
        """ The matrix which indicates the distance between an agent i and other j """
        positions = self.get_positions()
        d_matrix = cdist(positions, positions)
        return d_matrix

    def check_collisions(self, test_actions):

        return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

    def move_fleet_to_positions(self, goal_list):
        """ Move the fleet to the given positions.
         All goal positions must ve valid. """

        goal_list = np.atleast_2d(goal_list)

        for k in range(self.number_of_vehicles):
            self.vehicles[k].move_to_position(goal_position=goal_list[k])


class UncertaintyReductionMA(gym.Env):

    def __init__(self,
                 navigation_map,
                 number_of_agents,
                 initial_positions,
                 movement_length,
                 distance_budget,
                 initial_meas_locs,
                 number_of_actions=8,
                 max_number_of_collisions=5,
                 random_initial_positions=False,
                 static_scenario=True,
                 lengthscale=0.75):

        # Navigation map #
        self.navigation_map = navigation_map
        # All postions that are visitable #
        self.visitable_positions = np.column_stack(np.where(navigation_map == 1)).astype(float)
        self.number_of_agents = number_of_agents
        # Deploy positions #
        self.initial_positions = initial_positions
        # Whether to use random initial positions #
        self.random_initial_positions = random_initial_positions
        self.initial_meas_locs = initial_meas_locs
        self.distance_budget = distance_budget
        # Max number of collisions permitted before the episodes ends #
        self.max_number_of_collisions = max_number_of_collisions
        # Whether to consider the values or not #

        # Gym environment parameters #
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4, self.navigation_map.shape[0], self.navigation_map.shape[1]))

        self.action_space = gym.spaces.Discrete(number_of_actions)

        self.fleet = DiscreteFleet(number_of_vehicles=number_of_agents,
                                   n_actions=number_of_actions,
                                   initial_positions=initial_positions,
                                   movement_length=movement_length,
                                   navigation_map=navigation_map)

        self.state = None
        self.measured_locations = None
        self.measured_values = None
        self.uncertainty = None
        self._uncertainty = None
        self.uncertainty_initial = None
        self.fig = None
        self.mse = None
        self._mse = None
        self.evaluation_mode = False

        """
        # Gaussian Process parameters and objects #
        self.kernel = C(1.0, constant_value_bounds='fixed') * Matern(length_scale=2, length_scale_bounds='fixed') + W(noise_level=1E-3)
        self.GPR = GaussianProcessRegressor(kernel=self.kernel, optimizer=None)
        """

        self.lengthscale = lengthscale
        self.noise_level = 0.25
        self.kernel = Matern(length_scale=self.lengthscale, length_scale_bounds='fixed')

        # Benchmark #
        self.benchmark = OilSpillEnv(self.navigation_map, dt=0.5, flow=10, gamma=1, kc=1, kw=1)
        self.static_scenario = static_scenario

        # Reward function parameters #
        self.distance_penalization_weight = 1 / self.number_of_agents
        # In order: uncertainty reduction, collision, distance #
        self.r_lambda = [1.0, -1.0, -1.0]

        self.distance_threshold = [2 * movement_length, 10 * movement_length]
        self.distance_threshold_clipping = [1.0, 0.0]
        self.return_individual_rewards = False

    def eval(self):
        print("Eval mode activated")
        self.evaluation_mode = True

    def train(self):
        print("Train mode activated")
        self.evaluation_mode = False

    def valid_action(self, a):
        assert self.number_of_agents == 1, "Not implemented for Multi-Agent!"

        # Return the action valid flag #
        return not self.fleet.check_collisions([a])[0]

    @staticmethod
    def minmax_normalization(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    def get_metrics(self):
        """ Compute the different metrics and return a dictionary """
        # Compute the uncertainty #
        mean_uncertainty = np.mean(self.uncertainty)
        # Compute the max error #
        # Compute mean distance #
        distance_matrix = self.fleet.get_distance_matrix()
        mean_distance = np.sum(distance_matrix) / (self.number_of_agents - 1)
        # Individual agent distance #
        individual_distance = np.sum(distance_matrix, axis=1)

        return {'uncertainty': mean_uncertainty,'mean_distance': mean_distance, "individual_distance": individual_distance}

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Process movements #
        collision_array = self.fleet.move(action)

        # Take measurements in the current positions #
        if self.evaluation_mode:
            self.measured_values, self.measured_locations = self.fleet.measure(gt=self.benchmark)
        # Update the model
        self.update_model()
        # Compute the new reward #
        reward = self.reward_function(collision_array, self.return_individual_rewards)

        # Render the new state #
        self.state = self.render_state()
        # Terminal condition verification #
        done = np.mean(
            self.fleet.get_distances()) > self.distance_budget or self.fleet.fleet_collisions > self.max_number_of_collisions

        if not self.static_scenario:
            self.benchmark.step()

        return self.state, reward, done, {}

    """
    def update_model(self):

        # Predict the new model #

        non_redundant_locs, non_redundant_idxs = np.unique(self.measured_locations, axis=0, return_index=True)
        non_redundant_measured_values = np.asarray(self.measured_values)[non_redundant_idxs]

        self.GPR.fit(non_redundant_locs, non_redundant_measured_values)
        mu, std = self.GPR.predict(self.visitable_positions, return_std=True)

        # Reshape the mean map #
        self.mu = np.zeros_like(self.navigation_map)
        self.mu[self.visitable_positions[:, 0].astype(int), self.visitable_positions[:, 1].astype(int)] = mu
        # Compute the new uncertainty #
        self._uncertainty = np.copy(self.uncertainty)
        self.uncertainty = np.zeros_like(self.navigation_map)
        self.uncertainty[self.visitable_positions[:, 0].astype(int), self.visitable_positions[:, 1].astype(int)] = std

        # Get the normalized MSE #
        self.mse = np.mean(np.abs(self.true_map_normalized[
                                      self.visitable_positions[:, 0].astype(int), self.visitable_positions[:, 1].astype(
                                          int)] -
                                  self.minmax_normalization(self.mu[self.visitable_positions[:, 0].astype(
                                      int), self.visitable_positions[:, 1].astype(int)])))
    """

    def update_model(self):
        """ Update the uncertainty and the model if necesary"""

        # Compute the new uncertainty substraction mask #
        uncertainty_mask = np.zeros_like(self.navigation_map)
        for i in range(self.fleet.number_of_vehicles):
            uncertainty_values = (1 - self.noise_level) * self.kernel(self.fleet.get_agent_position(i), self.visitable_positions)
            uncertainty_mask[np.where(self.navigation_map == 1)] += uncertainty_values.flatten()
        # Update the uncertainty #
        self._uncertainty = np.copy(self.uncertainty)
        self.uncertainty = np.clip(self.uncertainty - uncertainty_mask, 0, 1)

    def reward_function(self, collition_array, return_individual_rewards=False):
        """ The reward is a sum of various components:
            1) The total uncertainty reduction
            2) A function of the distance between one agent and the other ones of the fleet
            3) A function of the collisions
            4) A function of the uncertainty reduction
            5) A function of the maximum regret

            R = r_lambda_1 * uncertainty_reduction + r_lambda_2 * distance_between_agents + r_lambda_3 * collisions + r_lambda_4 * regret

        """

        # Compute the collective reward #
        uncertainty_reduction = np.sum(self._uncertainty - self.uncertainty) / self.number_of_agents

        # Compute the individual reward vector #
        distance_matrix = self.fleet.get_distance_matrix()
        distance_between_agents = self.clipped_linear_function(distance_matrix, *self.distance_threshold,
                                                               *self.distance_threshold_clipping)
        distance_between_agents = np.sum(distance_between_agents, axis=1) - 1.0

        # Compute the individual reward vector #
        reward = self.r_lambda[0] * uncertainty_reduction + self.r_lambda[1] * np.asarray(collition_array).astype(int) + \
                 self.r_lambda[2] * distance_between_agents

        if return_individual_rewards:
            return reward, uncertainty_reduction, distance_between_agents, collition_array
        else:
            return reward

    @staticmethod
    def clipped_linear_function(x, max_x, min_x, max_y, min_y):
        """ Clipped linear function given its max_x, min_x, max_y, min_y """
        return np.clip(min_y + (x - min_x) * (max_y - min_y) / (max_x - min_x), min_y, max_y)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        # Reset the positions #
        if self.random_initial_positions:
            random_initial_indx = np.random.choice(np.arange(0, len(self.visitable_positions)), replace=False,
                                                   size=self.number_of_agents)
            self.fleet.reset(initial_positions=self.visitable_positions[random_initial_indx].astype(int))
        else:
            self.fleet.reset(initial_positions=self.initial_positions)

        # self.gt.reset()
        if self.evaluation_mode:
            self.benchmark.reset()
            self.benchmark.update_to_time(50)

            # Take measurements
            self.measured_values, self.measured_locations = self.fleet.measure(gt=self.benchmark)

        if self.initial_meas_locs is not None:
            self.fleet.measure(gt=self.benchmark, extra_positions=self.initial_meas_locs)

        # Reset the uncertainty and update the model #
        self.uncertainty = np.copy(self.navigation_map).astype(float)
        self.update_model()

        self.uncertainty_initial = np.sum(self.uncertainty)

        self.state = self.render_state()

        return self.state

    def render_state(self):
        """ The state is formed by:
            1) The static obstacles map
            2) The uncertainty map
            3) The current position map
            4) The other agents positions
            5) The model (if it is specified)
        """

        # The static obstacles map #
        nav_map = np.copy(self.navigation_map)

        # The individual agent positions #
        positions_map = np.zeros(shape=(self.number_of_agents, self.navigation_map.shape[0], self.navigation_map.shape[1]))
        for j in range(self.number_of_agents):
            positions_map[j, self.fleet.vehicles[j].position[0].astype(int), self.fleet.vehicles[j].position[1].astype(int)] = 1.0

        return np.concatenate((nav_map[np.newaxis], self.uncertainty[np.newaxis], positions_map))

    def render(self, mode='human', pauseint=0.1):

        import matplotlib.pyplot as plt

        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """

        if self.fig is None:
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 5)
            self.d0 = self.axs[0].imshow(self.state[0], cmap='gray')
            self.axs[0].set_title('Nav. Map')
            self.d1 = self.axs[1].imshow(self.state[1], cmap='gray', vmin=0, vmax=1, interpolation=None)
            self.axs[1].set_title('Uncertainty')
            positions = self.fleet.get_positions()
            positions[:, [0, 1]] = positions[:, [1, 0]]
            self.d2 = self.axs[2].imshow(np.sum(self.state[3:], axis=0), cmap='gray')
            self.d2_1 = self.axs[2].scatter(positions[:, 0], positions[:, 1], c=plt.get_cmap('tab10')(np.arange(self.number_of_agents, dtype=int)))
            self.axs[2].set_title('Other positions')
            self.d3 = self.axs[3].imshow(self.state[2], cmap='jet')
            self.axs[3].set_title('Real Field')

            if self.evaluation_mode:
                self.d4 = self.axs[4].imshow(self.benchmark.density, cmap='jet', vmin=0, vmax=np.max(self.benchmark.density), interpolation='bilinear')
            else:
                self.d4 = self.axs[4].imshow(np.zeros_like(self.navigation_map))

            self.axs[4].set_title('Benchmark Field')

        else:

            self.d0.set_data(self.state[0])
            self.d1.set_data(self.state[1])
            self.d2.set_data(np.sum(self.state[3:], axis=0))
            positions = self.fleet.get_positions()
            positions[:, [0, 1]] = positions[:, [1, 0]]
            self.d2_1.set_offsets(positions)
            self.d3.set_data(self.state[2])
            if self.evaluation_mode:
                self.d4.set_data(self.benchmark.density)

        self.fig.canvas.draw()
        plt.pause(pauseint)

    def seed(self, seed=None):

        np.random.seed(seed)

        return

    def individual_agent_observation(self, state=None, agent_num=0):

        if state is None:
            state = self.state
        assert 0 <= agent_num <= self.number_of_agents - 1, "Not enough agents for this observation request. "

        index = [0, 1, 2 + agent_num]
        pointer = 2

        common_states = state[index]

        other_agents_positions_state = np.sum(
            state[np.delete(np.arange(pointer, pointer + self.number_of_agents), agent_num), :, :], axis=0)

        return np.concatenate((common_states, other_agents_positions_state[np.newaxis]), axis=0)


if __name__ == '__main__':

    import time

    nav = np.genfromtxt('./example_map.csv', delimiter=',')
    n_agents = 4
    init_pos = np.array([[66, 74], [50, 50], [60, 50], [65, 50]]) / 3
    init_pos = init_pos.astype(int)

    env = UncertaintyReductionMA(navigation_map=nav, number_of_agents=n_agents, random_initial_positions=True,
                                 initial_positions=init_pos, movement_length=1, distance_budget=200,
                                 initial_meas_locs=None)

    env.seed(20)

    T = 10
    t0 = time.time()

    U = []
    D = []

    for t in range(T):

        print("Run ", t)
        s = env.reset()
        env.render()
        d = False

        while not d:

            action = np.random.randint(0, 8, n_agents)

            while any(env.fleet.check_collisions(action)):
                action = np.random.randint(0, 8, n_agents)

            s, r, d, _ = env.step(action)
            print("Reward: ", r)
            env.render()

    print("Tiempo medio por iteracion: ", (time.time() - t0) / T)
