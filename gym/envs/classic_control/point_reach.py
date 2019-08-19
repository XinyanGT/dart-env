import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PointReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low = np.array([-1, -1]), high = np.array([1, 1]))
        self.dt = 0.3

        self.mass = 1.5
        self.mass_range = [0.2, 10.0]

        self.wind = np.array([0.0, 0.0])

        self.input_target = False
        self.train_UP = False
        self.resample_MP = False

        if not self.train_UP:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf]), np.array([10, 10, np.inf, np.inf]))
        else:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf, -np.inf]),
                                                np.array([10, 10, np.inf, np.inf, np.inf]))

        self.targets = [np.array([0.0, 0.0])]

        if self.input_target:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf, -10, -10]), np.array([10, 10, np.inf, np.inf, 10, 10]))

        self._seed()
        self.viewer = None
        self.state = None

    def resample_task(self):
        # A point on half circle above
        radius = np.random.uniform(6.8, 7.0)
        # angle = np.random.uniform(0.0, np.pi)
        angle = np.random.choice(np.arange(0, np.pi+0.01, np.pi / 10.0))
        # if np.random.random() < 0.05:
        #     angle = - np.pi/2.0
        self.targets = [np.array([np.sin(angle) * radius, np.cos(angle) * radius])]

        return [np.copy(self.targets)]

    def set_task(self, task_params):
        self.targets = np.copy(task_params[0])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action.clip(-1, 1)

        self.state[2:] += action * self.dt / self.mass + self.wind
        self.state[0:2] += self.state[2:] * self.dt
        self.state = self.state.clip(-10, 10)

        reward = -0.05 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0]) - np.sum(np.abs(action)) * 5.0
        if np.linalg.norm(self.state[0:2] - self.targets[0]) < 0.8:
            reward += 25 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])

        done = False

        self.current_action = np.copy(action)

        obs = np.array(self.state)

        if self.train_UP:
            obs = np.concatenate([obs, [0.0]])

        if self.input_target:
            obs = np.concatenate([obs, self.targets[0] / 10.0])

        return obs, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[0] = self.np_random.uniform(-0.5, 0.5)
        self.state[1] = self.np_random.uniform(-0.5, 0.5)


        self.current_action = np.ones(2)

        if self.resample_MP:
            self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1])

        if self.train_UP:
            #obs = np.concatenate([self.state, [self.mass]])
            obs = np.concatenate([self.state, [1.5]])
        else:
            obs = np.copy(self.state)

        if self.input_target:
            obs = np.concatenate([obs, self.targets[0] / 10.0])

        return obs

    def _get_obs(self):
        return np.copy(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 500
        screen_height = 500

        world_width = 20.0
        scale = screen_width/world_width
        offset = np.array([screen_height / 2.0, screen_width / 2.0])

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(radius=0.2*scale)
            self.agent_transform = rendering.Transform(self.state[0:2] * scale + offset)
            agent.add_attr(self.agent_transform)
            agent.set_color(0.5, 0.5, 1.0)
            self.viewer.add_geom(agent)

            target1 = rendering.make_circle(radius=0.2*scale)
            self.target_transform = rendering.Transform(self.targets[0] * scale + offset)
            target1.add_attr(self.target_transform)
            target1.set_color(0.5, 1.0, 0.5)
            self.viewer.add_geom(target1)

        if self.state is None: return None

        new_pos = self.state[0:2] * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])

        new_target = self.targets[0] * scale + offset
        self.target_transform.set_translation(new_target[0], new_target[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def state_vector(self):
        return np.copy(self.state)

    def set_state_vector(self, s):
        self.state = np.copy(s)