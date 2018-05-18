import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartCartPole4PoleEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 10
        obs_dim = 8
        self.include_action_in_obs = True
        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        dart_env.DartEnv.__init__(self, 'cartpole_multilink/cartpole_4pole.skel', 2, obs_dim, self.control_bounds,
                                  dt=0.02, disableViewer=True)

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.net_modules = []

        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'revolute_enc'])
        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])

        if not self.include_action_in_obs:
            self.net_modules.append([[3, 7], 0, None])
            self.net_modules.append([[2, 6], 0, [0]])
            self.net_modules.append([[1, 5], 0, [1]])
            self.net_modules.append([[0, 4], 0, [2]])
        else:
            self.net_modules.append([[3, 7, 11], 0, None])
            self.net_modules.append([[2, 6, 10], 0, [0]])
            self.net_modules.append([[1, 5, 9], 0, [1]])
            self.net_modules.append([[0, 4, 8], 0, [2]])

        self.net_modules.append([[], 1, [3]])
        self.net_modules.append([[], 1, [3, 2]])
        self.net_modules.append([[], 1, [3, 1]])
        self.net_modules.append([[], 1, [3, 0]])

        self.net_modules.append([[], None, [4, 5, 6, 7], None, False])

        utils.EzPickle.__init__(self)

    def _step(self, a):
        a = np.clip(a, -1, 1)
        if self.include_action_in_obs:
            self.prev_a = np.copy(a)

        tau = a * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        #reward = -np.abs(self.robot_skeleton.bodynodes[-1].C[1] - 2.1) + 2.1
        reward = -np.linalg.norm(self.robot_skeleton.bodynodes[-1].C - np.array([0.6, 0.6, 0.0])) + 1.75

        reward -= 0.04 * np.square(a).sum()

        ob = self._get_obs()

        notdone = np.isfinite(ob).all() and (np.abs(self.state_vector()) < 50).all()
        done = not notdone
        return ob, reward, done, {}


    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q % (2*np.pi), self.robot_skeleton.dq]).ravel()
        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])
        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if np.random.random() < 0.5:
            qpos[0] += np.pi
        else:
            qpos[0] -= np.pi
        self.set_state(qpos, qvel)

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
