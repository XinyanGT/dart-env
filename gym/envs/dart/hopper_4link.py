import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHopper4LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 100
        self.include_action_in_obs = True
        self.randomize_dynamics = True
        obs_dim = 11

        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        dart_env.DartEnv.__init__(self, 'hopper_multilink/hopperid_4link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.net_modules = []

        self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'revolute_enc'])
        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])

        if not self.include_action_in_obs:
            self.net_modules.append([[4, 10], 1, None])
            self.net_modules.append([[3, 9], 1, [0]])
            self.net_modules.append([[2, 8], 1, [1]])
        else:
            self.net_modules.append([[4, 10, 3], 1, None])
            self.net_modules.append([[3, 9, 12], 1, [0]])
            self.net_modules.append([[2, 8, 11], 1, [1]])
        self.net_modules.append([[0, 1, 5, 6, 7], 0, [2]])

        '''self.net_modules.append([[2, 8], 1, [3]])
        self.net_modules.append([[3, 9], 1, [4]])
        self.net_modules.append([[4, 10], 1, [5]])

        self.net_modules.append([[], 2, [4]])
        self.net_modules.append([[], 2, [5]])
        self.net_modules.append([[], 2, [6]])'''

        self.net_modules.append([[], 2, [3, 2]])
        self.net_modules.append([[], 2, [3, 1]])
        self.net_modules.append([[], 2, [3, 0]])

        self.net_modules.append([[], None, [4, 5, 6], None, False])

        utils.EzPickle.__init__(self)

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        if self.include_action_in_obs:
            self.prev_a = np.copy(clamped_control)
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.bodynode1 == self.robot_skeleton.bodynodes[2] or contact.bodynode2 == \
                    self.robot_skeleton.bodynodes[2]:
                fall_on_ground = True

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        #print(self.num_steps)
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    True)#(height > self.init_height - 0.4) and (height < self.init_height + 0.5) and (abs(ang) < .4))
        if fall_on_ground:
            done = True
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        self.init_height = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        if self.randomize_dynamics:
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(
                    self.bodynode_original_masses[i] + np.random.uniform(-1.5, 1.5))
                self.robot_skeleton.bodynodes[i].set_friction_coeff(
                    self.bodynode_original_frictions[i] + np.random.uniform(-0.5, 0.5))

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5