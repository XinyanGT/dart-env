import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion
from gym import error, spaces

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0]) * 1.0
        self.train_UP = False
        self.noisy_input = True
        self.input_time = False

        self.fallstates = []

        self.terminate_for_not_moving = [0.5, 1.0] # [distance, time], need to mvoe distance in time

        self.action_filtering = 0  # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.action_filter_in_env = False # whether to filter out actions in the environment
        self.action_filter_inobs = False  # whether to add the previous actions to the observations

        obs_dim = 11

        self.test_jump_obstacle = False
        self.input_obs_height = False
        self.resample_task_on_reset = False
        self.vibrating_ground = False
        self.ground_vib_params = [0.14, 1.5]  # magnitude, frequency

        self.periodic_noise = False
        self.periodic_noise_params = [0.1, 4.5]  # magnitude, frequency

        self.learnable_perturbation = False
        self.learnable_perturbation_list = [['h_shin', 80, 0]] # [bodynode name, force magnitude, torque magnitude
        self.learnable_perturbation_space = spaces.Box(np.array([-1] * len(self.learnable_perturbation_list) * 6), np.array([1] * len(self.learnable_perturbation_list) * 6))
        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        self.alive_bonus = 1.0
        self.velrew_weight = 1.0
        self.UP_noise_level = 0.0
        self.resample_MP = True  # whether to resample the model paraeters

        self.actuator_nonlinearity = False
        self.actuator_nonlin_coef = 1.0

        self.param_manager = hopperContactMassManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        if self.action_filtering > 0 and self.action_filter_inobs:
            obs_dim += len(self.action_scale) * self.action_filtering

        if self.test_jump_obstacle:
            obs_dim += 1
            if self.input_obs_height:
                obs_dim += 1
                self.obs_height = 0.0

        self.t = 0

        self.action_buffer = []

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        if self.input_time:
            obs_dim += 1

        self.action_bound_model = None

        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel', 'hopper_box.skel', 'hopper_ellipsoid.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]
        self.initial_coms = [np.copy(bn.com()) for bn in self.robot_skeleton.bodynodes]

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)
        self.dart_worlds[1].set_collision_detector(2)
        self.dart_worlds[2].set_collision_detector(1)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        self.cycle_times = []  # gait cycle times
        self.previous_contact = None

        self.param_manager.set_simulator_parameters(self.current_param)

        self.height_threshold_low = 0.56 * self.robot_skeleton.bodynodes[2].com()[1]
        self.rot_threshold = 0.4

        self.short_perturb_params = []#[1.0, 1.3, np.array([-200, 0, 0])] # start time, end time, force

        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()
        self.active_param = self.param_manager.activated_param

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.cur_step = 0


        self.randomize_initial_state = True
        self.stop_velocity_reward = 1000.0
        self.height_penalty = 0.0
        self.obstacle_x_offset = 2.0
        if self.test_jump_obstacle:
            self.velrew_weight = 1.0
            self.stop_velocity_reward = 40.0 # stop giving velocity reward after travelling 40 meters
            self.randomize_initial_state = False
            self.height_penalty = 0.25  # penalize torso to be too high
            self.noisy_input = False

        #t = self.resample_task()
        #new_t = (t[0], t[1], t[2], 1.3)
        #self.set_task(new_t)
        #print("Resampled task: ", new_t)


        utils.EzPickle.__init__(self)

    def resample_task(self):
        world_selection = 1#np.random.randint(len(self.dart_worlds))
        self.dart_world = self.dart_worlds[world_selection]
        self.robot_skeleton = self.dart_world.skeletons[-1]
 
        self.resample_MP = False

        #self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()
        #self.velrew_weight = np.sign(np.random.randn(1))[0]

        #obstacle_height = np.random.choice(np.arange(0, 1.01, 0.01))
        obstacle_height = -1#np.random.random() * 1.0
        self.obs_height = obstacle_height

        for body in self.dart_world.skeletons[1].bodynodes:
            for shapenode in body.shapenodes:
                shapenode.set_offset([self.obstacle_x_offset, obstacle_height, 0])

        return np.array(self.current_param), self.velrew_weight, world_selection, obstacle_height

    def set_task(self, task_params):
        self.dart_world = self.dart_worlds[task_params[2]]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.param_manager.set_simulator_parameters(task_params[0])
        self.velrew_weight = task_params[1]

        self.dart_world.skeletons[1].bodynodes[0].shapenodes[0].set_offset([self.obstacle_x_offset, task_params[3], 0])
        self.dart_world.skeletons[1].bodynodes[0].shapenodes[1].set_offset([self.obstacle_x_offset, task_params[3], 0])
        self.obs_height = task_params[3]

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if len(self.short_perturb_params) > 0:
                if self.cur_step * self.dt > self.short_perturb_params[0] and\
                    self.cur_step * self.dt < self.short_perturb_params[1]:
                    self.robot_skeleton.bodynodes[2].add_ext_force(self.short_perturb_params[2])

            if self.learnable_perturbation: # if learn to perturb
                for bid, pert_param in enumerate(self.learnable_perturbation_list):
                    force_dir = self.learnable_perturbation_act[bid * 6: bid * 6 + 3]
                    torque_dir = self.learnable_perturbation_act[bid * 6 + 3: bid * 6 + 6]
                    if np.all(force_dir == 0):
                        pert_force = np.zeros(3)
                    else:
                        pert_force = pert_param[1] * force_dir / np.linalg.norm(force_dir)
                    if np.all(torque_dir == 0):
                        pert_torque = np.zeros(3)
                    else:
                        pert_torque = pert_param[2] * torque_dir / np.linalg.norm(torque_dir)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_force(pert_force)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_torque(pert_torque)


            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        if self.actuator_nonlinearity:
            a = np.tanh(self.actuator_nonlin_coef * a)
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay-1]


        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale


        self.do_simulation(tau, self.frame_skip)


    def about_to_contact(self):
        return False

    def post_advance(self):
        self.dart_world.check_collision()

    def terminated(self):
        '''if self.cur_step * self.dt > self.short_perturb_params[0] and \
                self.cur_step * self.dt < self.short_perturb_params[1] + 2: # allow 2 seconds to recover
            self.height_threshold_low = 0.0
            self.rot_threshold = 10
        else:
            self.height_threshold_low = 0.56 * self.initial_coms[2][1]
            self.rot_threshold = 0.4'''

        self.fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        permitted_contact_bodies = [self.robot_skeleton.bodynodes[-1], self.robot_skeleton.bodynodes[-2]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 not in permitted_contact_bodies and contact.bodynode2 not in permitted_contact_bodies:
                self.fall_on_ground = True

        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        ang = self.robot_skeleton.q[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all()\
            #and not self.fall_on_ground)
            and (height > self.height_threshold_low) and (abs(ang) < self.rot_threshold))

        if self.terminate_for_not_moving is not None:
            if self.t > self.terminate_for_not_moving[1] and \
                    (np.abs(self.robot_skeleton.q[0]) < self.terminate_for_not_moving[0] or
                     self.robot_skeleton.q[0] * self.velrew_weight < 0):
                done = True

        return done

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        if posafter > self.stop_velocity_reward:
            reward = 0
        reward += self.alive_bonus * step_skip
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        reward -= np.abs(self.robot_skeleton.q[2])
        reward -= self.height_penalty * np.clip(self.robot_skeleton.q[1] - 1.3, 0.0, 1e10)
        return reward

    def step(self, a):
        self.action_filter_cache.append(a)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0 and self.action_filter_in_env:
            a = np.mean(self.action_filter_cache, axis=0)

        self.action_buffer.append(np.copy(a))

        if self.vibrating_ground:
            self.dart_world.skeletons[0].joints[0].set_rest_position(0, self.ground_vib_params[0] * np.sin(2*np.pi*self.ground_vib_params[1] * self.cur_step * self.dt))

        if self.action_bound_model is not None:
            pred_bound = self.action_bound_model.predict(self._get_obs(False))[0]
            in_a = np.copy(a)
            up_bound = pred_bound[::2]
            low_bound = pred_bound[1::2]
            mid = 0.5 * (up_bound + low_bound)
            up_bound[up_bound - low_bound < 0.05] = mid[up_bound - low_bound < 0.05] + 0.05
            low_bound[up_bound - low_bound < 0.05] = mid[up_bound - low_bound < 0.05] - 0.05
            a = in_a * (up_bound - low_bound) + low_bound

        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        done = self.terminated()
        if done:
            reward = 0
        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if self.robot_skeleton.bodynode('h_foot') in [contact.bodynode1, contact.bodynode2] \
                    and \
                    self.dart_world.skeletons[0].bodynodes[0] in [contact.bodynode1, contact.bodynode2]:
                if self.previous_contact is None:
                    self.previous_contact = self.cur_step * self.dt
                else:
                    self.cycle_times.append(self.cur_step * self.dt - self.previous_contact)
                    self.previous_contact = self.cur_step * self.dt
        self.gait_freq = 0

        envinfo['gait_frequency'] = self.gait_freq
        envinfo['xdistance'] = self.robot_skeleton.q[0]
        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer = True):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.action_filtering > 0 and self.action_filter_inobs:
            state = np.concatenate([state] + self.action_filter_cache)

        if self.test_jump_obstacle:
            state = np.concatenate([state, [self.robot_skeleton.q[0] - 3.5]])
            if self.input_obs_height:
                state = np.concatenate([state, [self.obs_height]])

        if self.train_UP:
            UP = self.param_manager.get_simulator_parameters()
            if self.UP_noise_level > 0:
                UP += np.random.uniform(-self.UP_noise_level, self.UP_noise_level, len(UP))
                UP = np.clip(UP, -0.05, 1.05)
            state = np.concatenate([state, UP])

        if self.input_time:
            state = np.concatenate([state, [self.t]])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay-1-i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0]*0.0])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1-i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0]*len(self.control_bounds[0])])

        if self.periodic_noise:
            final_obs += np.random.randn(len(final_obs)) * self.periodic_noise_params[0] * (np.sin(2*np.pi*self.periodic_noise_params[1] * self.cur_step * self.dt) + 1)

        return final_obs

    def reset_model(self):
        if self.resample_task_on_reset:
           self.resample_task()
        for world in self.dart_worlds:
            world.reset()
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs) if self.randomize_initial_state else np.zeros(self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs) if self.randomize_initial_state else np.zeros(self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        self.action_filter_cache = []
        if self.action_filtering > 0:
            for i in range(self.action_filtering):
                self.action_filter_cache.append(np.zeros(len(self.action_scale)))

        state = self._get_obs(update_buffer = True)

        self.action_buffer = []

        self.cur_step = 0

        self.t = 0

        self.fall_on_ground = False

        self.cycle_times = []  # gait cycle times
        self.previous_contact = None

        if self.vibrating_ground:
            self.ground_vib_params[0] = np.random.random() * 0.14

        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        #self.short_perturb_params = [np.random.uniform(0.9, 1.2), np.random.uniform(1.2, 1.4), np.array([200*-1, 0, 0])]  # start time, end time, force

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        s[1] += self.zeroed_height
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        snew[1] -= self.zeroed_height
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]

    def set_sim_parameters(self, pm):
        self.param_manager.set_simulator_parameters(pm)

    def get_sim_parameters(self):
        return self.param_manager.get_simulator_parameters()

