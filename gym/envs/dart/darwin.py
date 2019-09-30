__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from scipy.optimize import minimize
from pydart2.collision_result import CollisionResult
from pydart2.bodynode import BodyNode
import pydart2.pydart2_api as papi
import random
from random import randrange
import pickle
import copy, os, time
from gym.envs.dart.dc_motor import DCMotor

from gym.envs.dart.darwin_utils import *
from gym.envs.dart.parameter_managers import *

from pydart2.utils.transformations import euler_from_matrix, quaternion_from_matrix

class DartDarwinEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.debug_env = False

        obs_dim = 40

        self.trans_euler_root = False

        self.root_input = True    # whether to include root dofs in the obs
        self.include_heading = True
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
        self.fallstate_input = False

        self.action_filtering = 5 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []

        self.obs_cache = []
        self.multipos_obs = 2 # give multiple steps of position info instead of pos + vel
        if self.multipos_obs > 0:
            obs_dim = 20 * self.multipos_obs

        self.mass_ratio = 1.0
        self.kp_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.kd_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.use_discrete_action = False

        self.use_sysid_model = True

        if self.use_sysid_model:
            self.param_manager = darwinParamManager(self)
            # self.param_manager.activated_param.remove(self.param_manager.NEURAL_MOTOR)
        else:
            self.param_manager = darwinSquatParamManager(self)

        self.use_delta_action = 0.0    # if > 0, use delta action from the current state as the next target
        self.use_DCMotor = False
        self.use_spd = False
        self.NN_motor = False
        self.NN_motor_hid_size = 5  # assume one layer
        self.NN_motor_parameters = [np.random.random((2, self.NN_motor_hid_size)),
                                    np.random.random(self.NN_motor_hid_size),
                                    np.random.random((self.NN_motor_hid_size, 2)), np.random.random(2)]
        self.NN_motor_bound = [[200.0, 1.0], [0.0, 0.0]]

        self.train_UP = False
        self.noisy_input = True
        self.resample_MP = False
        self.range_robust = 0.25
        self.randomize_timestep = True
        self.randomize_obstacle = True
        self.randomize_gyro_bias = False
        self.gyro_bias = [0.0, 0.0]
        self.joint_vel_limit = 200000.0
        self.stride_limit = 0.15
        self.control_interval = 0.035
        self.use_settled_initial_states = False
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/halfsquat_init.txt'))

        self.heuristic_balance = None#[0.01969571, 0.01289562, 0.03515518, 0.03291302] # balance gains for knee, angle pitch, hip, ankle roll

        self.kp = None
        self.kd = None

        if self.use_DCMotor:
            self.motors = DCMotor(0.0107, 8.3, 12, 193)

        if self.root_input:
            obs_dim += 4  # won't include linear part
            if self.include_heading:
                obs_dim += 2
        if self.fallstate_input:
            obs_dim += 2

        self.act_delay = 0
        self.randomized_action_delay = False

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.control_bounds = np.array([-np.ones(20, ), np.ones(20, )])

        # normal pose
        self.permitted_contact_ids = [-1, -2, -7, -8]

        if self.trans_euler_root:
            self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.12, 0.0])
        else:
            self.init_root_pert = np.array([0.0, 0.12, 0.0, 0.0, 0.0, 0.0])

        self.no_target_vel = True
        self.exp_target_vel_rew = False
        self.transition_input = True # whether to input transition bit
        self.linear_target_vel_reward = False
        self.target_vel = 0.0
        self.init_tv = 0.0
        self.final_tv = 0.15
        self.tv_endtime = 1.0
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.target_vel_cache = []

        self.assist_timeout = 0.0
        self.assist_schedule = [[0.0,[2000,1000]],[3.0,[1500,500]],[6.0,[1125.0,250.0]]]

        self.alive_bonus = 4.5
        self.energy_weight = 0.05
        self.work_weight = 0.03
        self.dq_weight = 0.5
        self.vel_reward_weight = 20.0
        self.pose_weight = 5.0
        self.contact_weight = 0.0
        self.soft_ground = False

        self.cur_step = 0

        self.torqueLimits = 10.0

        self.t = 0
        self.target = np.zeros(26, )
        self.tau = np.zeros(26, )
        self.avg_tau = np.zeros(26, )

        if not self.no_target_vel:
            obs_dim += 1
        if self.transition_input:
            obs_dim += 1

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
             -23, -24, -25, -20, -21, -22, -26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33])
        act_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])

        if self.root_input:
            beginid = len(obs_perm_base)
            if self.include_heading:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2,  -beginid - 3, beginid + 4, -beginid - 5]])
            else:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid+1, -beginid-2, beginid+3]])
        if self.fallstate_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1]])
        if not self.no_target_vel:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid]])
        if self.transition_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid]])
        if self.train_UP:
            obs_perm_base = np.concatenate([obs_perm_base, np.arange(len(obs_perm_base), len(obs_perm_base) + len(self.param_manager.activated_param))])

        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.copy(act_perm_base)

        if self.use_discrete_action:
            from gym import spaces
            self.action_space = spaces.MultiDiscrete([11] * 20)
            #self.action_space = spaces.MultiDiscrete([5, 5, 5, 5, 5, 5, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])

        model_file_list = ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision.URDF', 'darwinmodel/coord.urdf', 'darwinmodel/mocap_markers.urdf', 'darwinmodel/robotis_op2.urdf']
        if self.soft_ground:
            model_file_list[0] = 'darwinmodel/soft_ground.skel'
            model_file_list.insert(3, 'darwinmodel/soft_ground.urdf')
        dart_env.DartEnv.__init__(self, model_file_list, 15, obs_dim,
                                  self.control_bounds, dt=0.002, disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")

        if self.trans_euler_root:
            self.robot_skeleton.set_root_joint_to_trans_and_euler()
            self.robot_skeleton.joints[1].set_axis_order("ZYX")

        self.orig_bodynode_masses = [bn.mass() for bn in self.robot_skeleton.bodynodes]

        self.dart_world.set_gravity([0, 0, -9.81])

        self.dupSkel = self.dart_world.skeletons[1]
        self.dupSkel.set_mobile(False)

        self.markers = self.dart_world.skeletons[3]


        self.dart_world.set_collision_detector(0)

        self.robot_skeleton.set_self_collision_check(True)

        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_L'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_L'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_R'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_L'),
                                           self.robot_skeleton.bodynode('l_hand'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_R'),
                                           self.robot_skeleton.bodynode('r_hand'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_R'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_L'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_L'))

        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(1.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(1.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(0.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(0.0)

        self.add_perturbation = False
        self.perturbation_parameters = [0.2, 10, ['MP_ANKLE2_R', 'MP_ANKLE2_L'], 10]  # probability, magnitude, bodyids, duration
        self.perturbation_duration = 40
        self.perturbing_body_name = None

        for i in range(6, self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(0.515)
            j.set_coulomb_friction(0.0)
        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(6, self.robot_skeleton.ndofs):
            self.robot_skeleton.dofs[i].set_position_lower_limit(JOINT_LOW_BOUND[i - 6] - 0.01)
            self.robot_skeleton.dofs[i].set_position_upper_limit(JOINT_UP_BOUND[i - 6] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        self.initial_local_coms = [b.local_com() for b in self.robot_skeleton.bodynodes]

        ################# temp code, ugly for now, should fix later ###################################
        if self.use_sysid_model:
            self.param_manager.set_simulator_parameters(
                np.array([2.87159059e-01, 4.03160514e-01, 4.36576586e-01, 3.86221239e-01,
                          7.85789054e-01, 1.04277029e-01, 3.64862787e-01, 3.98563863e-01,
                          9.36966648e-01, 9.56131312e-01, 8.74345365e-01, 8.39548565e-01,
                          9.90829332e-01, 1.07563860e-01, 6.43309153e-01, 9.88438984e-01,
                          2.85672012e-01, 9.67511394e-01, 5.98024447e-01, 1.59794372e-01,
                          9.97536608e-01, 4.88691407e-01, 5.01293655e-01, 7.95171350e-01,
                          9.95825152e-02, 7.09580629e-03, 4.66536839e-01, 5.25860303e-01,
                          8.20514312e-01, 9.35216575e-04, 2.74604822e-01, 7.11505683e-02,
                          4.56312986e-01, 9.28976189e-01, 7.45092860e-01, 5.09716306e-01,
                          6.45103472e-01, 7.33841140e-01, 3.06389080e-01, 9.99043259e-01,
                          2.37641857e-01]))
            self.param_manager.controllable_param.remove(self.param_manager.NEURAL_MOTOR)
            self.param_manager.set_bounds(np.array([0.62754478, 1., 1., 0.91796176, 0.99481419,
                                                    0.62411285, 0.58039399, 1., 1., 1.,
                                                    1., 0.51500521, 1., 0.6]),
                                          np.array([0.1620302, 0.23465617, 0.18431452, 0.24797362, 0.53741423,
                                                    0., 0.052823, 0.22073834, 0.34243664, 0.74839466,
                                                    0., 0., 0., 0.1]))
        self.default_kp_ratios = np.copy(self.kp_ratios)
        self.default_kd_ratios = np.copy(self.kd_ratios)
        ######################################################################################################

        print('Total mass: ', self.robot_skeleton.mass())
        print('Permitted ground contact: ', self.permitted_contact_bodies)

        utils.EzPickle.__init__(self)

    def _bodynode_spd(self, bn, kp, dof, target_vel=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0
        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (bn.dC[dof] - target_vel)
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt
        return tau


    def advance(self, a):
        if self._get_viewer() is not None:
            if hasattr(self._get_viewer(), 'key_being_pressed'):
                if self._get_viewer().key_being_pressed is not None:
                    if self._get_viewer().key_being_pressed == b'p':
                        self.paused = not self.paused
                        time.sleep(0.1)

        if self.paused:
            return

        clamped_control = np.array(a)

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        if self.use_delta_action > 0:
            self.target[6:] = np.clip(clamped_control * self.use_delta_action + self.robot_skeleton.q[6:],
                                      CONTROL_LOW_BOUND, CONTROL_UP_BOUND)
        else:
            self.target[6:] = (clamped_control + 1.0) / 2.0 * (CONTROL_UP_BOUND - CONTROL_LOW_BOUND) + CONTROL_LOW_BOUND

        if self.heuristic_balance is not None: # apply heuristic balance control
            gyro = self.get_sim_bno55(supress_randomization=False)
            # hip
            gyro_lr = gyro[3]
            gyro_fb = gyro[4]

            #gyro_lr = (gyro[0] - self.initial_gyro[0]) * 5
            #gyro_fb = (gyro[1] - self.initial_gyro[1]) * 5
            self.target[21] -= self.heuristic_balance[2] * gyro_lr
            self.target[15] -= self.heuristic_balance[2] * gyro_lr

            # knee
            self.target[23] += self.heuristic_balance[0] * gyro_fb
            self.target[17] -= self.heuristic_balance[0] * gyro_fb

            # ankle pitch
            self.target[24] -= self.heuristic_balance[1] * gyro_fb
            self.target[18] += self.heuristic_balance[1] * gyro_fb

            # ankle roll
            self.target[25] += self.heuristic_balance[3] * gyro_lr
            self.target[19] += self.heuristic_balance[3] * gyro_lr
            self.target[6:] = np.clip(self.target[6:], CONTROL_LOW_BOUND, CONTROL_UP_BOUND)


        dup_pos = np.copy(self.target)
        dup_pos[4] = 0.5

        self.dupSkel.set_positions(dup_pos)
        self.dupSkel.set_velocities(self.target*0)

        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]
                    self.perturbation_duration = self.perturbation_parameters[3]
                    self.perturbing_body_name = np.random.choice(self.perturbation_parameters[2])
            else:
                self.perturbation_duration -= 1

        self.avg_tau = np.zeros(26, )
        for i in range(self.frame_skip):
            self.tau[6:] = self.PID()

            self.tau[0:6] *= 0.0

            #if self.t > 1.5 and self.t < 2.5:
            #    self.robot_skeleton.bodynodes[0].add_ext_force([np.sign(np.random.random()-0.5)*50, np.sign(np.random.random()-0.5)*50, 0])
            if self.add_perturbation and self.perturbation_duration > 0:
                self.robot_skeleton.bodynode(self.perturbing_body_name).add_ext_force(self.perturb_force)

            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.current_pd, 1)
                self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([0, force, 0]))

                if not self.no_target_vel:
                    force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.vel_enforce_kp, 0, self.target_vel)
                    self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([force, 0, 0]))

            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()
            self.avg_tau += self.tau
        self.avg_tau /= self.frame_skip

    def NN_forward(self, input):
        NN_out = np.dot(np.tanh(np.dot(input, self.NN_motor_parameters[0]) + self.NN_motor_parameters[1]),
                        self.NN_motor_parameters[2]) + self.NN_motor_parameters[3]

        NN_out = np.exp(-np.logaddexp(0, -NN_out))
        return NN_out

    def PID(self):
        # print("#########################################################################3")

        if self.use_DCMotor:
            if self.kp is not None:
                kp = self.kp
                kd = self.kd
            else:
                kp = np.array([4]*20)
                kd = np.array([0.032]*20)
            pwm_command = -1 * kp * (self.robot_skeleton.q[6:] - self.target[6:]) - kd * self.robot_skeleton.dq[6:]
            tau = np.zeros(26, )
            tau[6:] = self.motors.get_torque(pwm_command, self.robot_skeleton.dq[6:])
        elif self.use_spd:
            if self.kp is not None:
                kp = np.array([self.kp]*26)
                kd = np.array([self.kd]*26)
                kp[0:6] *= 0
                kd[0:6] *= 0
            else:
                kp = np.array([0]*26)
                kd = np.array([0.0]*26)

            p = -kp * (self.robot_skeleton.q + self.robot_skeleton.dq * self.sim_dt - self.target)
            d = -kd * self.robot_skeleton.dq
            qddot = np.linalg.solve(self.robot_skeleton.M + np.diagflat(kd) * self.sim_dt, -self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
            tau = p + d - kd * qddot * self.sim_dt
            tau[0:6] *= 0
        elif self.NN_motor:
            q = np.array(self.robot_skeleton.q)
            qdot = np.array(self.robot_skeleton.dq)
            tau = np.zeros(26, )

            input = np.vstack([np.abs(q[6:] - self.target[6:]) * 5.0, np.abs(qdot[6:])]).T

            NN_out = self.NN_forward(input)

            kp = NN_out[:, 0] * (self.NN_motor_bound[0][0] - self.NN_motor_bound[1][0]) + self.NN_motor_bound[1][0]
            kd = NN_out[:, 1] * (self.NN_motor_bound[0][1] - self.NN_motor_bound[1][1]) + self.NN_motor_bound[1][1]

            kp[0:6] *= self.kp_ratios[0]
            kp[7:8] *= self.kp_ratios[1]
            kp[8:11] *= self.kp_ratios[2]
            kp[14:17] *= self.kp_ratios[2]
            kp[11] *= self.kp_ratios[3]
            kp[17] *= self.kp_ratios[3]
            kp[12:14] *= self.kp_ratios[4]
            kp[18:20] *= self.kp_ratios[4]

            kd[0:6] *= self.kd_ratios[0]
            kd[7:8] *= self.kd_ratios[1]
            kd[8:11] *= self.kd_ratios[2]
            kd[14:17] *= self.kd_ratios[2]
            kd[11] *= self.kd_ratios[3]
            kd[17] *= self.kd_ratios[3]
            kd[12:14] *= self.kd_ratios[4]
            kd[18:20] *= self.kd_ratios[4]

            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:]

            tau[(np.abs(self.robot_skeleton.dq) > self.joint_vel_limit) * (
                    np.sign(self.robot_skeleton.dq) == np.sign(tau))] = 0
        else:
            if self.kp is not None:
                kp = self.kp
                kd = self.kd
            else:
                kp = np.array([2.1 + 10, 1.79 + 10, 4.93 + 10,
                               2.0 + 10, 2.02 + 10, 1.98 + 10,
                               2.2 + 10, 2.06 + 10,
                               60, 60, 60, 60, 153, 102,
                               60, 60, 60, 60, 153, 102])
                kd = np.array([0.021, 0.023, 0.022,
                               0.025, 0.021, 0.026,
                               0.028, 0.0213
                                  , 0.2, 0.2, 0.2, 0.2, 0.02, 0.02,
                               0.2, 0.2, 0.2, 0.2, 0.02, 0.02])

                kp[0:6] *= self.kp_ratios[0]
                kp[7:8] *= self.kp_ratios[1]
                kp[8:11] *= self.kp_ratios[2]
                kp[14:17] *= self.kp_ratios[2]
                kp[11] *= self.kp_ratios[3]
                kp[17] *= self.kp_ratios[3]
                kp[12:14] *= self.kp_ratios[4]
                kp[18:20] *= self.kp_ratios[4]

                kd[0:6] *= self.kd_ratios[0]
                kd[7:8] *= self.kd_ratios[1]
                kd[8:11] *= self.kd_ratios[2]
                kd[14:17] *= self.kd_ratios[2]
                kd[11] *= self.kd_ratios[3]
                kd[17] *= self.kd_ratios[3]
                kd[12:14] *= self.kd_ratios[4]
                kd[18:20] *= self.kd_ratios[4]

            q = self.robot_skeleton.q
            qdot = self.robot_skeleton.dq
            tau = np.zeros(26, )
            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:]

            # hacky speed limit
            tau[
                (np.abs(self.robot_skeleton.dq) > self.joint_vel_limit) * (np.sign(self.robot_skeleton.dq) == np.sign(tau))] = 0

        torqs = self.ClampTorques(tau)

        return torqs[6:]

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(6, 26):
            if torques[i] > torqueLimits:  #
                torques[i] = torqueLimits
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques

    def step(self, a):
        if self.use_discrete_action:
            a = a * 1.0/ np.floor(self.action_space.nvec/2.0) - 1.0

        self.action_filter_cache.append(a)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0:
            a = np.mean(self.action_filter_cache, axis=0)

        self.action_buffer.append(np.copy(a))

        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                self.final_tv - self.init_tv) + self.init_tv

        self.current_pd = self.assist_schedule[0][1][0]
        self.vel_enforce_kp = self.assist_schedule[0][1][1]
        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        # reward
        vel = (xpos_after - xpos_before) / self.dt
        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(1.0 / self.dt / 1.0):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        if self.no_target_vel:
            vel_rew = vel * self.vel_reward_weight
            if self.linear_target_vel_reward:
                if vel > self.final_tv:
                    vel_rew = np.max([0, (2*self.final_tv - vel)]) * self.vel_reward_weight
        else:
            if not self.exp_target_vel_rew:
                vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
            else:
                vel_rew = self.vel_reward_weight * np.exp(-25*np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache)))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5

        upright_rew = np.abs(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz')[1])

        reward = -self.energy_weight * np.sum(
            np.abs(self.avg_tau)) + vel_rew + self.alive_bonus - upright_rew * self.pose_weight
        reward -= self.work_weight * np.sum(np.abs(self.avg_tau * self.robot_skeleton.dq))
        reward -= self.dq_weight * np.mean(np.abs(np.array(self.robot_skeleton.dq)[14:17]) + np.abs(np.array(self.robot_skeleton.dq)[20:23])) # penalize dq on the legs

        reward -= np.abs(self.robot_skeleton.q[4]) * 1.5 # prevent moving sideways

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and (com_height > -0.55) and
                    (abs(self.robot_skeleton.q[0]) < 0.8) and (
                            abs(self.robot_skeleton.q[1]) < 0.8))

        self.fall_on_ground = False
        self_colliding = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        total_contact_force = np.zeros(3)

        ground_bodies = [self.dart_world.skeletons[0].bodynodes[0]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            total_contact_force += contact.force
            if contact.bodynode1 not in self.permitted_contact_bodies and contact.bodynode2 not in self.permitted_contact_bodies:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    self.fall_on_ground = True
            if contact.bodynode1.skel == contact.bodynode2.skel:
                self_colliding = True
        self.foot_lift = False
        for contact in contacts:
            if contact.bodynode1.name == 'MP_ANKLE2_R' or contact.bodynode2.name == 'MP_ANKLE2_R':
                if self.robot_skeleton.bodynode('MP_ANKLE2_R').C[2] > -0.3:
                    self.foot_lift = True
            if contact.bodynode1.name == 'MP_ANKLE2_L' or contact.bodynode2.name == 'MP_ANKLE2_L':
                if self.robot_skeleton.bodynode('MP_ANKLE2_L').C[2] > -0.3:
                    self.foot_lift = True

        reward -= self.contact_weight * np.linalg.norm(total_contact_force)

        if self.stride_limit < np.linalg.norm(self.robot_skeleton.bodynode('MP_ANKLE2_R').C - self.robot_skeleton.bodynode('MP_ANKLE2_L').C):
            done = True

        if self.fall_on_ground:
            done = True

        if self_colliding:
            done = True

        if self.foot_lift:
            done = True

        if done:
            reward = 0

        if not self.paused:
            self.t += self.dt * 1.0
            self.cur_step += 1


        ob = self._get_obs()
        if self.debug_env:
            print('avg vel: ', (self.robot_skeleton.q[3] - self.init_q[3]) / self.t, vel_rew, np.mean(self.vel_cache))

        # move the obstacle forward when the robot has passed it
        if self.randomize_obstacle and not self.soft_ground:
            if self.robot_skeleton.C[0] - 0.4 > self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].offset()[0]:
                offset = np.copy(self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].offset())
                offset[0] += 1.0
                self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].set_offset(offset)
                self.dart_world.skeletons[0].bodynodes[1].shapenodes[1].set_offset(offset)

        return ob, reward, done, {'avg_vel': np.mean(self.vel_cache)}

    def get_body_quaternion(self):
        q = quaternion_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T)
        return q

    def get_sim_bno55(self, supress_randomization=False):
        # simulate bno55 reading
        tinv = np.linalg.inv(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3])
        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]
        langvel = np.dot(tinv, angvel) + np.random.uniform(-0.1, 0.1, 3)

        if self.trans_euler_root:
            euler = np.array(self.robot_skeleton.q[3:6])
            euler = [euler[2], euler[1], euler[0]]
        else:
            euler = np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz'))

        if self.randomize_gyro_bias and not supress_randomization:
            euler += np.random.uniform(-0.03, 0.03, 3)
            euler[0:2] += self.gyro_bias
        return np.array([euler[0]+0.08040237422714677, euler[1]-0.075-0.12483721034195938, euler[2], langvel[0], langvel[1], langvel[2]])

    def falling_state(self): # detect if it's falling fwd/bwd or left/right
        gyro = self.get_sim_bno55()# - self.initial_gyro
        fall_flags = [0, 0]
        if np.abs(gyro[0]) > 0.5:
            fall_flags[0] = np.sign(gyro[0])
        if np.abs(gyro[1]) > 0.5:
            fall_flags[1] = np.sign(gyro[1])
        return fall_flags

    def _get_obs(self):
        # phi = np.array([self.count/self.ref_traj.shape[0]])
        # print(ContactList.shape)
        state = np.concatenate([self.robot_skeleton.q[6:], self.robot_skeleton.dq[6:]])
        if self.multipos_obs > 0:
            state = self.robot_skeleton.q[6:]

            self.obs_cache.append(state)
            while len(self.obs_cache) < self.multipos_obs:
                self.obs_cache.append(state)
            if len(self.obs_cache) > self.multipos_obs:
                self.obs_cache.pop(0)

            state = np.concatenate(self.obs_cache)

        if self.root_input:
            gyro = self.get_sim_bno55()# - self.initial_gyro
            if not self.include_heading:
                gyro = np.array([gyro[0], gyro[1], self.last_root[0], self.last_root[1]])
                self.last_root = [gyro[0], gyro[1]]
            else:
                adjusted_heading = (gyro[2] - self.initial_heading) % (2 * np.pi)
                adjusted_heading = adjusted_heading - 2 * np.pi if adjusted_heading > np.pi else adjusted_heading
                gyro = np.array(
                    [gyro[0], gyro[1], adjusted_heading, self.last_root[0], self.last_root[1], self.last_root[2]])
                self.last_root = [gyro[0], gyro[1], adjusted_heading]
            state = np.concatenate([state, gyro])
        if self.fallstate_input:
            state = np.concatenate([state, self.falling_state()])

        if not self.no_target_vel:
            state = np.concatenate([state, [self.target_vel]])

        if self.transition_input:
            if self.t < self.tv_endtime:
                state = np.concatenate([state, [0]])
            else:
                state = np.concatenate([state, [1]])

        if self.train_UP:
            state = np.concatenate([state, self.current_param])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(
            self.robot_skeleton.ndofs)  # self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005,
                                                               size=self.robot_skeleton.ndofs)  # np.zeros(self.robot_skeleton.ndofs) #

        qpos[6:] = 0.5*(pose_squat_rad + pose_stand_rad)


        # qpos[0:3] += np.random.uniform(low=-0.1, high=0.1, size=3)
        qpos[6:] += np.random.uniform(low=-0.05, high=0.05, size=20)
        # self.target = qpos
        self.count = 0
        qpos[0:6] += self.init_root_pert
        self.set_state(qpos, qvel)

        q = self.robot_skeleton.q
        qid = 2 if self.trans_euler_root else 5
        q[qid] += -0.335 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-8].C[2]])
        if self.use_settled_initial_states:
            q = self.init_states_candidates[np.random.randint(len(self.init_states_candidates))]
        self.robot_skeleton.q = q

        self.init_q = np.copy(self.robot_skeleton.q)

        self.t = 0
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
            self.initial_heading = self.get_sim_bno55()[2]
        self.initial_gyro = self.get_sim_bno55(supress_randomization=True)

        self.action_buffer = []

        self.current_param = self.param_manager.get_simulator_parameters()
        if self.resample_MP:
            self.param_manager.resample_parameters()
            if self.range_robust > 0:
                lb = np.clip(self.current_param - self.range_robust, -0.05, 1.05)
                ub = np.clip(self.current_param + self.range_robust, -0.05, 1.05)
                self.current_param = np.random.uniform(lb, ub)

        if self.randomize_timestep:
            new_control_dt = self.control_interval + np.random.uniform(0.0, 0.01)
            default_fs = int(self.control_interval / 0.002)
            self.frame_skip = np.random.randint(-5, 5) + default_fs
            self.dart_world.dt = new_control_dt / self.frame_skip

        self.obs_cache = []

        if self.resample_MP or self.mass_ratio != 0:
            # set the ratio of the mass
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(self.orig_bodynode_masses[i] * self.mass_ratio)

        self.dart_world.skeletons[2].q = [0,0,0, 100, 100, 100]

        self.action_filter_cache = []
        #for i in range(self.action_filtering):
        #    self.action_filter_cache.append((0.5*(pose_squat_rad + pose_stand_rad) - CONTROL_LOW_BOUND)/(CONTROL_UP_BOUND - CONTROL_LOW_BOUND) * 2.0 - 1.0)

        self.avg_rew_weighting = []
        self.vel_cache = []
        self.target_vel_cache = []

        self.target = np.zeros(26, )
        self.tau = np.zeros(26, )
        self.avg_tau = np.zeros(26, )

        if self.randomize_obstacle and not self.soft_ground:
            horizontal_range = [0.6, 0.7]
            vertical_range = [-1.378, -1.378]
            sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
            sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1])
            self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].set_offset([sampled_h, 0, sampled_v])
            self.dart_world.skeletons[0].bodynodes[1].shapenodes[1].set_offset([sampled_h, 0, sampled_v])

        if self.randomize_gyro_bias:
            self.gyro_bias = np.random.uniform(-0.3, 0.3, 2)


        if self.randomized_action_delay:
            self.act_delay = np.random.randint(0, 2)

        q = self.dupSkel.q
        q[4] = 0.5
        self.dupSkel.set_positions(q)

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.0
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0