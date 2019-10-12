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
import copy, os
from gym.envs.dart.dc_motor import DCMotor

from gym.envs.dart.darwin_utils import *
from gym.envs.dart.parameter_managers import *
from gym.envs.dart.action_filter import *
import time

from pydart2.utils.transformations import euler_from_matrix, quaternion_from_matrix, euler_from_quaternion

class DartDarwinSquatEnv(dart_env.DartEnv, utils.EzPickle):
    WALK, SQUATSTAND, STEPPING, FALLING, HOP, CRAWL, STRANGEWALK, KUNGFU, BONGOBOARD, CONSTANT = list(range(10))

    def __init__(self):

        obs_dim = 40

        self.streaming_mode = False     # Mimic the streaming reading mode on the robot
        self.state_cache = [None, 0.0]
        self.gyro_cache = [None, 0.0]
        self.action_head_past = [None, None]

        self.root_input = True
        self.include_heading = True
        self.transition_input = False  # whether to input transition bit
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
        self.fallstate_input = False

        self.gyro_only_mode = False

        self.action_filtering = 5 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.butterworth_filter = False

        self.action_delay = 0.0
        self.action_queue = []

        self.future_ref_pose = 0  # step of future trajectories as input

        self.obs_cache = []
        self.multipos_obs = 2 # give multiple steps of position info instead of pos + vel
        if self.multipos_obs > 0:
            obs_dim = 20 * self.multipos_obs

        self.kp_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.kd_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.use_discrete_action = False

        self.use_sysid_model = True

        if self.use_sysid_model:
            self.param_manager = darwinParamManager(self)
            #self.param_manager.activated_param.remove(self.param_manager.NEURAL_MOTOR)
        else:
            self.param_manager = darwinSquatParamManager(self)

        self.use_SPD = False

        self.use_DCMotor = False
        self.NN_motor = False
        self.NN_motor_hid_size = 5 # assume one layer
        self.NN_motor_parameters = [np.random.random((2, self.NN_motor_hid_size)), np.random.random(self.NN_motor_hid_size),
                                    np.random.random((self.NN_motor_hid_size, 2)), np.random.random(2)]
        self.NN_motor_bound = [[200.0, 1.0], [0.0, 0.0]]

        self.supress_all_randomness = False
        self.use_settled_initial_states = False
        self.limited_joint_vel = True
        self.joint_vel_limit = 20000.0
        self.train_UP = False
        self.noisy_input = True
        self.resample_MP = True
        self.range_robust = 0.25 # std to sample at each step
        self.randomize_timestep = True
        self.randomize_action_delay = False
        self.load_keyframe_from_file = True
        self.randomize_gravity_sch = False
        self.randomize_obstacle = True
        self.randomize_gyro_bias = True
        self.gyro_bias = [0.0, 0.0]
        self.height_drop_threshold = 0.8    # terminate if com height drops for this amount
        self.orientation_threshold = 1.0    # terminate if body rotates for this amount
        self.control_interval = 0.034  # control every 50 ms
        self.sim_timestep = 0.0005
        self.forward_reward = 20.0
        self.velocity_clip = 0.3
        self.contact_pen = 0.0
        self.kp = None
        self.kd = None
        self.kc = None

        self.soft_ground = False
        self.soft_foot = False
        self.task_mode = self.STEPPING
        self.side_walk = False

        if self.gyro_only_mode:
            obs_dim = 0
            self.control_interval = 0.006

        if self.use_DCMotor:
            self.motors = DCMotor(0.0107, 8.3, 12, 193)

        obs_dim += self.future_ref_pose * 20

        if self.root_input:
            obs_dim += 4
            if self.include_heading:
                obs_dim += 2
        if self.fallstate_input:
            obs_dim += 2

        if self.transition_input:
            obs_dim += 1

        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        if self.task_mode == self.BONGOBOARD: # add observation about board
            obs_dim += 3

        self.control_bounds = np.array([-np.ones(20, ), np.ones(20, )])

        self.observation_buffer = []
        self.obs_delay = 0

        self.gravity_sch = [[0.0, np.array([0, 0, -9.81])]]

        self.initialize_falling = False # initialize darwin to have large ang vel so that it falls

        if self.side_walk:
            self.init_root_pert = np.array([0.0, 0., -1.57, 0.0, 0.0, 0.0])


        self.delta_angle_scale = 0.3

        self.alive_bonus = 3.0
        self.energy_weight = 0.0
        self.work_weight = 0.0
        self.pose_weight = 0.0
        self.upright_weight = 0.0
        self.comvel_pen = 0.0
        self.compos_pen = 0.0
        self.compos_range = 0.5

        self.cur_step = 0

        self.torqueLimits = 10.0

        self.t = 0
        self.target = np.zeros(20, )
        self.tau = np.zeros(20, )

        self.include_obs_history = 1
        self.include_act_history = 0
        self.input_obs_difference = False
        self.input_difference_sign = False    # Whether to use the sign instead of actual value for joint velocity
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
             -23, -24, -25, -20, -21, -22, -26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33])
        if self.gyro_only_mode:
            obs_perm_base = np.array([])
        act_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])
        self.obs_perm = np.copy(obs_perm_base)

        if self.root_input:
            beginid = len(obs_perm_base)
            if self.include_heading:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2,  -beginid - 3, beginid + 4, -beginid - 5]])
            else:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2, beginid + 3]])
        if self.fallstate_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1]])
        if self.transition_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid-0.0001]])
        if self.task_mode == self.BONGOBOARD:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2]])
        if self.train_UP:
            obs_perm_base = np.concatenate([obs_perm_base, np.arange(len(obs_perm_base), len(obs_perm_base) + len(
                self.param_manager.activated_param))])

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
            #self.action_space = spaces.MultiDiscrete([5,5,5,5,5,5, 3,3, 7,7,7,7,7,7, 7,7,7,7,7,7])

        model_file_list = ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision.URDF', 'darwinmodel/coord.urdf', 'darwinmodel/tracking_box.urdf', 'darwinmodel/bongo_board.urdf', 'darwinmodel/robotis_op2.urdf']
        if self.soft_foot:
            model_file_list[5] = 'darwinmodel/robotis_op2_softfoot.urdf'
        if self.task_mode != self.BONGOBOARD:
            model_file_list.remove('darwinmodel/bongo_board.urdf')
        if self.soft_ground:
            model_file_list[0] = 'darwinmodel/soft_ground.skel'
            model_file_list.insert(4, 'darwinmodel/soft_ground.urdf')

        dart_env.DartEnv.__init__(self, model_file_list, int(self.control_interval / self.sim_timestep), obs_dim,
                                  self.control_bounds, dt=self.sim_timestep, disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")

        self.body_parts = [bn for bn in self.robot_skeleton.bodynodes if 'SHOE' not in bn.name and 'base_link' not in bn.name]
        self.body_part_ids = np.array([bn.id for bn in self.body_parts])

        self.actuated_dofs = [df for df in self.robot_skeleton.dofs if 'root' not in df.name and 'shoe' not in df.name]
        self.actuated_dof_ids = [df.id for df in self.actuated_dofs]
        self.observed_dof_ids = [df.id for df in self.robot_skeleton.dofs if 'shoe' not in df.name]

        self.mass_ratios = np.ones(len(self.body_part_ids))
        self.inertia_ratios = np.ones(len(self.body_part_ids))

        if self.soft_foot:
            self.left_foot_shoe_bodies = [bn for bn in self.robot_skeleton.bodynodes if 'SHOE_PIECE' in bn.name and '_L' in bn.name]
            self.right_foot_shoe_bodies = [bn for bn in self.robot_skeleton.bodynodes if 'SHOE_PIECE' in bn.name and '_R' in bn.name]
            self.left_foot_shoe_ids = [bn.id for bn in self.left_foot_shoe_bodies]
            self.right_foot_shoe_ids = [bn.id for bn in self.right_foot_shoe_bodies]
            self.left_shoe_dofs = [df for df in self.robot_skeleton.dofs if 'shoe' in df.name and '_l' in df.name]
            self.right_shoe_dofs = [df for df in self.robot_skeleton.dofs if 'shoe' in df.name and '_r' in df.name]
            for df in self.left_shoe_dofs:
                df.set_spring_stiffness(300)
            for df in self.right_shoe_dofs:
                df.set_spring_stiffness(300)

        # crawl
        if self.task_mode == self.CRAWL:
            self.permitted_contact_ids = self.body_part_ids[[-1, -2, -7, -8, 5, 10]]  # [-1, -2, -7, -8]
            self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.08, 0.0])
        else:
            # normal pose
            self.permitted_contact_ids = self.body_part_ids[[-1, -2, -7, -8, 5, 10]]
            self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        if self.soft_foot:
            self.permitted_contact_ids = np.concatenate([self.permitted_contact_ids, self.left_foot_shoe_ids,
                                                         self.right_foot_shoe_ids])


        self.orig_bodynode_masses = [bn.mass() for bn in self.body_parts]
        self.orig_bodynode_inertias = [bn.I for bn in self.body_parts]

        self.dart_world.set_gravity([0, 0, -9.81])

        self.dupSkel = self.dart_world.skeletons[1]
        self.dupSkel.set_mobile(False)
        self.dart_world.skeletons[2].set_mobile(False)

        if self.soft_ground:
            self.dart_world.set_collision_detector(1)
        elif self.task_mode == self.BONGOBOARD:
            self.dart_world.set_collision_detector(3)
        else:
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

        if self.soft_foot:
            for i in range(len(self.left_foot_shoe_bodies)):
                for j in range(i+1, len(self.left_foot_shoe_bodies)):
                    collision_filter.add_to_black_list(self.left_foot_shoe_bodies[i], self.left_foot_shoe_bodies[j])
            for i in range(len(self.right_foot_shoe_bodies)):
                for j in range(i+1, len(self.right_foot_shoe_bodies)):
                    collision_filter.add_to_black_list(self.right_foot_shoe_bodies[i], self.right_foot_shoe_bodies[j])


        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(1.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(1.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(2.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(2.0)

        self.add_perturbation = True
        self.perturbation_parameters = [1.0, 0.3, 1.7, [2, 4], 1]  # begin time, duration, interval, magnitude, bodyid

        for j in self.actuated_dofs:
            j.set_damping_coefficient(0.515)
            j.set_coulomb_friction(0.0)

        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(len(self.actuated_dofs)):
            self.actuated_dofs[i].set_position_lower_limit(JOINT_LOW_BOUND[i] - 0.01)
            self.actuated_dofs[i].set_position_upper_limit(JOINT_UP_BOUND[i] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        if self.task_mode == self.BONGOBOARD:
            self.permitted_contact_bodies += [b for b in self.dart_world.skeletons[4].bodynodes]

        self.initial_local_coms = [b.local_com() for b in self.body_parts]

        ################# temp code, ugly for now, should fix later ###################################
        if self.use_sysid_model:
            self.param_manager.controllable_param = [self.param_manager.NEURAL_MOTOR,
                                                     self.param_manager.GROUP_JOINT_DAMPING,
                                                     self.param_manager.TORQUE_LIM, self.param_manager.COM_OFFSET,
                                                     self.param_manager.GROUND_FRICTION,
                                                     self.param_manager.KP_RATIO_NEW, self.param_manager.KD_RATIO_NEW]
            self.param_manager.set_simulator_parameters(
                np.array([7.63827124e-01, 6.52141948e-01, 4.37448673e-01, 5.58407296e-01,
                           9.92869534e-01, 4.15985976e-01, 4.80333550e-04, 9.95652158e-01,
                           3.94852253e-01, 9.99425825e-01, 1.94822923e-01, 2.77363703e-01,
                           9.94341147e-01, 3.74610988e-02, 2.81161919e-01, 9.91237362e-01,
                           3.40033552e-01, 9.97303374e-01, 7.62400773e-01, 8.32268394e-03,
                           4.57166501e-02, 9.53054820e-01, 3.33607744e-01, 6.55950626e-01,
                           1.47100487e-02, 9.72871517e-02, 4.70468391e-01, 3.57097825e-01,
                           4.62482096e-01, 1.81480958e-01, 6.74777006e-01, 4.27127607e-01,
                           1.32549360e-01, 3.40771818e-01, 3.62564442e-01, 8.65710806e-01,
                           9.51805312e-01, 9.80734701e-01, 9.87864536e-01, 4.80389655e-02,
                           2.70850867e-01, 6.09420342e-01, 9.89786802e-01, 5.30789064e-01,
                           6.14362127e-01, 4.70147831e-02, 4.08334293e-01, 3.46425041e-01,
                           5.22267832e-01, 7.98739873e-01]))
            self.param_manager.controllable_param.remove(self.param_manager.NEURAL_MOTOR)
            self.param_manager.set_bounds(np.array([0.58061583, 0.67096192, 0.48950198, 0.93728849, 0.64818803,
                                                       0.28555102, 0.38719763, 0.61374865, 0.9998833 , 0.99968724,
                                                       0.99995329, 0.99999941, 0.34414109, 0.51244207, 0.86086581,
                                                       0.99998168, 0.66543859, 0.99704476, 0.22571845, 0.72031463,
                                                       0.49945877, 0.91750337, 0.91780714]),
                                          np.array([1.87736673e-01, 3.17682560e-01, 1.84464656e-03, 4.54161927e-01,
                                                       3.08896622e-01, 2.11543341e-02, 2.66393798e-01, 3.47638089e-02,
                                                       7.11617758e-01, 7.76779150e-01, 7.79217160e-01, 6.93054001e-01,
                                                       1.51816427e-02, 1.57287201e-01, 2.26766506e-01, 8.72094471e-01,
                                                       4.02714797e-01, 3.32055486e-01, 1.13931736e-07, 2.01030647e-01,
                                                       1.31721938e-01, 3.11503980e-01, 5.05246348e-01]))

            # self.param_manager.controllable_param = [self.param_manager.KP_RATIO, self.param_manager.KD_RATIO,
            #                                          self.param_manager.TORQUE_LIM, self.param_manager.COM_OFFSET,
            #                                          self.param_manager.GROUND_FRICTION,
            #                                          self.param_manager.MASS_RATIO]
            # pm = np.array([5.58685582e-01, 6.25612074e-01, 4.47443811e-01, 9.38706017e-01,
            #                       4.66674078e-01, 4.02949134e-01, 6.70079410e-01, 4.29772267e-01,
            #                       9.52876766e-01, 7.97069983e-01, 9.94781663e-01, 6.37435346e-02,
            #                       9.41793335e-01, 5.57858669e-01, 6.76382567e-01, 7.55240228e-01,
            #                       5.24886084e-01, 8.77853574e-01, 1.13199863e-02, 3.26414657e-01,
            #                       7.09153255e-01, 3.48032417e-01, 9.71712751e-01, 4.04702689e-01,
            #                       8.85026626e-04, 9.65573687e-03, 7.57381564e-01, 7.05641848e-01,
            #                       4.29009217e-01, 3.63941552e-02, 6.01583898e-01, 4.15528272e-01,
            #                       9.99878761e-01, 5.26067489e-01, 6.45363644e-01, 2.53598772e-01,
            #                       7.64184323e-01, 4.81381088e-01, 5.30623991e-01, 7.26399313e-01])
            # ub = np.clip(pm * 1.2, 0.0, 1.0)
            # lb = np.clip(pm * 0.8, 0.0, 1.0)
            # self.param_manager.set_simulator_parameters(pm)
            # self.param_manager.set_bounds(ub, lb)


        self.default_kp_ratios = np.copy(self.kp_ratios)
        self.default_kd_ratios = np.copy(self.kd_ratios)
        ######################################################################################################

        if self.use_SPD:
            self.Kp = np.diagflat([0.0] * 6 + [500.0] * (self.robot_skeleton.ndofs - 6))
            self.Kd = np.diagflat([0.0] * 6 + [1.0] * (self.robot_skeleton.ndofs - 6))

        print('Total mass: ', self.robot_skeleton.mass())
        print('Bodynodes: ', [b.name for b in self.robot_skeleton.bodynodes])

        if self.task_mode == self.WALK:
            self.setup_walk()
        elif self.task_mode == self.STEPPING:
            self.setup_stepping()
        elif self.task_mode == self.SQUATSTAND:
            self.setup_squatstand()
        elif self.task_mode == self.FALLING:
            self.setup_fall()
        elif self.task_mode == self.HOP:
            self.setup_hop()
        elif self.task_mode == self.CRAWL:
            self.setup_crawl()
        elif self.task_mode == self.STRANGEWALK:
            self.setup_strangewalk()
        elif self.task_mode == self.KUNGFU:
            self.setup_kungfu()
        elif self.task_mode == self.BONGOBOARD:
            self.setup_bongoboard()
        elif self.task_mode == self.CONSTANT:
            self.setup_constref()

        if self.butterworth_filter:
            self.action_filter = ActionFilter(self.act_dim, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=3.0)

        # self.set_robot_optimization_parameters(np.array([-0.27573608, -0.04001381,  0.16576692, -0.45604828,  0.83507119,
        # 0.2363036 , -0.37442629, -0.77073466,  0.69862929, -0.85059406]) * 0.01)

        utils.EzPickle.__init__(self)


    def setup_walk(self): # step up walk task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe.txt')
            rig_keyframe = np.loadtxt(fullpath)

            '''for i in range(len(rig_keyframe)):
                rig_keyframe[i][0:6] = 0.0
                rig_keyframe[i][1] = -0.5
                rig_keyframe[i][2] = 0.75
                rig_keyframe[i][4] = 0.5
                rig_keyframe[i][5] = -0.75'''

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.25
            for i in range(20):
                for k in range(1, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.25
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.5
        self.forward_reward = 10.0
        self.delta_angle_scale = 0.3
        self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/stand_init.txt'))

    def setup_stepping(self): # step up stepping task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_step.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, VAL2RADIAN(0.5 * (np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                        2047, 2171,
                                        2032, 2039, 2795, 648, 1241, 2040, 2041, 2060, 1281, 3448, 2855, 2073]) + np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                        2048, 2048,
                                        2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])))]]
            #rig_keyframe = [HW2SIM_INDEX(v) for v in VAL2RADIAN(rig_keyframe)]
            #self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.2
            for i in range(1):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.03
            self.interp_sch.append([interp_time, rig_keyframe[0]])
        self.compos_range = 0.5
        self.forward_reward = 10.0
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/halfsquat_init.txt'))

    def setup_constref(self): # constant reference motion
        const_pose = VAL2RADIAN(0.5 * (np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                                                 2047, 2171,
                                                                 2032, 2039, 2795, 648, 1241, 2040, 2041, 2060, 1281,
                                                                 3448, 2855, 2073]) + np.array(
                [1500, 2048, 2048, 2500, 2048, 2048,
                 2048, 2048,
                 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])))
        self.interp_sch = [[0.0, const_pose], [8.0, const_pose]]
        self.compos_range = 0.5
        self.forward_reward = 10.0
        self.delta_angle_scale = 0.6
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(
                os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/halfsquat_init.txt'))

    def setup_squatstand(self): # set up squat stand task
        self.interp_sch = [[0.0, pose_stand_rad],
                           [2.0, pose_squat_rad],
                           [3.5, pose_squat_rad],
                           [4.0, pose_stand_rad],
                           [5.0, pose_stand_rad],
                           [7.0, pose_squat_rad],
                           ]
        self.compos_range = 100.0
        self.forward_reward = 0.0
        self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 0.2
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt('darwinmodel/stand_init.txt')

    def setup_fall(self): # set up the falling task
        self.interp_sch = [[0.0, 0.5 * (pose_squat_rad + pose_stand_rad)],
                           [4.0, 0.5 * (pose_squat_rad + pose_stand_rad)]]
        self.compos_range = 100.0
        self.forward_reward = 0.0
        self.contact_pen = 0.05
        self.delta_angle_scale = 0.6
        self.alive_bonus = 8.0
        self.height_drop_threshold = 10.0
        self.orientation_threshold = 10.0
        self.initialize_falling = True
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/stand_init.txt'))

    def setup_hop(self): # set up hop task
        p0 = 0.2 * pose_stand_rad + 0.8 * pose_squat_rad
        p1 = 0.85 * pose_stand_rad + 0.15 * pose_squat_rad
        p1[0] -= 0.7
        p1[3] += 0.7
        self.interp_sch = []
        curtime = 0
        for i in range(20):
            self.interp_sch.append([curtime, p0])
            self.interp_sch.append([curtime+0.2, p1])
            #self.interp_sch.append([curtime+0.4, p0])
            curtime += 0.4

        self.compos_range = 100.0
        self.forward_reward = 10.0
        self.init_root_pert = np.array([0.0, 0.16, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 0.3
        self.energy_weight = 0.005
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/squat_init.txt'))

    def setup_crawl(self): # set up crawling task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_crawl.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.15
            for i in range(10):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.15
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.forward_reward = 10.0
        self.height_drop_threshold = 10.0
        self.orientation_threshold = 10.0

    def setup_strangewalk(self):
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_strangewalk.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.25
            for i in range(20):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.25
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.delta_angle_scale = 0.2
        self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        self.forward_reward = 10.0
        self.upright_weight = 1.0

    def setup_kungfu(self):
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_kungfu.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.3
            for i in range(1):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.1
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.forward_reward = 0.0
        self.upright_weight = 1.0

    def setup_bongoboard(self):
        from pydart2.constraints import BallJointConstraint
        p = 0.5 * (pose_squat_rad + pose_stand_rad)
        p[1] = -0.5
        p[2] = 0.75
        p[4] = 0.5
        p[5] = -0.75
        p[9] -= 0.3
        p[15] += 0.3
        p[13] -= 0.3
        p[19] += 0.3
        p = np.clip(p, JOINT_LOW_BOUND, JOINT_UP_BOUND)
        self.interp_sch = [[0, p], [5, p]]

        self.compos_range = 0.3
        self.forward_reward = 0.0
        self.init_root_pert = np.array([0.0, 0.06, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 1.0
        self.upright_weight = 0.5
        self.comvel_pen = 0.5
        self.compos_pen = 1.0
        if self.task_mode == self.BONGOBOARD:
            self.dart_world.skeletons[4].bodynodes[0].set_friction_coeff(20.0)
        #self.dart_world.skeletons[0].bodynodes[2].shapenodes[0].set_offset([0, 0.5, 0])
        #self.dart_world.skeletons[0].bodynodes[2].shapenodes[1].set_offset([0, 0.5, 0])
        self.param_manager.MU_UP_BOUNDS[self.param_manager.GROUND_FRICTION] = [2.0]
        self.param_manager.MU_LOW_BOUNDS[self.param_manager.GROUND_FRICTION] = [1.0]

        self.assist_timeout = 0.0
        self.assist_schedule = [[0.0, [2000, 2000]], [2.0, [1500, 1500]], [4.0, [1125.0, 1125.0]]]


    def adjust_root(self): # adjust root dof such that foot is roughly flat
        q = self.robot_skeleton.q
        q[1] += -1.57 - np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_ANKLE2_L').T[0:3, 0:3], 'sxyz'))[1]

        if not self.soft_foot:
            q[5] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        else:
            q[5] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2], self.left_foot_shoe_bodies[0].C[2], self.right_foot_shoe_bodies[0].C[2]])
        self.robot_skeleton.q = q

    def get_body_quaternion(self):
        q = quaternion_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T)
        return q

    def get_sim_bno55(self, supress_randomization=False):
        # simulate bno55 reading
        tinv = np.linalg.inv(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3])
        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]
        langvel = np.dot(tinv, angvel)
        euler = np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz'))

        if self.randomize_gyro_bias and not supress_randomization and not self.supress_all_randomness:
            euler[0:2] += self.gyro_bias
            # add noise
            euler += np.random.uniform(-0.01, 0.01, 3)
            langvel += np.random.uniform(-0.1, 0.1, 3)
        # if euler[0] > 1.5:
        #     euler[0] -= np.pi
        # return np.array([euler[0], euler[1]-0.025, euler[2], langvel[0], langvel[1], langvel[2]])
        return np.array(
            [euler[0] + 0.08040237422714677, euler[1] - 0.075 - 0.12483721034195938, euler[2], langvel[0], langvel[1],
             langvel[2]])

    def falling_state(self): # detect if it's falling fwd/bwd or left/right
        gyro = self.get_sim_bno55()
        fall_flags = [0, 0]
        if np.abs(gyro[0]) > 0.5:
            fall_flags[0] = np.sign(gyro[0])
        if np.abs(gyro[1]) > 0.5:
            fall_flags[1] = np.sign(gyro[1])
        return fall_flags

    def spd(self, target_q):
        invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.sim_dt)
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.sim_dt - np.concatenate([[0.0]*6, target_q]))
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.sim_dt
        return tau[6:]

    def advance(self, a):
        if self._get_viewer() is not None:
            if hasattr(self._get_viewer(), 'key_being_pressed'):
                if self._get_viewer().key_being_pressed is not None:
                    if self._get_viewer().key_being_pressed == b'p':
                        self.paused = not self.paused
                        time.sleep(0.1)

        if self.paused and self.t > 0:
            return

        clamped_control = np.array(a)

        if self.butterworth_filter:
            clamped_control = self.action_filter.filter_action(clamped_control)

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        self.ref_target = self.get_ref_pose(self.t)


        cur_target = self.ref_target + clamped_control * self.delta_angle_scale

        cur_target = np.clip(cur_target, JOINT_LOW_BOUND, JOINT_UP_BOUND)

        self.apply_target_pose(cur_target)

    def apply_target_pose(self, target_pose):
        dup_pos = np.concatenate([[0.0] * 6, target_pose])
        dup_pos[4] = 0.5
        self.dupSkel.set_positions(dup_pos)
        self.dupSkel.set_velocities(np.zeros(len(target_pose) + 6))

        self.action_queue.append([target_pose, self.t + self.action_delay])
        if self.add_perturbation and self.t >= self.perturbation_parameters[0]:# and not self.supress_all_randomness:
            if (self.t - self.perturbation_parameters[0]) % self.perturbation_parameters[2] <= 0.04:
                force_mag = np.random.uniform(self.perturbation_parameters[3][0], self.perturbation_parameters[3][1])
                force_dir = np.array([np.random.uniform(-1, 1), np.random.uniform(-0.3, 0.3), np.random.uniform(-0.1, 0.1)])
                self.perturb_force = force_dir / np.linalg.norm(force_dir) * force_mag
            elif (self.t - self.perturbation_parameters[0]) % self.perturbation_parameters[2] > \
                    self.perturbation_parameters[1]:
                self.perturb_force *= 0

        if self.streaming_mode:
            if self.action_head_past[1] is None:
                self.action_head_past[0] = np.copy(target_pose)
                self.action_head_past[1] = np.copy(target_pose)

        for i in range(self.frame_skip):
            if self.streaming_mode:
                if self.t + i * self.sim_dt - self.state_cache[1] > self.state_cache[2]:
                    self.action_head_past[1] = np.copy(self.action_head_past[0])
                    self.action_head_past[0] = np.copy(target_pose)
                self.target = np.copy(self.action_head_past[1])
            else:
                if len(self.action_queue) > 0 and self.t + i * self.sim_dt >= self.action_queue[0][1]:
                    self.target = np.copy(self.action_queue[0][0])
                    self.action_queue.pop(0)

            if self.use_SPD:
                self.tau = self.spd(target_pose)
            else:
                self.tau = self.PID()

            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[4]].add_ext_force(self.perturb_force)
            # if self.t > 1.0 and self.t < 1.4:
            #     self.robot_skeleton.bodynodes[self.perturbation_parameters[4]].add_ext_force([-4.0,0,0])

            robot_force = np.zeros(self.robot_skeleton.ndofs)
            robot_force[self.actuated_dof_ids] = self.tau
            self.robot_skeleton.set_forces(robot_force)

            self.dart_world.step()

            if self.streaming_mode:
                if self.t + i * self.sim_dt - self.state_cache[1] > self.state_cache[2]:
                    self.state_cache = [self.robot_skeleton.q[6:], self.t + i * self.sim_dt, 0.015 + np.random.normal(0.0, 0.005)]
                if self.t + i * self.sim_dt - self.gyro_cache[1] > self.gyro_cache[2]:
                    self.gyro_cache = [self.get_sim_bno55(), self.t + i * self.sim_dt, 0.015 + np.random.normal(0.0, 0.005)]

        if not self.paused or self.t == 0:
            self.t += self.dt * 1.0
            self.cur_step += 1

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
            pwm_command = -1 * kp * (np.array(self.robot_skeleton.q)[self.actuated_dof_ids] - self.target) - kd * np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]
            tau = self.motors.get_torque(pwm_command, np.array(self.robot_skeleton.dq)[self.actuated_dof_ids])
        elif self.NN_motor:
            q = np.array(self.robot_skeleton.q)
            qdot = np.array(self.robot_skeleton.dq)
            tau = np.zeros(20, )

            input = np.vstack([np.abs(q[self.actuated_dof_ids] - self.target) * 5.0, np.abs(qdot[self.actuated_dof_ids])]).T

            NN_out = self.NN_forward(input)

            kp = NN_out[:, 0] * (self.NN_motor_bound[0][0] - self.NN_motor_bound[1][0]) + self.NN_motor_bound[1][0]
            kd = NN_out[:, 1] * (self.NN_motor_bound[0][1] - self.NN_motor_bound[1][1]) + self.NN_motor_bound[1][1]

            if len(self.kp_ratios) == 5:
                kp[0:6] *= self.kp_ratios[0]
                kp[7:8] *= self.kp_ratios[1]
                kp[8:11] *= self.kp_ratios[2]
                kp[14:17] *= self.kp_ratios[2]
                kp[11] *= self.kp_ratios[3]
                kp[17] *= self.kp_ratios[3]
                kp[12:14] *= self.kp_ratios[4]
                kp[18:20] *= self.kp_ratios[4]

            if len(self.kp_ratios) == 10:
                kp[[0, 1, 2, 6, 8,9,10,11,12,13]] *= self.kp_ratios
                kp[[3, 4, 5, 7, 14,15,16,17,18,19]] *= self.kp_ratios

            if len(self.kp_ratios) == 7:
                kp[0:8] *= self.kp_ratios[0]
                kp[8] *= self.kp_ratios[1]
                kp[9] *= self.kp_ratios[2]
                kp[10] *= self.kp_ratios[3]
                kp[11] *= self.kp_ratios[4]
                kp[12] *= self.kp_ratios[5]
                kp[13] *= self.kp_ratios[6]
                kp[14] *= self.kp_ratios[1]
                kp[15] *= self.kp_ratios[2]
                kp[16] *= self.kp_ratios[3]
                kp[17] *= self.kp_ratios[4]
                kp[18] *= self.kp_ratios[5]
                kp[19] *= self.kp_ratios[6]

            if len(self.kd_ratios) == 5:
                kd[0:6] *= self.kd_ratios[0]
                kd[7:8] *= self.kd_ratios[1]
                kd[8:11] *= self.kd_ratios[2]
                kd[14:17] *= self.kd_ratios[2]
                kd[11] *= self.kd_ratios[3]
                kd[17] *= self.kd_ratios[3]
                kd[12:14] *= self.kd_ratios[4]
                kd[18:20] *= self.kd_ratios[4]

            if len(self.kd_ratios) == 10:
                kd[[0, 1, 2, 6, 8, 9, 10, 11, 12, 13]] *= self.kd_ratios
                kd[[3, 4, 5, 7, 14, 15, 16, 17, 18, 19]] *= self.kd_ratios

            if len(self.kd_ratios) == 7:
                kd[0:8] *= self.kd_ratios[0]
                kd[8] *= self.kd_ratios[1]
                kd[9] *= self.kd_ratios[2]
                kd[10] *= self.kd_ratios[3]
                kd[11] *= self.kd_ratios[4]
                kd[12] *= self.kd_ratios[5]
                kd[13] *= self.kd_ratios[6]
                kd[14] *= self.kd_ratios[1]
                kd[15] *= self.kd_ratios[2]
                kd[16] *= self.kd_ratios[3]
                kd[17] *= self.kd_ratios[4]
                kd[18] *= self.kd_ratios[5]
                kd[19] *= self.kd_ratios[6]

            tau = -kp * (q[self.actuated_dof_ids] - self.target) - kd * qdot[self.actuated_dof_ids]

            if self.limited_joint_vel:
                tau[(np.abs(np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]) > self.joint_vel_limit) * (
                        np.sign(np.array(self.robot_skeleton.dq))[self.actuated_dof_ids] == np.sign(tau))] = 0
        else:
            raise NotImplementedError

        torqs = self.ClampTorques(tau)

        return torqs

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(len(torques)):
            if torques[i] > torqueLimits:  #
                torques[i] = torqueLimits
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques

    def get_ref_pose(self, t):
        ref_target = self.interp_sch[0][1]

        for i in range(len(self.interp_sch) - 1):
            if t >= self.interp_sch[i][0] and t < self.interp_sch[i + 1][0]:
                ratio = (t - self.interp_sch[i][0]) / (self.interp_sch[i + 1][0] - self.interp_sch[i][0])
                ref_target = ratio * self.interp_sch[i + 1][1] + (1 - ratio) * self.interp_sch[i][1]
        if t > self.interp_sch[-1][0]:
            ref_target = self.interp_sch[-1][1]
        return ref_target

    def step(self, a):
        if self.task_mode == self.BONGOBOARD:
            self.current_assist = self.assist_schedule[0][1]
            if len(self.assist_schedule) > 0:
                for sch in self.assist_schedule:
                    if self.t > sch[0]:
                        self.current_assist = sch[1]

        if self.use_discrete_action:
            a = a * 1.0/ np.floor(self.action_space.nvec/2.0) - 1.0

        if not self.butterworth_filter:
            self.action_filter_cache.append(a)
            if len(self.action_filter_cache) > self.action_filtering:
                self.action_filter_cache.pop(0)
            if self.action_filtering > 0:
                a = np.mean(self.action_filter_cache, axis=0)

        self.action_buffer.append(np.copy(a))

        # modify gravity according to schedule
        grav = self.gravity_sch[0][1]

        for i in range(len(self.gravity_sch) - 1):
            if self.t >= self.gravity_sch[i][0] and self.t < self.gravity_sch[i + 1][0]:
                ratio = (self.t - self.gravity_sch[i][0]) / (self.gravity_sch[i + 1][0] - self.gravity_sch[i][0])
                grav = ratio * self.gravity_sch[i + 1][1] + (1 - ratio) * self.gravity_sch[i][1]
        if self.t > self.gravity_sch[-1][0]:
            grav = self.gravity_sch[-1][1]
        self.dart_world.set_gravity(grav)

        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        upright_rew = np.abs(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz')[1])

        pose_math_rew = np.sum(
            np.abs(np.array(self.ref_target - np.array(self.robot_skeleton.q)[self.actuated_dof_ids])) ** 2)
        reward = -self.energy_weight * np.sum(
            self.tau ** 2) + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.sum(np.abs(self.tau * np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]))
        # reward -= 2.0 * np.abs(self.robot_skeleton.dC[1])
        reward -= self.upright_weight * upright_rew

        # if falling to the back, encourage the robot to take steps backward
        if np.abs(self.robot_skeleton.q[1]) > 0.25:
            current_forward_reward = self.forward_reward * 0.5 * np.sign(self.robot_skeleton.q[1])
        else:
            current_forward_reward = self.forward_reward
        reward += current_forward_reward * np.clip((xpos_after - xpos_before) / self.dt, -self.velocity_clip,
                                                self.velocity_clip)


        reward -= self.comvel_pen * np.linalg.norm(self.robot_skeleton.dC)
        reward -= self.compos_pen * np.linalg.norm(self.init_q[3:6] - self.robot_skeleton.q[3:6])

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all())

        if np.any(np.abs(np.array(self.robot_skeleton.q)[0:2]) > self.orientation_threshold):
            done = True

        if not self.side_walk and np.abs(np.array(self.robot_skeleton.q)[2]) > self.orientation_threshold:
            done = True

        self.fall_on_ground = False
        self_colliding = False
        contacts = self.dart_world.collision_result.contacts
        total_force = np.zeros(3)

        ground_bodies = [self.dart_world.skeletons[0].bodynodes[0]]

        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                total_force += contact.force
            if contact.bodynode1 not in self.permitted_contact_bodies and contact.bodynode2 not in self.permitted_contact_bodies:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    self.fall_on_ground = True
            if contact.bodynode1.skel == contact.bodynode2.skel and self.cur_step > 1:
                self_colliding = True

        if self.t > self.interp_sch[-1][0] + 2:
            done = True
        if self.fall_on_ground:
            done = True
        if self_colliding:
            done = True
        if self.init_q[5] - self.robot_skeleton.q[5] > self.height_drop_threshold:
            done = True


        if self.compos_range > 0:
            if self.forward_reward == 0:
                if np.linalg.norm(self.init_q[3:6] - self.robot_skeleton.q[3:6]) > self.compos_range:
                    done = True
            else:
                if np.linalg.norm(self.init_q[4:6] - self.robot_skeleton.q[4:6]) > self.compos_range:
                    done = True

        if self.task_mode == self.BONGOBOARD:
            reward -= 10.0 * np.abs(self.dart_world.skeletons[4].q[0])
            board_touching_ground = False
            for contact in contacts:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    if contact.bodynode1 == self.dart_world.skeletons[4].bodynodes[1] or contact.bodynode2 == self.dart_world.skeletons[4].bodynodes[1]:
                        board_touching_ground = True
            if board_touching_ground:
                reward -= 5

        reward -= self.contact_pen * np.linalg.norm(total_force) # penalize contact forces

        if self.task_mode == self.WALK and 0.2 < np.linalg.norm(self.robot_skeleton.bodynode('MP_ANKLE2_R').C - self.robot_skeleton.bodynode('MP_ANKLE2_L').C):
            done = True

        if done:
            reward = 0

        if self.t > self.interp_sch[-1][0] + 2:
            done = True

        ob = self._get_obs()

        # move the obstacle forward when the robot has passed it
        if self.randomize_obstacle and not self.soft_ground and not self.supress_all_randomness:
            if self.robot_skeleton.C[0] - 0.4 > self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].offset()[0]:
                offset = np.copy(self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].offset())
                offset[0] += 1.0
                self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].set_offset(offset)
                self.dart_world.skeletons[0].bodynodes[1].shapenodes[1].set_offset(offset)

        #if self.range_robust > 0:
        #    rand_param = np.clip(self.current_param + np.random.normal(0, self.range_robust, len(self.current_param)), -0.05, 1.05)
        #    self.param_manager.set_simulator_parameters(rand_param)

        return ob, reward, done, {}

    def _get_obs(self, update_buffer=True):
        # phi = np.array([self.count/self.ref_traj.shape[0]])
        # print(ContactList.shape)
        state = np.concatenate([np.array(self.robot_skeleton.q)[self.actuated_dof_ids], np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]])
        if self.multipos_obs > 0:
            if self.streaming_mode:
                state = self.state_cache[0]
            else:
                state = np.array(self.robot_skeleton.q)[self.actuated_dof_ids]

            self.obs_cache.append(state)
            while len(self.obs_cache) < self.multipos_obs:
                self.obs_cache.append(state)
            if len(self.obs_cache) > self.multipos_obs:
                self.obs_cache.pop(0)

            for i in range(len(self.obs_cache)-1):
                if self.input_obs_difference:
                    if self.input_difference_sign:
                        state = np.concatenate([np.sign(self.obs_cache[i] - self.obs_cache[-1]), state])
                    else:
                        state = np.concatenate([self.obs_cache[i] - self.obs_cache[-1], state])
                else:
                    state = np.concatenate([self.obs_cache[i], state])

        for i in range(self.future_ref_pose):
            state = np.concatenate([state, self.get_ref_pose(self.t + self.dt * (i+1))])

        if self.gyro_only_mode:
            state = np.array([])

        if self.root_input:
            if self.streaming_mode:
                gyro = self.gyro_cache[0]
            else:
                gyro = self.get_sim_bno55()
            if not self.include_heading:
                gyro = np.array([gyro[0], gyro[1], self.last_root[0], self.last_root[1]])
                self.last_root = [gyro[0], gyro[1]]
            else:
                adjusted_heading = (gyro[2] - self.initial_heading) % (2*np.pi)
                adjusted_heading = adjusted_heading - 2*np.pi if adjusted_heading > np.pi else adjusted_heading
                gyro = np.array([gyro[0], gyro[1], adjusted_heading, self.last_root[0], self.last_root[1], self.last_root[2]])
                gyro[0:2] -= self.initial_gyro[0:2]
                self.last_root = [gyro[0], gyro[1], adjusted_heading]
            state = np.concatenate([state, gyro])
        if self.fallstate_input:
            state = np.concatenate([state, self.falling_state()])

        if self.transition_input:
            if self.t < 1.0:
                state = np.concatenate([state, [0]])
            else:
                state = np.concatenate([state, [1]])

        if self.task_mode == self.BONGOBOARD:
            board_ori = np.array(euler_from_matrix(self.dart_world.skeletons[4].bodynode('board').T[0:3, 0:3], 'sxyz'))
            state = np.concatenate([state, board_ori])

        if self.train_UP:
            #UP = self.param_manager.get_simulator_parameters()
            state = np.concatenate([state, self.current_param])

        if self.noisy_input and not self.supress_all_randomness:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                if i > 0 and self.input_obs_difference:
                    if self.input_difference_sign:
                        final_obs = np.concatenate([final_obs, np.sign(self.observation_buffer[-self.obs_delay - 1 - i] -
                                                    self.observation_buffer[-self.obs_delay - 1])])
                    else:
                        final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i] - self.observation_buffer[-self.obs_delay - 1]])
                else:
                    final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[- 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, np.zeros(self.act_dim)])

        return final_obs

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq
        if not self.supress_all_randomness:
            qpos += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        # LEFT HAND
        if self.interp_sch is not None:
            qpos[self.actuated_dof_ids] = np.clip(self.interp_sch[0][1], JOINT_LOW_BOUND, JOINT_UP_BOUND)
        else:
            qpos[self.actuated_dof_ids] = np.clip(0.5 * (pose_squat_rad + pose_stand_rad), JOINT_LOW_BOUND, JOINT_UP_BOUND)

        self.count = 0
        qpos[0:6] += self.init_root_pert

        if self.initialize_falling and not self.supress_all_randomness:
            qvel[0] = np.random.uniform(-2.0, 2.0)
            qvel[1] = np.random.uniform(-2.0, 2.0)

        self.set_state(qpos, qvel)

        q = self.robot_skeleton.q

        qid = 5
        if self.task_mode == self.CRAWL:
            q[qid] += -0.3 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        elif self.task_mode == self.BONGOBOARD:
            q[qid] += -0.25 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        else:
            if not self.soft_foot:
                q[qid] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
            else:
                q[qid] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2], self.left_foot_shoe_bodies[0].C[2], self.right_foot_shoe_bodies[0].C[2]])

        if self.use_settled_initial_states:
            q = self.init_states_candidates[np.random.randint(len(self.init_states_candidates))]

        self.robot_skeleton.q = q

        self.init_q = np.copy(self.robot_skeleton.q)

        self.initial_gyro = self.get_sim_bno55(supress_randomization=True)

        self.t = 0
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
            self.initial_heading = self.get_sim_bno55()[2]

        self.observation_buffer = []
        self.action_buffer = []

        if self.resample_MP and not self.supress_all_randomness:
            self.param_manager.resample_parameters()
            self.current_param = np.copy(self.param_manager.get_simulator_parameters())
            if self.range_robust > 0:
                lb = np.clip(self.current_param - self.range_robust, -0.05, 1.05)
                ub = np.clip(self.current_param + self.range_robust, -0.05, 1.05)
                self.current_param = np.random.uniform(lb, ub)

        if self.randomize_timestep and not self.supress_all_randomness:
            new_control_dt = self.control_interval + np.random.uniform(0.0, 0.01)
            default_fs = int(self.control_interval / self.sim_timestep)
            if not self.gyro_only_mode:
                self.frame_skip = np.random.randint(-5, 5) + default_fs

            self.dart_world.dt = new_control_dt / self.frame_skip

        self.obs_cache = []
        if self.resample_MP or self.mass_ratios[0] != 0:
            for i in range(len(self.body_parts)):
                self.body_parts[i].set_mass(self.orig_bodynode_masses[i] * self.mass_ratios[i])
            for i in range(len(self.body_parts)):
                self.body_parts[i].set_inertia(self.orig_bodynode_inertias[i] * self.inertia_ratios[i])

        self.dart_world.skeletons[2].q = [0,0,0, 100, 100, 100]

        if self.randomize_gravity_sch and not self.supress_all_randomness:
            self.gravity_sch = [[0.0, np.array([0,0,-9.81])]] # always start from normal gravity
            num_change = np.random.randint(1, 3) # number of gravity changes
            interv = self.interp_sch[-1][0] / num_change
            for i in range(num_change):
                rots = np.random.uniform(-0.5, 0.5, 2)
                self.gravity_sch.append([(i+1) * interv, np.array([np.cos(rots[0])*np.sin(rots[1]), np.sin(rots[0]), -np.cos(rots[0])*np.cos(rots[1])]) * 9.81])

        self.action_filter_cache = []
        for i in range(self.action_filtering):
            self.action_filter_cache.append(np.zeros(20))

        if self.randomize_obstacle and not self.soft_ground:
            horizontal_range = [0.6, 0.7]
            vertical_range = [-1.378, -1.378]
            sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
            sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1])
            self.dart_world.skeletons[0].bodynodes[1].shapenodes[0].set_offset([sampled_h, 0, sampled_v])
            self.dart_world.skeletons[0].bodynodes[1].shapenodes[1].set_offset([sampled_h, 0, sampled_v])

        if self.randomize_gyro_bias and not self.supress_all_randomness:
            self.gyro_bias = np.random.uniform(-0.1, 0.1, 2)

        if self.randomize_action_delay and not self.supress_all_randomness:
            self.action_delay = np.random.uniform(0.0, 0.03)

        self.perturb_force = np.array([0.0, 0.0, 0.0])
        self.cur_step = 0

        if self.butterworth_filter:
            self.action_filter.reset_filter()

        self.state_cache = [np.array(self.robot_skeleton.q)[6:], 0.0, 0.015 + np.random.normal(0.0, 0.005)]
        self.gyro_cache = [self.get_sim_bno55(), 0.0, 0.015 + np.random.normal(0.0, 0.005)]

        self.action_queue = []
        self.target = self.init_q[self.actuated_dof_ids]

        return self._get_obs()

    def resample_task(self):
        self.resample_MP = False

        self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()

        return np.copy(self.current_param)

    def set_task(self, task_params):
        self.param_manager.set_simulator_parameters(task_params)


    def get_robot_optimization_setup(self):
        # optimize the shape
        # dim = 10
        # upper_bound = np.ones(dim) * 0.01
        # lower_bound = -np.ones(dim) * 0.01

        # optimize the spring stiffness
        dim = 10
        upper_bound = np.ones(dim) * 4000
        lower_bound = np.ones(dim) * 300
        return dim, upper_bound, lower_bound

    def set_robot_optimization_parameters(self, parameters):
        # set offset of the shoe
        assert(len(parameters) == 10)
        # for i in range(10):
        #     for sn in self.robot_skeleton.bodynode('SHOE_PIECE'+str(i+1)+'_L').shapenodes:
        #         sn.set_offset([parameters[i], 0.0, 0.0])
        #     for sn in self.robot_skeleton.bodynode('SHOE_PIECE'+str(i+1)+'_R').shapenodes:
        #         sn.set_offset([-parameters[i], 0.0, 0.0])
        for i in range(len(self.left_shoe_dofs)):
            self.left_shoe_dofs[i].set_spring_stiffness(parameters[i])
        for i in range(len(self.right_shoe_dofs)):
            self.right_shoe_dofs[i].set_spring_stiffness(parameters[i])


    def viewer_setup(self):
        if not self.disableViewer:
            #self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.0
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0