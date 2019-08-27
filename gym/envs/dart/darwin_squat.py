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

from pydart2.utils.transformations import euler_from_matrix

class DartDarwinSquatEnv(dart_env.DartEnv, utils.EzPickle):
    WALK, SQUATSTAND, STEPPING, FALLING, HOP, CRAWL, STRANGEWALK, KUNGFU, BONGOBOARD = list(range(9))

    def __init__(self):

        obs_dim = 40

        self.root_input = True
        self.include_heading = True
        self.transition_input = False  # whether to input transition bit
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
        self.fallstate_input = False

        self.action_filtering = 5 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.butterworth_filter = True

        self.future_ref_pose = 0  # step of future trajectories as input

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
            #self.param_manager.activated_param.remove(self.param_manager.NEURAL_MOTOR)
        else:
            self.param_manager = darwinSquatParamManager(self)

        self.use_DCMotor = False
        self.use_spd = False
        self.same_gain_model = False # whether to use the model optimized with all motors to have the same gains
        self.NN_motor = False
        self.NN_motor_hid_size = 5 # assume one layer
        self.NN_motor_parameters = [np.random.random((2, self.NN_motor_hid_size)), np.random.random(self.NN_motor_hid_size),
                                    np.random.random((self.NN_motor_hid_size, 2)), np.random.random(2)]
        self.NN_motor_bound = [[200.0, 1.0], [0.0, 0.0]]

        self.use_settled_initial_states = False
        self.limited_joint_vel = True
        self.joint_vel_limit = 20000.0
        self.train_UP = False
        self.noisy_input = True
        self.resample_MP = False
        self.range_robust = 0.25 # std to sample at each step
        self.randomize_timestep = True
        self.load_keyframe_from_file = True
        self.randomize_gravity_sch = False
        self.randomize_obstacle = True
        self.randomize_gyro_bias = True
        self.gyro_bias = [0.0, 0.0]
        self.height_drop_threshold = 0.8    # terminate if com height drops for this amount
        self.orientation_threshold = 1.0    # terminate if body rotates for this amount
        self.control_interval = 0.035  # control every 50 ms
        self.sim_timestep = 0.002
        self.forward_reward = 30.0
        self.velocity_clip = 0.3
        self.contact_pen = 0.0
        self.kp = None
        self.kd = None
        self.kc = None

        self.soft_ground = False
        self.task_mode = self.STEPPING
        self.side_walk = False

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

        self.heuristic_balance = None#[0.02085987, 0.05597697, 0.00031102, 0.00471298]


        # single leg stand
        #self.permitted_contact_ids = [-7, -8] #[-1, -2, -7, -8]
        #self.init_root_pert = np.array([0.0, 1.2, 0.0, 0.0, 0.0, 0.0])

        # crawl
        if self.task_mode == self.CRAWL:
            self.permitted_contact_ids = [-1, -2, -7, -8, 6, 11]  # [-1, -2, -7, -8]
            self.init_root_pert = np.array([0.0, 1.6, 0.0, 0.0, 0.0, 0.0])
        else:
            # normal pose
            self.permitted_contact_ids = [-1, -2, -7, -8, 6, 11]
            self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        self.initialize_falling = False # initialize darwin to have large ang vel so that it falls

        if self.side_walk:
            self.init_root_pert = np.array([0.0, 0., -1.57, 0.0, 0.0, 0.0])

        # cartwheel
        #self.permitted_contact_ids = [-1, -2, -7, -8, 6, 11]
        #self.init_root_pert = np.array([-0.8, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.delta_angle_scale = 0.3

        self.alive_bonus = 1.0
        self.energy_weight = 0.0
        self.work_weight = 0.0
        self.pose_weight = 0.0
        self.upright_weight = 0.0
        self.comvel_pen = 0.0
        self.compos_pen = 0.0
        self.compos_range = 0.5

        self.cur_step = 0

        self.torqueLimits = 10.0
        if self.same_gain_model:
            self.torqueLimits = 2.5

        self.t = 0
        self.target = np.zeros(26, )
        self.tau = np.zeros(26, )
        self.init = np.zeros(26, )

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
             -23, -24, -25, -20, -21, -22, -26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33])
        act_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])
        self.obs_perm = np.copy(obs_perm_base)

        if self.root_input:
            beginid = len(obs_perm_base)
            if self.include_heading:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2,  -beginid - 3, beginid + 4, -beginid - 5]])
            else:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2, beginid + 3]])
        if self.fallstate_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1]])
        if self.transition_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid]])
        if self.task_mode == self.BONGOBOARD:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid + 1, -beginid - 2]])
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
        if self.task_mode != self.BONGOBOARD:
            model_file_list.remove('darwinmodel/bongo_board.urdf')
        if self.soft_ground:
            model_file_list[0] = 'darwinmodel/soft_ground.skel'
            model_file_list.insert(4, 'darwinmodel/soft_ground.urdf')
        dart_env.DartEnv.__init__(self, model_file_list, int(self.control_interval / self.sim_timestep), obs_dim,
                                  self.control_bounds, dt=self.sim_timestep, disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")
        self.orig_bodynode_masses = [bn.mass() for bn in self.robot_skeleton.bodynodes]

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

        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(1.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(1.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(2.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(2.0)

        self.add_perturbation = True
        self.perturbation_parameters = [50, 5, 50, [2, 4], 1]  # begin step, duration, interval, magnitude, bodyid

        for i in range(6, self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            if not self.same_gain_model:
                j.set_damping_coefficient(0.515)
                j.set_coulomb_friction(0.0)
            else:
                j.set_damping_coefficient(0.26498)
                #j.set_coulomb_friction(12.3019)
                j.set_spring_stiffness(0.6825)

        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(6, self.robot_skeleton.ndofs):
            self.robot_skeleton.dofs[i].set_position_lower_limit(JOINT_LOW_BOUND[i - 6] - 0.01)
            self.robot_skeleton.dofs[i].set_position_upper_limit(JOINT_UP_BOUND[i - 6] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        if self.task_mode == self.BONGOBOARD:
            self.permitted_contact_bodies += [b for b in self.dart_world.skeletons[4].bodynodes]

        self.initial_local_coms = [b.local_com() for b in self.robot_skeleton.bodynodes]

        ################# temp code, ugly for now, should fix later ###################################
        if self.use_sysid_model:
            self.param_manager.controllable_param = [self.param_manager.KP_RATIO, self.param_manager.KD_RATIO,
                                                     self.param_manager.NEURAL_MOTOR, self.param_manager.TORQUE_LIM,
                                                     self.param_manager.COM_OFFSET, self.param_manager.GROUND_FRICTION]
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

        if self.butterworth_filter:
            self.action_filter = ActionFilter(self.act_dim, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=3.0)

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
        self.delta_angle_scale = 0.2
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

        '''q = self.robot_skeleton.q
        q[6:] = p
        self.robot_skeleton.q = q
        # gradually move the robot down until it touches the board
        while True:
            self.dart_world.check_collision()
            self.contacts = self.dart_world.collision_result.contacts

            feet = [self.robot_skeleton.bodynode('MP_ANKLE2_L'), self.robot_skeleton.bodynode('MP_ANKLE2_R')]
            board = self.dart_world.skeletons[4].bodynodes[1]

            fb_contact_count = 0
            for contact in self.contacts:
                if contact.bodynode1 in feet and contact.bodynode2 == board:
                    fb_contact_count += 1
                if contact.bodynode1 == board and contact.bodynode2 in feet:
                    fb_contact_count += 1
            if fb_contact_count > 3:
                break

            q = self.robot_skeleton.q
            q[5] -= 0.003
            self.robot_skeleton.q = q

        for contact in self.contacts:
            if (contact.bodynode1 in feet and contact.bodynode2 == board) or \
                (contact.bodynode1 == board and contact.bodynode2 in feet):
                bc = BallJointConstraint(contact.bodynode1, contact.bodynode2, contact.point)
                bc.add_to_world(self.dart_world)'''


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


    def adjust_root(self): # adjust root dof such that foot is roughly flat
        q = self.robot_skeleton.q
        q[1] += -1.57 - np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_ANKLE2_L').T[0:3, 0:3], 'sxyz'))[1]
        q[5] += -0.335 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-7].C[2]])
        self.robot_skeleton.q = q


    def get_sim_bno55(self, supress_randomization=False):
        # simulate bno55 reading
        tinv = np.linalg.inv(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3])
        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]
        langvel = np.dot(tinv, angvel)

        euler = np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz'))
        if self.randomize_gyro_bias and not supress_randomization:
            euler[0:2] += self.gyro_bias
            # add noise
            euler += np.random.uniform(-0.01, 0.01, 3)
            langvel += np.random.uniform(-0.1, 0.1, 3)
        return np.array([euler[0], euler[1]-0.075, euler[2], langvel[0], langvel[1], langvel[2]])

    def falling_state(self): # detect if it's falling fwd/bwd or left/right
        gyro = self.get_sim_bno55()
        fall_flags = [0, 0]
        if np.abs(gyro[0]) > 0.5:
            fall_flags[0] = np.sign(gyro[0])
        if np.abs(gyro[1]) > 0.5:
            fall_flags[1] = np.sign(gyro[1])
        return fall_flags


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

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        self.ref_target = self.get_ref_pose(self.t)

        clamped_control = self.action_filter.filter_action(clamped_control)

        self.target[6:] = self.ref_target + clamped_control * self.delta_angle_scale

        if self.heuristic_balance is not None: # apply heuristic balance control
            gyro = self.get_sim_bno55()
            # hip
            gyro_lr = gyro[3]
            gyro_fb = gyro[4]

            gyro_lr = (gyro[0] - self.initial_gyro[0]) * 5
            gyro_fb = (gyro[1] - self.initial_gyro[1]) * 5
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

        self.target[6:] = np.clip(self.target[6:], JOINT_LOW_BOUND, JOINT_UP_BOUND)

        dup_pos = np.copy(self.target)
        dup_pos[4] = 0.8
        self.dupSkel.set_positions(dup_pos)
        self.dupSkel.set_velocities(self.target*0)

        if self.add_perturbation and self.cur_step >= self.perturbation_parameters[0]:
            if (self.cur_step - self.perturbation_parameters[0]) % self.perturbation_parameters[2] == 0:
                force_mag = np.random.uniform(self.perturbation_parameters[3][0], self.perturbation_parameters[3][1])
                force_dir = np.concatenate([np.random.uniform(-1, 1, 2), [0.0]])
                self.perturb_force = force_dir / np.linalg.norm(force_dir) * force_mag
            elif (self.cur_step - self.perturbation_parameters[0]) % self.perturbation_parameters[2] > self.perturbation_parameters[1]:
                self.perturb_force *= 0

        for i in range(self.frame_skip):
            self.tau[6:] = self.PID()
            self.tau[0:6] *= 0.0

            if self.task_mode == self.BONGOBOARD:
                if self.t < self.assist_timeout:
                    force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.current_assist[0], 1)
                    self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([0, force, 0]))

                    force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.current_assist[1], 0)
                    self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([force, 0, 0]))

            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[4]].add_ext_force(self.perturb_force)
            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()

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

            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:]

            if self.limited_joint_vel:
                tau[(np.abs(self.robot_skeleton.dq) > self.joint_vel_limit) * (
                        np.sign(self.robot_skeleton.dq) == np.sign(tau))] = 0
        else:
            if self.kp is not None:
                kp = self.kp
            else:
                if not self.same_gain_model:
                    kp = np.array([2.1+10, 1.79+10, 4.93+10,
                               2.0+10, 2.02+10, 1.98+10,
                               2.2+10, 2.06+10,
                                   60, 60, 60, 60, 153, 102,
                                   60, 60, 60, 60, 153, 102])

                    kp[0:6] *= self.kp_ratios[0]
                    kp[7:8] *= self.kp_ratios[1]
                    kp[8:11] *= self.kp_ratios[2]
                    kp[14:17] *= self.kp_ratios[2]
                    kp[11] *= self.kp_ratios[3]
                    kp[17] *= self.kp_ratios[3]
                    kp[12:14] *= self.kp_ratios[4]
                    kp[18:20] *= self.kp_ratios[4]
                else:
                    kp = 54.019

            if self.kd is not None:
                kd = self.kd
            else:
                if not self.same_gain_model:
                    kd = np.array([0.021, 0.023, 0.022,
                               0.025, 0.021, 0.026,
                               0.028, 0.0213
                               ,0.2, 0.2, 0.2, 0.2, 0.02, 0.02,
                               0.2, 0.2, 0.2, 0.2, 0.02, 0.02])
                    kd[0:6] *= self.kd_ratios[0]
                    kd[7:8] *= self.kd_ratios[1]
                    kd[8:11] *= self.kd_ratios[2]
                    kd[14:17] *= self.kd_ratios[2]
                    kd[11] *= self.kd_ratios[3]
                    kd[17] *= self.kd_ratios[3]
                    kd[12:14] *= self.kd_ratios[4]
                    kd[18:20] *= self.kd_ratios[4]
                else:
                    kd = 2.5002

            if self.kc is not None:
                kc = self.kc
            else:
                kc = 0.0

            q = self.robot_skeleton.q
            qdot = self.robot_skeleton.dq
            tau = np.zeros(26, )

            # velocity limiting
            '''lamb = kp / kd
            x_tilde = q[6:] - self.target[6:]
            vmax = 5.75 * self.sim_dt / lamb
            sat = vmax / (lamb * np.abs(x_tilde))
            scale = np.ones(20)
            if np.any(sat < 1):
                index = np.argmin(sat)
                unclipped = kp * x_tilde[index]
                clipped = kd * vmax * np.sign(x_tilde[index])
                scale = np.ones(20) * clipped / unclipped
                scale[index] = 1
            tau[6:] = -kd * (qdot[6:] + np.clip(sat / scale, 0, 1) *
                           scale * lamb * x_tilde)'''

            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:] - kc * np.sign(qdot[6:])

            if self.limited_joint_vel:
                tau[(np.abs(self.robot_skeleton.dq) > self.joint_vel_limit) * (np.sign(self.robot_skeleton.dq) == np.sign(tau))] = 0

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

        # modify gravity according to schedule
        grav = self.gravity_sch[0][1]

        for i in range(len(self.gravity_sch) - 1):
            if self.t >= self.gravity_sch[i][0] and self.t < self.gravity_sch[i + 1][0]:
                ratio = (self.t - self.gravity_sch[i][0]) / (self.gravity_sch[i + 1][0] - self.gravity_sch[i][0])
                grav = ratio * self.gravity_sch[i + 1][1] + (1 - ratio) * self.gravity_sch[i][1]
        if self.t > self.gravity_sch[-1][0]:
            grav = self.gravity_sch[-1][1]
        self.dart_world.set_gravity(grav)

        self.action_buffer.append(np.copy(a))
        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        upright_rew = np.abs(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz')[1])

        pose_math_rew = np.sum(
            np.abs(np.array(self.ref_target - self.robot_skeleton.q[6:])) ** 2)
        reward = -self.energy_weight * np.sum(
            self.tau ** 2) + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.sum(np.abs(self.tau * self.robot_skeleton.dq))
        # reward -= 2.0 * np.abs(self.robot_skeleton.dC[1])
        reward -= self.upright_weight * upright_rew
        reward += self.forward_reward * np.clip((xpos_after - xpos_before) / self.dt, -self.velocity_clip, self.velocity_clip)

        reward -= self.comvel_pen * np.linalg.norm(self.robot_skeleton.dC)
        reward -= self.compos_pen * np.linalg.norm(self.init_q[3:6] - self.robot_skeleton.q[3:6])

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all())

        if np.any(np.abs(np.array(self.robot_skeleton.q)[0:3]) > self.orientation_threshold):
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

        if not self.paused or self.t == 0:
            self.t += self.dt * 1.0
            self.cur_step += 1

        ob = self._get_obs()

        # move the obstacle forward when the robot has passed it
        if self.randomize_obstacle and not self.soft_ground:
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
        state = np.concatenate([self.robot_skeleton.q[6:], self.robot_skeleton.dq[6:]])
        if self.multipos_obs > 0:
            state = self.robot_skeleton.q[6:]

            self.obs_cache.append(state)
            while len(self.obs_cache) < self.multipos_obs:
                self.obs_cache.append(state)
            if len(self.obs_cache) > self.multipos_obs:
                self.obs_cache.pop(0)

            state = np.concatenate(self.obs_cache)

        for i in range(self.future_ref_pose):
            state = np.concatenate([state, self.get_ref_pose(self.t + self.dt * (i+1))])

        if self.root_input:
            gyro = self.get_sim_bno55()
            if not self.include_heading:
                gyro = np.array([gyro[0], gyro[1], self.last_root[0], self.last_root[1]])
                self.last_root = [gyro[0], gyro[1]]
            else:
                adjusted_heading = (gyro[2] - self.initial_heading) % (2*np.pi)
                adjusted_heading = adjusted_heading - 2*np.pi if adjusted_heading > np.pi else adjusted_heading
                gyro = np.array([gyro[0], gyro[1], adjusted_heading, self.last_root[0], self.last_root[1], self.last_root[2]])
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

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])

        return final_obs

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(
            self.robot_skeleton.ndofs) + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005,
                                                               size=self.robot_skeleton.ndofs)  # np.zeros(self.robot_skeleton.ndofs) #

        #if self.task_mode != self.BONGOBOARD:
        #    qpos[1] += np.random.uniform(-0.1, 0.1)

        # LEFT HAND
        if self.interp_sch is not None:
            qpos[6:] = np.clip(self.interp_sch[0][1], JOINT_LOW_BOUND, JOINT_UP_BOUND)
        else:
            qpos[6:] = np.clip(0.5 * (pose_squat_rad + pose_stand_rad), JOINT_LOW_BOUND, JOINT_UP_BOUND)

        #qpos[6:] += np.random.uniform(low=-0.01, high=0.01, size=20)
        # self.target = qpos
        self.count = 0
        qpos[0:6] += self.init_root_pert

        if self.initialize_falling:
            qvel[0] = np.random.uniform(-2.0, 2.0)
            qvel[1] = np.random.uniform(-2.0, 2.0)

        self.set_state(qpos, qvel)

        q = self.robot_skeleton.q
        if self.task_mode == self.CRAWL:
            q[5] += -0.3 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-7].C[2]])
        elif self.task_mode == self.BONGOBOARD:
            q[5] += -0.25 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-7].C[2]])
        else:
            q[5] += -0.335 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-7].C[2]])

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

        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = np.copy(self.param_manager.get_simulator_parameters())
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

        if self.randomize_gravity_sch:
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

        if self.randomize_gyro_bias:
            self.gyro_bias = np.random.uniform(-0.03, 0.03, 2)

        self.perturb_force = np.array([0.0, 0.0, 0.0])
        self.cur_step = 0

        if self.butterworth_filter:
            self.action_filter.reset_filter()

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            #self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.0
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0