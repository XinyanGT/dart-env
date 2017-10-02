__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os


class DartWalker3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*15,[-1.0]*15])
        self.action_scale = np.array([150.0]*15)
        self.action_scale[[-1,-2,-7,-8]] = 60
        self.action_scale[[0, 1, 2]] = 100
        #self.action_scale = np.array([40,80,150, 150,60,80,120,60,60, 150,60,80,120,60,60])
        self.action_scale = np.array([200, 200, 200, 150, 60, 80, 150, 60, 60, 150, 60, 80, 150, 60, 60])
        obs_dim = 41

        self.t = 0
        self.target_vel = 0.9
        self.init_push = False
        self.enforce_target_vel = True
        self.hard_enforce = True
        self.treadmill = True
        self.base_policy = None
        modelpath = os.path.join(os.path.dirname(__file__), "models")
        self.cur_step = 0
        self.stepwise_rewards = []
        #self.base_policy = joblib.load(os.path.join(modelpath, 'walker3d_init/init_policy_forward_newlimit.pkl'))

        if self.base_policy is not None:
            # when training balance delta function
            self.base_action_indices = [0,1,2,3, 4,5,6,7, 8,9, 10,11,12,13 ,14]
            self.deltacontrol_bounds = np.array([[2.0]*8,[-2.0]*8])
            self.delta_action_indices = [0,1, 4,5, 8, 10,11, 14]

            '''# when training forward delta function
            self.base_action_indices = [0,1, 4,5, 8, 10,11, 14]
            self.deltacontrol_bounds = np.array([[1.0]*15,[-1.0]*15])
            self.delta_action_indices = [0,1,2,3, 4,5,6,7, 8,9, 10,11,12,13 ,14]'''

            dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 15, obs_dim, self.deltacontrol_bounds, disableViewer=False)
        elif self.treadmill:
            dart_env.DartEnv.__init__(self, 'walker3d_treadmill.skel', 15, obs_dim, self.control_bounds, disableViewer=True)
        else:
            dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True)

        #self.dart_world.set_collision_detector(3)
        self.robot_skeleton.set_self_collision_check(True)

        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)

        utils.EzPickle.__init__(self)

    def advance(self, a):
        if self.base_policy is not None:
            full_a = np.zeros(np.max([len(self.base_action_indices), len(self.delta_action_indices)]))
            base_action = self.base_policy.get_action(self._get_obs())[1]['mean']
            full_a[self.base_action_indices] += base_action
            full_a[self.delta_action_indices] += a
            a = np.copy(full_a)
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        if self.enforce_target_vel:
            if not self.hard_enforce:
                if np.abs(self.robot_skeleton.dq[0] - self.target_vel) > 0.1:
                    tau[0] = 50*(self.target_vel - self.robot_skeleton.dq[0])
            if self.hard_enforce and self.treadmill:
                current_dq_tread = self.dart_world.skeletons[0].dq
                current_dq_tread[0] = -self.target_vel
                self.dart_world.skeletons[0].dq = current_dq_tread
            elif self.hard_enforce:
                current_dq = self.robot_skeleton.dq
                current_dq[0] = self.target_vel
                self.robot_skeleton.dq = current_dq
        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.advance(np.copy(a))

        posafter = self.robot_skeleton.bodynodes[1].com()[0]
        height = self.robot_skeleton.bodynodes[1].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[1].com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-3, -9]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05 and a[j] < 0:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05 and a[j] > 0:
                joint_limit_penalty += abs(1.5)
        joint_limit_penalty *= 0

        actuator_pen_multiplier = np.ones(len(a)) # penalize actuation more if it is trying to push against limit
        '''for j in range(1,len(a)+1):
            if (self.robot_skeleton.q_lower[-j] - self.robot_skeleton.q[-j]) > -0.05 and a[-j] < 0:
                actuator_pen_multiplier[-j] = 100
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[-j]) < 0.05 and a[-j] > 0:
                actuator_pen_multiplier[-j] = 100'''

        if self.base_policy is None:
            alive_bonus = 3.0
            vel = (posafter - posbefore) / self.dt
            if not self.treadmill:
                vel_rew = -2*(np.abs(self.target_vel - vel))#1.0 * (posafter - posbefore) / self.dt
            else:
                vel_rew = 0.0 
            #action_pen = 5e-1 * (np.square(a)* actuator_pen_multiplier).sum()
            action_pen =5e-1 * np.abs(a).sum()
            #action_pen = 5e-3 * np.sum(np.square(a)* self.robot_skeleton.dq[6:]* actuator_pen_multiplier)
            deviation_pen = 1e-3 * abs(side_deviation)
            reward = vel_rew + alive_bonus - action_pen - deviation_pen
        else:
            alive_bonus = 2.0
            vel_rew = 1.0 * (posafter - posbefore) / self.dt
            action_pen = 1e-2 * np.square(a).sum()
            joint_pen = 2e-1 * joint_limit_penalty
            deviation_pen = 1e-3 * abs(side_deviation)
            reward = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen

        #reward -= 1e-7 * total_force_mag

        #div = self.get_div()
        #reward -= 1e-1 * np.min([(div**2), 10])

        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.0) and (height < 2.0) and (abs(ang_cos_uwd) < 1.0) and (abs(ang_cos_fwd) < 1.0)
                    and np.abs(angle) < 0.8 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        self.stepwise_rewards.append(reward)

        if self.treadmill:
            if np.abs(self.robot_skeleton.q[0]) > 0.3:
                done = True

        ob = self._get_obs()

        foot1_com = self.robot_skeleton.bodynode('h_foot').com()
        foot2_com = self.robot_skeleton.bodynode('h_foot_left').com()
        robot_com = self.robot_skeleton.com()
        com_foot_offset1 = robot_com - foot1_com
        com_foot_offset2 = robot_com - foot2_com
 
        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'deviation_pen':deviation_pen, 'aux_pred':np.hstack([com_foot_offset1, com_foot_offset2, [reward]]), 'done_return':done, 'dyn_model_id':0, 'state_index':0}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10),
            #[self.t]
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        sign = np.sign(np.random.uniform(-1, 1))
        #qpos[9] = sign * self.np_random.uniform(low=0.1, high=0.15, size=1)
        #qpos[15] = -sign * self.np_random.uniform(low=0.1, high=0.15, size=1)
        qpos[11] = sign * self.np_random.uniform(low=-0.1, high=-0.05, size=1)
        qpos[17] = -sign * self.np_random.uniform(low=0.05, high=0.1, size=1)
        if self.init_push:
            qpos[0] = self.target_vel
        self.set_state(qpos, qvel)
        self.t = 0
        self.cur_step = 0
        self.stepwise_rewards = []

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5
