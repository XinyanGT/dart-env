import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import pydart2.utils.transformations as transformations

class DartBridgePuzzle(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([23.0, 0.5, 0.0])
        self.action_scale = 100

        self.action_filtering = 0  # window size of filtering, 0 means no filtering
        self.action_filter_cache = []

        self.hierarchical = False
        if self.hierarchical:
            self.h_horizon = 20
            self.action_scale = 2.0
            self.num_way_point = 1

        self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])

        if self.hierarchical:
            self.control_bounds = np.array([[1.0] * (2*self.num_way_point), [-1.0] * (2*self.num_way_point)])

        dart_env.DartEnv.__init__(self, 'bridge_puzzle.skel', 2, 6, self.control_bounds, dt=0.01, disableViewer=True)

        #self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(5.0)
        for bn in self.dart_world.skeletons[1].bodynodes:
            bn.set_friction_coeff(0.3)
        for bn in self.dart_world.skeletons[-1].bodynodes:
            bn.set_friction_coeff(0.3)

        self.set_task([0.1])

        utils.EzPickle.__init__(self)

    def _rotate_bridge(self, angle):
        pos_rot_matrix = transformations.rotation_matrix(angle, [0, 1, 0])
        neg_rot_matrix = transformations.rotation_matrix(-angle, [0, 1, 0])
        for i in range(1, int(len(self.dart_world.skeletons[0].bodynodes[0].shapenodes)/2-2)):
            current_trans = self.dart_world.skeletons[0].bodynodes[0].shapenodes[i].relative_transform()
            current_trans[0:3, 0:3] = np.identity(3)
            current_trans = np.dot(current_trans, neg_rot_matrix if i % 2 == 0 else pos_rot_matrix)
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[i].set_relative_transform(current_trans)
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[int(i + len(self.dart_world.skeletons[0].bodynodes[0].shapenodes)/2)].set_relative_transform(current_trans)


    def resample_task(self):
        bridge_rotation = np.random.random() * 0.85
        self._rotate_bridge(bridge_rotation)
        return [bridge_rotation]

    def set_task(self, task_params):
        bridge_rotation = task_params[0]
        self._rotate_bridge(bridge_rotation)

    def step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        self.action_filter_cache.append(clamped_control)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0:
            clamped_control = np.mean(self.action_filter_cache, axis=0)

        if not self.hierarchical:
            tau = np.zeros(3)
            tau[[0,2]] = np.multiply(clamped_control, self.action_scale)
            self.do_simulation(tau, self.frame_skip)
        else:
            prev_pos = self.robot_skeleton.bodynodes[-1].com()[[0,2]]
            scaled_a = a * self.action_scale
            targets = []
            for i in range(self.num_way_point):
                new_pos = prev_pos + scaled_a[i*2:i*2+2]
                targets.append(new_pos)
                prev_pos = np.copy(new_pos)

            for i in range(self.h_horizon):
                tau = np.zeros(3)
                tvec = targets[int(i / (self.h_horizon * 1.0 / self.num_way_point))] - self.robot_skeleton.bodynodes[-1].com()[[0,2]]
                tvec /= np.linalg.norm(tvec)
                tau[[0,2]] = tvec * 50
                self.do_simulation(tau, self.frame_skip)
            self.dart_world.skeletons[-2].q = np.array([targets[0][0], 0.0, targets[0][1]])

        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        reward_dist = - np.linalg.norm(vec) * 0.65
        reward_ctrl = - np.square(a).sum()*0.1
        reward = reward_dist + reward_ctrl + 15

        if np.linalg.norm(vec) < 0.8:
            reward += 10

        #    reward -= 15

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        if self.robot_skeleton.bodynodes[-1].com()[1] < -0.0 or self.robot_skeleton.bodynodes[-1].com()[1] > 0.5:
            done = True

        return ob, reward, done, {'done_return':done}

    def _get_obs(self):
        pos = self.robot_skeleton.bodynodes[-1].C
        vel = self.robot_skeleton.bodynodes[-1].dC
        return np.array([pos[0], pos[2], vel[0], vel[2], self.dart_world.skeletons[-2].C[0], self.dart_world.skeletons[-2].C[2]])

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qpos[1] = 0.0
        qvel[1] = 0.0
        self.set_state(qpos, qvel)

        if not self.hierarchical:
            self.dart_world.skeletons[-2].q = np.array([1000, 10, 10])
        self.dart_world.skeletons[-2].q = self.target

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -40.0
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0
