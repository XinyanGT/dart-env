#Author: Alexander Clegg (alexanderwclegg@gmail.com)

#----------------------------------------------------------
# Base Env for all human/iiwa dressing interaction tasks
#----------------------------------------------------------

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import time
import random
from gym import utils
import joblib
import shutil, errno

from gym.envs.dart.static_window import *
from gym.envs.dart.norender_window import *
from gym.envs.dart.static_cloth_window import *

import pybullet as p

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))

import pydart2.joint as Joint
import pydart2.collision_result as CollisionResult

try:
    import pyPhysX as pyphysx
    pyphysx.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pyphysx.)".format(e))

from pyPhysX.clothfeature import * #Plane and Feature
import pyPhysX.meshgraph as meshgraph
from pyPhysX.colors import *

def LERP(p0, p1, t):
    return p0 + (p1 - p0) * t

class RewardManager:
    def __init__(self, env):
        self.env = env
        self.reward_limits = (0,0) #max and min reward
        self.reward_terms = [] #list of active reward terms
        self.previous_reward = 0
        self.rewards_data = renderUtils.RewardsData([], [], [], [])
        self.render_rewards_data = True

    def computeReward(self):
        reward = 0.0
        reward_record = []
        for term in self.reward_terms:
            term_reward = term.evaluateReward()
            reward_record.append(term_reward/term.weight)
            reward += term_reward
        self.rewards_data.update(rewards=reward_record)
        self.previous_reward = reward
        if math.isinf(reward) or math.isnan(reward):
            print("REWARD IS INVALID!!! " + str(reward))
            return self.reward_limits[0]
        return reward

    def addTerm(self, term):
        #add a new reward term to the set
        self.reward_terms.append(term)
        self.reward_limits = (self.reward_limits[0]+term.min*term.weight, self.reward_limits[1]+term.max*term.weight)
        self.rewards_data.addReward(label=term.name, rmin=term.min, rmax=term.max, rval=0, rweight=term.weight)

    def draw(self):
        for reward_term in self.reward_terms:
            reward_term.draw()
        if self.render_rewards_data and self.env.detail_render:
            m_viewport = self.env.viewer.viewport
            self.rewards_data.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15], dimensions=[400, -m_viewport[3] + 30])

    def reset(self):
        self.rewards_data.reset()
        for term in self.reward_terms:
            term.reset()

class RewardTerm:
    #superclass for reward terms defining the structure
    def __init__(self, name, weight, min, max):
        self.name = name
        self.weight = weight
        self.min = min
        self.max = max
        self.previous_evaluation = 0.0

    def evaluateReward(self):
        #override this in subclasses
        self.previous_evaluation = 0.0
        return self.previous_evaluation

    def draw(self):
        pass

    def reset(self):
        pass

#TODO: convert various reward terms into class structures for re-use
#TODO: oracle displacement

class RestPoseRewardTerm(RewardTerm):
    #skeleton rest pose of any weight pose subset
    def __init__(self, skel, pose, weights, name="rest pose"):
        upper_lims = skel.position_upper_limits()
        lower_lims = skel.position_lower_limits()
        j_range = upper_lims - lower_lims
        for rix,r in enumerate(j_range):
            if math.isinf(r) or math.isnan(r):
                j_range[rix] = 6.28 #360 rotation
        min_reward = -np.sum(j_range * weights)
        #print("min_reward = " + str(min_reward))
        RewardTerm.__init__(self, name=name, weight=1.0, min=min_reward, max=0)
        self.rest_pose = np.array(pose)
        self.rest_pose_weights = np.array(weights) #also acts as a mask w/ 0's
        self.skel = skel

    def evaluateReward(self):
        self.previous_evaluation = -np.sum(np.absolute(self.rest_pose-self.skel.q)*self.rest_pose_weights)
        return self.previous_evaluation

class LimbProgressRewardTerm(RewardTerm):
    #progress of a limb through a garment feature
    def __init__(self, dressing_target, continuous=True, terminal=False, success_threshold=0.825, name="limb progress", weight=1.0, min=-2.0, max=1.0):
        self.dressing_target = dressing_target
        self.continuous = continuous
        self.terminal = terminal
        self.success_threshold=success_threshold
        self.previous_progress = 0
        RewardTerm.__init__(self, name=name, weight=weight, min=min, max=max)

    def evaluateReward(self):
        self.previous_evaluation = 0
        self.previous_progress = self.dressing_target.getLimbProgress()
        if self.continuous:
            #if not terminal, give reward for progress up to the threshold value
            if self.previous_progress > 0:
                self.previous_evaluation += (min(self.previous_progress, self.success_threshold)/self.success_threshold)*self.weight
            else:
                self.previous_evaluation += self.previous_progress*self.weight
        if self.terminal:
            #if terminal, only give binary reward for reaching threshold
            if self.previous_progress >= self.success_threshold:
                self.previous_evaluation += self.weight
        return self.previous_evaluation

    def draw(self):
        #render the threshold
        limb = pyutils.limbFromNodeSequence(self.dressing_target.skel, nodes=self.dressing_target.limb_sequence, offset=self.dressing_target.distal_offset)
        p,v = pyutils.getLimbPointVec(limb, 1.0-self.success_threshold)
        point_plane = Plane(org=p, normal=v/np.linalg.norm(v))
        point_plane.draw(0.1)

class GeodesicContactRewardTerm(RewardTerm):
    #reward for touching the cloth close to the target feature
    def __init__(self, sensor_index, env, separated_mesh, dressing_target, name="geodesic contact", weight=1.0, min=0, max=1.0):
        self.sensor_index = sensor_index
        self.env = env
        self.separated_mesh = separated_mesh
        self.dressing_target = dressing_target #used to determine if the max should be returned (this must be updated during simulation)
        RewardTerm.__init__(self, name=name, weight=weight, min=min, max=max)

    def evaluateReward(self):
        self.previous_evaluation = 0

        if self.dressing_target.previous_evaluation > 0:
            #already in the feature, max reward here
            self.previous_evaluation = self.weight
        else:
            contactInfo = pyutils.getContactIXGeoSide(sensorix=self.sensor_index, clothscene=self.env.clothScene, meshgraph=self.separated_mesh)
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)
                self.previous_evaluation = 1.0-(avgContactGeodesic / self.separated_mesh.maxGeo)
                self.previous_evaluation *= self.weight

        return self.previous_evaluation

class ClothDeformationRewardTerm(RewardTerm):
    #penalty for deforming the garment
    def __init__(self, env, tanh_params=(2, 0.7, 15), name="cloth deformation", weight=1.0, min=-1.0, max=0.0):
        self.env = env
        self.tanh_params = tanh_params
        RewardTerm.__init__(self, name=name, weight=weight, min=min, max=max)

    def evaluateReward(self):
        deformation = self.env.clothScene.getMaxDeformationRatio(0)
        #params = (z,s,t)
        #reward = tanh(s*t - s*deformation + z)-1
        z = self.tanh_params[0]
        s = self.tanh_params[1]
        t = self.tanh_params[2]
        #note, this tanh will return [0,-2], so divide by 2 to get min = -weight
        self.previous_evaluation = (math.tanh(s*t - s*deformation + z) - 1) * self.weight * 0.5
        return self.previous_evaluation

class HumanContactRewardTerm(RewardTerm):
    #penalty for max avg force percieved by a single human sensor
    def __init__(self, env, tanh_params=(2, 0.2, 9.5), name="human force penalty", weight=1.0, min=-1.0, max=0.0):
        self.env = env
        self.tanh_params = tanh_params
        RewardTerm.__init__(self, name=name, weight=weight, min=min, max=max)
        self.max_avg_force = 0
        self.true_max_avg_force = 0

    def evaluateReward(self):
        cumulative_readings = np.zeros(self.env.haptic_sensor_data["num_sensors"]*3)
        if self.env.haptic_sensor_data["cloth_steps"] > 0:
            #average contact forces over the cloth readings
            cumulative_readings += self.env.haptic_sensor_data["cloth_data"]/self.env.haptic_sensor_data["cloth_steps"]
        if self.env.haptic_sensor_data["rigid_steps"] > 0:
            #average contact forces over the rigid readings
            cumulative_readings += self.env.haptic_sensor_data["rigid_data"] / self.env.haptic_sensor_data["rigid_steps"]

        max_avg_force = 0
        for s in range(self.env.haptic_sensor_data["num_sensors"]):
            f = cumulative_readings[s*3:s*3+3]
            f_mag = np.linalg.norm(f)
            max_avg_force = max(f_mag, max_avg_force)

        #params = (z,s,t)
        #reward = tanh(s*t - s*max_avg_force + z)-1
        z = self.tanh_params[0]
        s = self.tanh_params[1]
        t = self.tanh_params[2]
        # note, this tanh will return [0,-2], so divide by 2 to get min = -weight
        self.previous_evaluation = (math.tanh(s*t - s*max_avg_force + z) - 1) * self.weight * 0.5
        self.max_avg_force = max_avg_force
        self.true_max_avg_force = max(self.true_max_avg_force, self.max_avg_force)
        return self.previous_evaluation

    def draw(self):
        self.env.text_queue.append("max_avg_force: " + str(self.max_avg_force) + " -> reward: " + str(self.previous_evaluation))
        self.env.text_queue.append("true_max_avg_force: " + str(self.true_max_avg_force))

    def reset(self):
        self.true_max_avg_force = 0

class ObservationManager:
    def __init__(self):
        self.obs_features = []  # list of active reward terms
        self.obs_size = 0
        self.prev_obs = np.zeros(0)

    def getObs(self):
        obs = np.zeros(0)
        for feature in self.obs_features:
            obs = np.concatenate([obs, feature.getObs()])
        self.prev_obs = np.array(obs)
        return obs

    def addObsFeature(self, feature):
        self.obs_features.append(feature)
        self.obs_size += feature.obs_dim

    def draw(self):
        for feature in self.obs_features:
            if feature.render:
                feature.draw()

    def reset(self):
        for feature in self.obs_features:
            feature.reset()

class ObservationFeature:
    #superclass wrapper for Observation Features
    def __init__(self, name, dim, render=False):
        self.obs_dim = dim
        self.name = name
        self.render = render

    def getObs(self):
        #override this
        return np.zeros(0)

    def draw(self):
        pass

    def reset(self):
        pass

class ProprioceptionObsFeature(ObservationFeature):
    #cos(q), sin(q), dq
    def __init__(self, skel, start_dof=0, name="proprioception", render=False):
        self.start_dof = start_dof
        ObservationFeature.__init__(self, name=name, dim=(len(skel.q)-start_dof)*3, render=render)
        self.skel = skel

    def getObs(self):
        obs = np.concatenate([np.cos(self.skel.q[self.start_dof:]),np.sin(self.skel.q[self.start_dof:]), self.skel.dq[self.start_dof:]])
        return obs

class HumanHapticObsFeature(ObservationFeature):
    #human haptic sensor observation
    def __init__(self, env, mag_scale=30, contact_ids=True, name="haptics", render=False):
        dim = env.haptic_sensor_data["num_sensors"]*3
        if contact_ids:
            dim += env.haptic_sensor_data["num_sensors"]
        ObservationFeature.__init__(self, name=name, dim=dim, render=render)
        self.contact_ids = contact_ids
        self.env = env
        self.prev_obs = np.zeros(dim)
        self.mag_scale = mag_scale

    def getObs(self):
        cloth_data = np.zeros(self.env.haptic_sensor_data["num_sensors"]*3)
        rigid_data = np.zeros(self.env.haptic_sensor_data["num_sensors"]*3)

        if self.env.haptic_sensor_data["cloth_steps"] > 0:
            #average contact forces over the cloth readings
            cloth_data = self.env.haptic_sensor_data["cloth_data"]/self.env.haptic_sensor_data["cloth_steps"]

        if self.env.haptic_sensor_data["rigid_steps"] > 0:
            #average contact forces over the rigid readings
            rigid_data = self.env.haptic_sensor_data["rigid_data"] / self.env.haptic_sensor_data["rigid_steps"]

        obs = cloth_data + rigid_data

        #normalize the readings to max 30N
        max_cloth_force = 0
        max_rigid_force = 0
        for s in range(self.env.haptic_sensor_data["num_sensors"]):
            f = obs[s*3:s*3+3]
            f /= self.mag_scale
            f_mag = np.linalg.norm(f)
            if(f_mag > 1.0):
               f /= f_mag
            obs[s*3:s*3+3] = f
            max_cloth_force = max(max_cloth_force, np.linalg.norm(cloth_data[s*3:s*3+3]))
            max_rigid_force = max(max_rigid_force, np.linalg.norm(rigid_data[s*3:s*3+3]))

        self.env.text_queue.append("max_cloth_force = " + str(max_cloth_force))
        self.env.text_queue.append("max_rigid_force = " + str(max_rigid_force))

        HSIDs = self.env.clothScene.getHapticSensorContactIDs()
        obs = np.concatenate([obs, HSIDs]).ravel()
        self.prev_obs = np.array(obs)
        return obs

    def draw(self):
        haptic_pos = self.env.clothScene.getHapticSensorLocations()
        haptic_radii = self.env.clothScene.getHapticSensorRadii()
        for h in range(self.env.clothScene.getNumHapticSensors()):
            renderUtils.setColor(color=[1, 1, 0])
            f = self.prev_obs[h * 3:h * 3 + 3]
            f_mag = np.linalg.norm(f)
            if (f_mag > 0.001):
                renderUtils.setColor(color=[0, 1, 0])
            renderUtils.drawSphere(pos=haptic_pos[h*3:h*3 + 3], rad=haptic_radii[h] * 1.1, solid=False)
            if (f_mag > 0.001):
                renderUtils.drawArrow(p0=haptic_pos[h*3:h*3 + 3]+ (f/f_mag)*haptic_radii[h]*1.1, p1=haptic_pos[h*3:h*3 + 3] + f + (f/f_mag)*haptic_radii[h]*1.1)

class JointPositionObsFeature(ObservationFeature):
    #joint positions of a particular skeleton
    def __init__(self, skel, ignored_joints=None, name="joint positions", render=False):
        self.ignored_joints = ignored_joints
        self.skel = skel
        if self.ignored_joints is None:
            self.ignored_joints = []

        numJoints = (len(skel.joints) - len(self.ignored_joints))*3

        ObservationFeature.__init__(self, name=name, dim=numJoints, render=render)

    def getObs(self):
        locs = np.zeros(0)
        for jix, j in enumerate(self.skel.joints):
            if (jix in self.ignored_joints):
                continue
            locs = np.concatenate([locs, j.position_in_world_frame()])
        return locs

class SPDTargetObsFeature (ObservationFeature):
    #observation of the interpolation pose for human SPD
    def __init__(self, env, name="Human SPD Interpolation Target", render=False):
        self.env = env
        ObservationFeature.__init__(self, name=name, dim=22, render=render)

    def getObs(self):
        obs = np.array(self.env.humanSPDIntperolationTarget)
        return obs

class RobotFramesObsFeature(ObservationFeature):
    #observation of the postion and eulers of robot interpolation frame and end effector
    def __init__(self, iiwa, name="Robot Frames Obs", render=False):
        self.iiwa = iiwa
        ObservationFeature.__init__(self, name=name, dim=12, render=render)

    def getObs(self):
        hn = self.iiwa.skel.bodynodes[8]
        roboFrame = pyutils.ShapeFrame()
        if np.isfinite(self.iiwa.skel.q).all(): #handle nan here...
            roboFrame.setTransform(hn.T)
        robotEulerStates = pyutils.getEulerAngles3(roboFrame.orientation)
        obs = np.concatenate([hn.to_world(np.zeros(3)), robotEulerStates, self.iiwa.frameInterpolator["target_pos"], self.iiwa.frameInterpolator["eulers"]]).ravel()
        return obs

class FTSensorObsFeature(ObservationFeature):
    #observation of a 6dof force/torque sensor reading
    def __init__(self, env, iiwa, name="FT Sensor Obs", render=False):
        self.env = env
        self.iiwa = iiwa
        ObservationFeature.__init__(self, name=name, dim=6, render=render)

    def getObs(self):
        obs = np.zeros(6)
        try:
            resultantFT = self.iiwa.getFTSensorReading()
            self.env.text_queue.append("FT sensor " + str(self.iiwa.index) +" reading: " + str(resultantFT))
            resultantFT[:3] *= 0.033 #force scaled ~30 -> 1
            normalizedResultantFT = np.clip(resultantFT, -1, 1) #clip anything above 30N...
            obs = normalizedResultantFT
        except:
            print("failed obs FT")
            pass
        return obs

class CapacitiveSensorObsFeature(ObservationFeature):
    def __init__(self, iiwa, name="Capacitive Sensor Obs", render=False):
        self.iiwa = iiwa
        ObservationFeature.__init__(self, name=name, dim=6, render=render)

    def getObs(self):
        return self.iiwa.capacitiveSensor.getAggregateSensorReading() / 0.15 #default 15cm range

class CollisionMPCObsFeature(ObservationFeature):
    #observation of the time since last collision warning for either robot or human
    def __init__(self, env, is_human=True, name="Collision MPC Obs", render=False):
        self.is_human = is_human
        self.env = env
        dim = 1
        if not self.is_human:
            dim = len(self.env.iiwas)
        ObservationFeature.__init__(self, name=name, dim=dim, render=render)

    def getObs(self):
        obs = np.zeros(0)
        if self.is_human:
            obs = np.array([self.env.human_collision_warning])
        else:
            readings = []
            for iiwa in self.env.iiwas:
                readings.append(iiwa.near_collision)
            obs = np.array(readings)
        return obs

class DataDrivenJointLimitsObsFeature(ObservationFeature):
    #observation of the time since last collision warning for either robot or human
    def __init__(self, env, name="Data Driven Joint Limits Obs", render=False):
        self.env = env
        ObservationFeature.__init__(self, name=name, dim=len(self.env.data_driven_constraints), render=render)

    def getObs(self):
        obs = np.zeros(0)
        # show the current constraint query
        self.env.data_driven_constraint_query = []
        for constraint in self.env.data_driven_constraints:
            self.env.data_driven_constraint_query.append(constraint.query(self.env.dart_world, False))
            obs = np.concatenate([obs, np.array([self.env.data_driven_constraint_query[-1]])])
        self.env.text_queue.append("Constraint Query: " + str(self.env.data_driven_constraint_query))
        return obs

class WeaknessScaleObsFeature(ObservationFeature):
    #observation of the weakness scale of a set of dofs
    def __init__(self, env, dofs, scale_range=(0.2,0.5), name="Weakness Scale Obs", render=True):
        '''
        :param env:
        :param dofs: a list of dof indices
        :param scale_range: a tuple (>0) defining the scaling range of initialScale (since obs should normalize to [0,1])
        :param name:
        :param render:
        '''
        self.env = env
        self.dofs = dofs
        self.scale_range = scale_range
        ObservationFeature.__init__(self, name=name, dim=1, render=render)
        self.currentScale = 1.0

    def getObs(self):
        obs = np.array([self.currentScale])
        return obs

    def reset(self):
        #draw a new scale value
        self.currentScale = np.random.uniform(0,1.0)
        print("applied action scale: " + str(self.currentScale) + " to dofs: " + str(self.dofs))

        #update the human's capability
        for d in self.dofs:
            self.env.human_action_scale[d] = LERP(self.env.initial_human_action_scale[d]*self.scale_range[0], self.env.initial_human_action_scale[d]*self.scale_range[1], self.currentScale)

    def draw(self):
        self.env.text_queue.append("Weakness Scale: %0.3f -> %0.3f" % (self.currentScale, LERP(self.scale_range[0], self.scale_range[1], self.currentScale)))

class IntentionTremorObsFeature(ObservationFeature):
    #Intention Tremor: broad, coarse, low frequency (5Hz) tremor.
    #   Amplitude increases as an extremity approaches the endpoint of deliberate and visually guided movement.
    #TODO: model this, note: end effector is most effected. Maybe dofs correspond to scale also?
    def __init__(self, env, dofs, scale_range=(0.0,0.15), frequency_range=(4,6), name="Intention Tremor Obs", render=True):
        '''
        :param env:
        :param dofs: a list of dof indices
        :param scale_range: a tuple (>0) defining the scaling range of the tremor (since obs should normalize to [0,1])
        :param frequency_range: a tuple (>0) defining the scaling range of the tremor frequency in Hz (since obs should normalize to [0,1])
        :param name:
        :param render:
        '''
        self.env = env
        self.dofs = dofs
        self.scale_range = scale_range
        ObservationFeature.__init__(self, name=name, dim=1, render=render)
        self.currentScale = 1.0

    def getObs(self):
        obs = np.array([self.currentScale])
        return obs

    def reset(self):
        #draw a new scale value
        self.currentScale = np.random.uniform(0,1.0)
        print("applied Intention Tremor scale: " + str(self.currentScale) + " to dofs: " + str(self.dofs))

        #update the human's capability
        for d in self.dofs:
            self.env.human_action_scale[d] = LERP(self.env.initial_human_action_scale[d]*self.scale_range[0], self.env.initial_human_action_scale[d]*self.scale_range[1], self.currentScale)

    def draw(self):
        self.env.text_queue.append("Intention Tremor Scale: %0.3f -> %0.3f" % (self.currentScale, LERP(self.scale_range[0], self.scale_range[1], self.currentScale)))

class OracleObsFeature(ObservationFeature):
    #observation of an oracle vector pointing from a skel sensor to a dressing target or contact geodesic gradient
    def __init__(self, env, sensor_ix, dressing_target, sep_mesh, name="oracle obs", render=True):
        self.env = env
        self.sensor_ix = sensor_ix
        self.dressing_target = dressing_target
        self.sep_mesh = sep_mesh
        ObservationFeature.__init__(self, name=name, dim=3, render=render)
        #initialize the oracle
        self.oracle_ix = len(self.env.oracles) #add a new oracle vector, that's ours
        self.env.oracles.append(np.zeros(3))

    def getObs(self):
        oracle = np.zeros(3)
        if self.dressing_target.previous_evaluation > 0:
            oracle = self.dressing_target.feature.plane.normal
        else:
            minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=self.sensor_ix,
                                                                                     clothscene=self.env.clothScene,
                                                                                     meshgraph=self.sep_mesh,
                                                                                     returnOnlyGeo=False)
            if minGeoVix is not None:
                vixSide = 0
                if _side:
                    vixSide = 1
                if minGeoVix >= 0:
                    oracle = self.sep_mesh.geoVectorAt(minGeoVix, side=vixSide)

                if minContactGeodesic == 0:
                    minGeoVix = None
                    #print("re-directing to centroid")

            if minGeoVix is None:
                #oracle points to the garment when ef not in contact
                sen_pos = self.env.clothScene.getHapticSensorLocations()[self.sensor_ix*3:self.sensor_ix*3 + 3]

                centroid = self.dressing_target.feature.plane.org

                target = np.array(centroid)
                vec = target - sen_pos
                oracle = vec/np.linalg.norm(vec)

        self.env.oracles[self.oracle_ix] = oracle
        return np.array(self.env.oracles[self.oracle_ix])

    def draw(self):
        sen_pos = self.env.clothScene.getHapticSensorLocations()[self.sensor_ix * 3:self.sensor_ix * 3 + 3]
        renderUtils.setColor(color=[0,0,0])
        renderUtils.drawArrow(p0=sen_pos, p1=sen_pos+self.env.oracles[self.oracle_ix]*0.15)

class SPDController:
    def __init__(self, env, skel, target=None, timestep=0.01, startDof=6, ckp=30000.0, ckd=100.0):

        self.target = target
        if self.target is None:
            self.target = np.array(skel.q)
        self.startDof = startDof
        self.env = env
        self.skel = skel

        self.h = timestep
        ndofs = self.skel.ndofs-startDof
        self.qhat = self.skel.q

        self.Kp = np.diagflat([ckp] * (ndofs))
        self.Kd = np.diagflat([ckd] * (ndofs))

        self.preoffset = 0.0

    def query(self, skel=None):

        #override which skeleton is controlled if necessary
        if skel is None:
            skel = self.skel

        #SPD
        self.qhat = self.target
        p = -self.Kp.dot(skel.q[self.startDof:] + skel.dq[self.startDof:] * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq[self.startDof:])
        b = -skel.c[self.startDof:] + p + d# + skel.constraint_forces()[self.startDof:]
        A = skel.M[self.startDof:, self.startDof:] + self.Kd * self.h

        #near singular matrix check ... remove for speed
        #if not np.linalg.cond(A) < 1/sys.float_info.epsilon:
        #    print("Near singular...")

        x = np.linalg.solve(A, b)

        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class ContinuousCapacitiveSensor:
    def __init__(self, env, bodynode=None, offset=np.zeros(3), R=np.identity(3)):
        self.env = env
        self.bodynode = bodynode
        self.offset = offset
        self.R = R
        self.frame = pyutils.ShapeFrame()
        self.sensor_offsets = []
        self.sensor_globals = []
        self.sensor_readings = []
        self.sensor_ranges = []
        self.numSensorRayZones = 1

    def update(self):
        try:
            self.frame.setOrg(org=self.bodynode.to_world(self.offset))
            self.frame.orientation = np.dot(self.bodynode.T[:3,:3], self.R)
            self.frame.updateQuaternion()
        except:
            print("update failed")
            pass

        # update sensor ray global info
        self.sensor_globals = []
        for off in self.sensor_offsets:
            pos = self.frame.toGlobal(p=off)
            self.sensor_globals.append(pos)

    def getReading(self):
        self.sensor_readings = []

        #first truncate all capsules to the frame half plane
        norm = self.frame.toGlobal(np.array([0,0,1])) - self.frame.toGlobal(np.zeros(3))
        norm /= np.linalg.norm(norm)
        frame_plane = Plane(org=self.frame.org, normal=norm)
        capsules = []
        #for c in self.env.skelCapsules:
        for capsule in self.env.skelCapsules:
            p0 = self.env.human_skel.bodynodes[capsule[0]].to_world(capsule[2])
            p1 = self.env.human_skel.bodynodes[capsule[3]].to_world(capsule[5])
            r0 = capsule[1]
            r1 = capsule[4]
            #print("rs: " + str(r0) + " " + str(r1))
            c_axis = p1-p0
            c_len = np.linalg.norm(c_axis)
            v0o = p0 - frame_plane.org
            v1o = p1 - frame_plane.org
            side0 = v0o.dot(norm) > 0
            side1 = v1o.dot(norm) > 0
            if side0 and side1: #capsule is entirely on the correct side of the plane
                #add it to the list unedited
                capsules.append([p0,p1,r0,r1])
            elif side0: #slice p1 down as necessary
                slice_dist = frame_plane.lineIntersect(p0, p1)[1]
                p1 = p0 + c_axis*slice_dist
                r1 = LERP(r0,r1,slice_dist/c_len)
                capsules.append([p0, p1, r0, r1])
            elif side1: #clise p0 down as necessary
                slice_dist = frame_plane.lineIntersect(p1, p0)[1]
                p0 = p1 - c_axis * slice_dist
                r0 = LERP(r1, r0, slice_dist / c_len)
                capsules.append([p0, p1, r0, r1])
            #else: don't add anything
        #print("completed phase 1 getSensorReading")

        #then for each sensor location, find the closest capsule projection
        for ix,point in enumerate(self.sensor_globals):
            dist_to_closest_point = -1
            closest_point = None
            for cap in capsules:
                #print(cap)
                closest_cap_point,dist_to_cap_point = pyutils.projectToCapsule(p=point,c0=cap[0],c1=cap[1],r0=cap[2],r1=cap[3])
                #print(dist_to_cap_point)
                #disp_to_cap_point = closest_cap_point - point
                #dist_to_cap_point = disp_to_cap_point / np.linalg.norm(disp_to_cap_point)
                if dist_to_closest_point < 0 or dist_to_closest_point > dist_to_cap_point:
                    dist_to_closest_point = dist_to_cap_point
                    closest_point = np.array(closest_cap_point)

            if dist_to_closest_point < self.sensor_ranges[ix] and dist_to_closest_point>0:
                self.sensor_readings.append([dist_to_closest_point, closest_point])
            else:
                self.sensor_readings.append([self.sensor_ranges[ix], None])
        #print("finished getSensorReading")

    def getReadingSpherical(self):
        self.sensor_readings = []

        capsules = []
        for capsule in self.env.skelCapsules:
            p0 = self.env.human_skel.bodynodes[capsule[0]].to_world(capsule[2])
            p1 = self.env.human_skel.bodynodes[capsule[3]].to_world(capsule[5])
            r0 = capsule[1]
            r1 = capsule[4]
            capsules.append([p0, p1, r0, r1])

        #then for each sensor location, find the closest capsule projection
        for ix,point in enumerate(self.sensor_globals):
            dist_to_closest_point = -1
            closest_point = None
            for cap in capsules:
                #print(cap)
                closest_cap_point,dist_to_cap_point = pyutils.projectToCapsule(p=point,c0=cap[0],c1=cap[1],r0=cap[2],r1=cap[3])
                #print(dist_to_cap_point)
                #disp_to_cap_point = closest_cap_point - point
                #dist_to_cap_point = disp_to_cap_point / np.linalg.norm(disp_to_cap_point)
                if dist_to_closest_point < 0 or dist_to_closest_point > dist_to_cap_point:
                    dist_to_closest_point = dist_to_cap_point
                    closest_point = np.array(closest_cap_point)

            if dist_to_closest_point < self.sensor_ranges[ix] and dist_to_closest_point>0:
                self.sensor_readings.append([dist_to_closest_point, closest_point])
            else:
                self.sensor_readings.append([self.sensor_ranges[ix], None])

    def draw(self, ranges=True):
        norm = self.frame.toGlobal(np.array([0, 0, 1])) - self.frame.toGlobal(np.zeros(3))
        norm /= np.linalg.norm(norm)
        lines = [[self.frame.org, self.frame.org+norm]]
        for ix,p in enumerate(self.sensor_globals):
            renderUtils.drawSphere(pos=p)
            #renderUtils.drawSphere(pos=p, rad=self.sensor_ranges[ix], solid=False)
            if self.sensor_readings[ix][0] < self.sensor_ranges[ix]:
                lines.append([p, self.sensor_readings[ix][1]])
        renderUtils.drawLines(lines)

    def default2x3setup(self, spacing=0.07):
        self.sensor_offsets = []
        self.sensor_globals = []
        self.sensor_readings = []
        self.sensor_ranges = []

        for ix in range(3):
            for iy in range(2):
                local = np.array([ix*spacing - spacing, iy*spacing - spacing/2.0, 0.065])
                self.sensor_offsets.append(local)
                self.sensor_ranges.append(0.15)
                self.sensor_readings.append([0.15, None])
        self.update()
        self.numSensorRayZones = 6

    def getAggregateSensorReading(self):
        sensorReadings = np.zeros(0)
        for reading in self.sensor_readings:
            sensorReadings = np.concatenate([sensorReadings, np.array([reading[0]])])
        return sensorReadings

class IiwaFrameController:
    def __init__(self, env):
        self.env = env

    def query(self):
        #note:  control[3] = about red (pitch)
        #       control[4] = about blue (yaw)
        #       control[5] = about green (roll)
        return np.zeros(6)

    def reset(self):
        #if any state is held, reset it
        pass

    def draw(self):
        #any necessary visualizations for the controller
        pass

class IiwaLimbTraversalController(IiwaFrameController):
    def __init__(self, env, skel, iiwa, limb, ef_offset, offset_dists, traversal_threshold=0.85):
        IiwaFrameController.__init__(self, env)
        self.skel = skel
        self.limb = limb
        self.ef_offset = ef_offset
        self.offset_dists = offset_dists
        self.iiwa = iiwa
        self.traversal_ratio = 0.0
        self.traversal_threshold = traversal_threshold
        self.p_spline = pyutils.Spline()
        self.d_spline = pyutils.Spline()

    def query(self):
        control = np.zeros(6)

        #rebuild the spline from current limb positions
        j_pos = pyutils.limbFromNodeSequence(self.skel, self.limb, self.ef_offset)
        j_pos.reverse()
        '''
        new_points = []
        limb_length = 0
        seg_lengths = [0.0]
        body_up = self.skel.bodynodes[14].to_world(np.zeros(3)) - self.skel.bodynodes[0].to_world(np.zeros(3))
        body_up /= np.linalg.norm(body_up)
        for ix in range(1,len(j_pos)-1):
            pos = j_pos[ix]
            n_pos = j_pos[ix+1]
            p_pos = j_pos[ix-1]
            from_vec = pos-p_pos
            limb_length += np.linalg.norm(from_vec)
            seg_lengths.append(limb_length)
            to_vec = n_pos-pos
            from_dir = from_vec/np.linalg.norm(from_vec)
            to_dir = to_vec/np.linalg.norm(to_vec)
            from_side = np.cross(from_dir, np.array([0,1.0,0]))
            to_side = np.cross(to_dir, np.array([0,1.0,0]))
            #from_side = np.cross(from_dir, body_up)
            #to_side = np.cross(to_dir, body_up)
            from_up = np.cross(from_side,from_dir)
            to_up = np.cross(to_side,to_dir)
            from_up /= np.linalg.norm(from_up)
            to_up /= np.linalg.norm(to_up)
            avg_up = (from_up + to_up)/2.0
            avg_up /= np.linalg.norm(avg_up)
            if ix == 1:
                new_points.append(p_pos + from_up*self.offset_dists[0])
            new_points.append(pos + avg_up*self.offset_dists[ix])
            if ix == len(j_pos)-2:
                new_points.append(n_pos + to_up * self.offset_dists[-1])
                limb_length += np.linalg.norm(to_vec)
                seg_lengths.append(limb_length)
        #self.p_spline = pyutils.Spline()
        #for ix, p in enumerate(new_points):
        #    self.p_spline.insert(t=seg_lengths[ix] / limb_length, p=p)
        '''

        spline_point = self.p_spline.pos(self.traversal_ratio)
        while np.linalg.norm(spline_point - self.iiwa.skel.bodynodes[9].to_world(np.zeros(3))) < self.d_spline.pos(self.traversal_ratio)+0.05 and self.traversal_ratio < self.traversal_threshold:
            self.traversal_ratio += 0.01
            spline_point = self.p_spline.pos(self.traversal_ratio)
        pos_error = spline_point - self.iiwa.frameInterpolator["target_pos"]
        tar_dist = self.d_spline.pos(self.traversal_ratio)
        #print("tar_dist: " + str(tar_dist))
        if np.linalg.norm(pos_error) < tar_dist:
            pos_error = (self.iiwa.skel.bodynodes[9].to_world(np.zeros(3)) - spline_point)
            pos_error /= np.linalg.norm(pos_error)
            pos_error *= tar_dist
        control[:3] = pos_error

        # angular control: orient toward the shoulder to within some error
        frame = pyutils.ShapeFrame()
        frame.setOrg(org=self.iiwa.frameInterpolator["target_pos"])
        frame.orientation = np.array(self.iiwa.frameInterpolator["target_frame"])
        frame.updateQuaternion()
        v0 = frame.toGlobal(p=[1, 0, 0]) - frame.org
        v1 = frame.toGlobal(p=[0, 1, 0]) - frame.org
        v2 = frame.toGlobal(p=[0, 0, 1]) - frame.org
        eulers_current = pyutils.getEulerAngles3(frame.orientation)

        # toShoulder = self.env.robot_skeleton.bodynodes[9].to_world(np.zeros(3)) - renderFrame.org
        t_dir = self.p_spline.vel(self.traversal_ratio)
        t_dir /= np.linalg.norm(t_dir)
        R = pyutils.rotateTo(v1 / np.linalg.norm(v1), t_dir)
        frame.applyRotationMatrix(R)
        eulers_target = pyutils.getEulerAngles3(frame.orientation)
        eulers_diff = eulers_target - eulers_current
        control[3:] = eulers_diff
        # print(eulers_diff)

        #rotate sensors parallel to -y
        Rdown = pyutils.rotateTo(v2 / np.linalg.norm(v2), np.array([0, -1, 0]))
        frame.applyRotationMatrix(Rdown)
        eulers_target = pyutils.getEulerAngles3(frame.orientation)
        eulers_diff_down = eulers_target - eulers_current
        control[3] += eulers_diff_down[0]
        #control[3:] += eulers_diff_down*0.5

        to_point = spline_point - self.iiwa.skel.bodynodes[9].to_world(np.zeros(3))
        to_point /= np.linalg.norm(to_point)
        Rdown = pyutils.rotateTo(v2 / np.linalg.norm(v2), to_point)
        frame.applyRotationMatrix(Rdown)
        eulers_target = pyutils.getEulerAngles3(frame.orientation)
        eulers_diff_down = eulers_target - eulers_current
        #control[3] += eulers_diff_down[0]
        control[3:] += eulers_diff_down

        control = np.clip(control, -self.env.robot_action_scale*0.5, self.env.robot_action_scale*0.5)

        return control

    def reset(self):
        self.traversal_ratio = 0.0

    def draw(self):

        sides = []
        ups = []
        avg_ups = []

        # rebuild the spline from current limb positions
        j_pos = pyutils.limbFromNodeSequence(self.skel, self.limb, self.ef_offset)
        j_pos.reverse()

        '''
        new_points = []
        limb_length = 0
        seg_lengths = [0.0]
        body_up = self.skel.bodynodes[14].to_world(np.zeros(3)) - self.skel.bodynodes[0].to_world(np.zeros(3))
        body_up /= np.linalg.norm(body_up)
        renderUtils.drawArrow(p0=self.skel.bodynodes[0].to_world(np.zeros(3)), p1=self.skel.bodynodes[14].to_world(np.zeros(3)))
        for ix in range(1, len(j_pos) - 1):
            pos = j_pos[ix]
            n_pos = j_pos[ix + 1]
            p_pos = j_pos[ix - 1]
            from_vec = pos - p_pos
            limb_length += np.linalg.norm(from_vec)
            seg_lengths.append(limb_length)
            to_vec = n_pos - pos
            from_dir = from_vec / np.linalg.norm(from_vec)
            to_dir = to_vec / np.linalg.norm(to_vec)
            from_side = np.cross(from_dir, np.array([0, 1.0, 0]))
            to_side = np.cross(to_dir, np.array([0, 1.0, 0]))
            #from_side = np.cross(from_dir, body_up)
            #to_side = np.cross(to_dir, body_up)
            from_side /= np.linalg.norm(from_side)
            to_side /= np.linalg.norm(to_side)
            sides.append([pos,pos+to_side])
            sides.append([pos,pos+from_side])
            from_up = np.cross(from_side, from_dir)
            to_up = np.cross(to_side, to_dir)
            from_up /= np.linalg.norm(from_up)
            to_up /= np.linalg.norm(to_up)
            if np.dot(from_up, self.iiwa.root_dofs[3:]-pos) < 0:
                from_up *= -1.0
            if np.dot(to_up, self.iiwa.root_dofs[3:]-pos) < 0:
                to_up *= -1.0
            ups.append([pos, pos + to_up])
            ups.append([pos, pos + from_up])
            avg_up = (from_up + to_up) / 2.0
            avg_up /= np.linalg.norm(avg_up)
            avg_ups.append([pos, pos + avg_up])
            if ix == 1:
                new_points.append(p_pos + from_up * self.offset_dists[0])
                sides.append([p_pos, p_pos + from_side])
                ups.append([p_pos, p_pos + from_up])
            new_points.append(pos + avg_up * self.offset_dists[ix])
            if ix == len(j_pos) - 2:
                new_points.append(n_pos + to_up * self.offset_dists[-1])
                limb_length += np.linalg.norm(to_vec)
                seg_lengths.append(limb_length)
                sides.append([n_pos, n_pos + to_side])
                ups.append([n_pos, n_pos + to_up])

        self.p_spline = pyutils.Spline()
        for ix, p in enumerate(new_points):
            self.p_spline.insert(t=seg_lengths[ix] / limb_length, p=p)
        '''
        self.d_spline = pyutils.Spline()
        self.p_spline = pyutils.Spline()
        seg_lengths = [0.0]
        limb_length = 0
        for ix, p in enumerate(j_pos):
            if ix==0:
                continue
            limb_length += np.linalg.norm(p-j_pos[ix-1])
            seg_lengths.append(limb_length)

        for ix, p in enumerate(j_pos):
            self.p_spline.insert(t=seg_lengths[ix] / limb_length, p=p)
            self.d_spline.insert(t=seg_lengths[ix] / limb_length, p=self.offset_dists[ix])

        self.p_spline.draw()
        spline_point = self.p_spline.pos(self.traversal_ratio)
        renderUtils.drawArrow(p0=spline_point, p1=spline_point + self.p_spline.vel(self.traversal_ratio)/np.linalg.norm(self.p_spline.vel(self.traversal_ratio))*0.15)
        renderUtils.drawSphere(pos=spline_point)
        end_plane = Plane(org=self.p_spline.pos(self.traversal_threshold), normal=self.p_spline.vel(self.traversal_threshold))
        end_plane.draw(size=0.05)

        renderUtils.setColor(color=[0, 1, 1])
        renderUtils.drawLines(lines=sides)
        renderUtils.setColor(color=[1, 0, 1])
        renderUtils.drawLines(lines=ups)
        renderUtils.setColor(color=[1,0,0])
        renderUtils.drawLines(lines=avg_ups)

class IiwaApproachHoverProceedAvoidController(IiwaFrameController):
    # This controller approaches the character and hovers at a reasonable range to allow the human to dress the sleeve
    def __init__(self, env, iiwa, dressingTargets, target_node, node_offset, distance, noise=0.0, control_fraction=0.3, slack=(0.1, 0.075), hold_time=1.0, avoidDist=0.1, other_iiwas=None):
        IiwaFrameController.__init__(self, env)
        self.iiwa = iiwa
        self.other_iiwas = []
        if other_iiwas is not None: #list of other robots this one should regulate distance with
            self.other_iiwas = other_iiwas
        self.dressingTargets = dressingTargets
        self.target_node = target_node
        self.target_position = np.zeros(3)
        self.distance = distance
        self.noise = noise
        self.control_fraction = control_fraction
        self.slack = slack
        self.hold_time = hold_time  # time to wait after limb progress > 0 before continuing
        self.time_held = 0.0
        self.proceeding = False
        self.nodeOffset = node_offset
        self.avoid_dist = avoidDist
        self.closest_point = None
        self.dist_to_closest_point = -1

    def query(self):
        control = np.zeros(6)
        #noiseAddition = np.random.uniform(-self.env.robot_action_scale, self.env.robot_action_scale) * self.noise
        worldTarget = self.env.human_skel.bodynodes[self.target_node].to_world(self.nodeOffset)
        if self.proceeding:
            worldTarget = np.array(self.target_position)
        iiwaEf = self.iiwa.skel.bodynodes[8].to_world(np.array([0, 0, 0.05]))
        iiwa_frame_org = self.iiwa.frameInterpolator["target_pos"]
        t_disp = worldTarget - iiwa_frame_org
        t_dist = np.linalg.norm(t_disp)
        t_dir = t_disp / t_dist

        # move down to level with target by default (should counter the avoidance)
        y_disp = worldTarget[1] - iiwa_frame_org[1]
        if y_disp < 0:
            control[1] = y_disp

        relevant_limb_progress = 1.0
        for tar in self.dressingTargets:
            relevant_limb_progress = min(relevant_limb_progress, tar.previous_evaluation)

        #print(relevant_limb_progress)

        if relevant_limb_progress < 0 and not self.proceeding:
            #move toward the target distance
            if t_dist > (self.slack[0] + self.distance):
                control[:3] = t_dir * (t_dist - self.distance + self.slack[0])
            elif t_dist < (self.distance - self.slack[0]):
                control[:3] = t_dir * (t_dist - self.distance - self.slack[0])
        else:
            self.time_held += self.env.dt*self.env.frame_skip
            #print("self.time_held: " + str(self.time_held))
            if self.time_held > self.hold_time:
                if not self.proceeding:
                    self.proceeding = True
                    self.target_position = np.array(worldTarget)
                # now begin moving toward the arm again (still using the slack distance)
                # TODO: better way to measure distance at the end?
                control[:3] = t_dir * (t_dist - self.slack[0])
                #print("moving toward shoulder")

        # angular control: orient toward the shoulder to within some error
        frame = pyutils.ShapeFrame()
        frame.setOrg(org=self.iiwa.frameInterpolator["target_pos"])
        frame.orientation = np.array(self.iiwa.frameInterpolator["target_frame"])
        frame.updateQuaternion()
        v0 = frame.toGlobal(p=[1, 0, 0]) - frame.org
        v1 = frame.toGlobal(p=[0, 1, 0]) - frame.org
        v2 = frame.toGlobal(p=[0, 0, 1]) - frame.org
        eulers_current = pyutils.getEulerAngles3(frame.orientation)

        # toShoulder = self.env.robot_skeleton.bodynodes[9].to_world(np.zeros(3)) - renderFrame.org
        R = pyutils.rotateTo(v1 / np.linalg.norm(v1), t_dir)
        frame.applyRotationMatrix(R)
        eulers_target = pyutils.getEulerAngles3(frame.orientation)
        eulers_diff = eulers_target - eulers_current
        control[3:] = eulers_diff
        # print(eulers_diff)

        Rdown = pyutils.rotateTo(v2 / np.linalg.norm(v2), np.array([0, -1, 0]))
        frame.applyRotationMatrix(Rdown)
        eulers_target = pyutils.getEulerAngles3(frame.orientation)
        eulers_diff_down = eulers_target - eulers_current
        control[3] += eulers_diff_down[0]
        # control[5] += 1

        for i in range(3):
            if abs(control[3 + i]) < self.slack[1]:
                control[3 + i] = 0

        control = np.clip(control, -self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+6]* self.control_fraction, self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+6]*self.control_fraction)

        #check control frame position against body distance and deflect if necessary

        #1: find the closest point on the body
        projected_frame_center = control[:3] + iiwa_frame_org
        self.dist_to_closest_point = -1
        for cap in self.env.skelCapsules:
            p0 = self.env.human_skel.bodynodes[cap[0]].to_world(cap[2])
            p1 = self.env.human_skel.bodynodes[cap[3]].to_world(cap[5])
            #renderUtils.drawCapsule(p0=p0, p1=p1, r0=cap[1], r1=cap[4])
            closestPoint = pyutils.projectToCapsule(p=projected_frame_center,c0=p0,c1=p1,r0=cap[1],r1=cap[4])[0]
            #print(closestPoint)
            disp_to_point = (projected_frame_center-closestPoint)
            dist_to_point = np.linalg.norm(disp_to_point)
            if(self.dist_to_closest_point > dist_to_point or self.dist_to_closest_point<0):
                self.dist_to_closest_point = dist_to_point
                self.closest_point = np.array(closestPoint)

        #if that point is too close, push back against it as necessary
        if self.dist_to_closest_point < self.avoid_dist:
            disp_to_point = self.closest_point - projected_frame_center
            dir_to_point = disp_to_point/self.dist_to_closest_point
            control[:3] += -dir_to_point*(self.avoid_dist-self.dist_to_closest_point)
            #control[:3] = np.clip(control[:3], -self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction,self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction)

        for other_iiwa in self.other_iiwas:
            #if the two robots are too far from one another, move toward the other
            v_bt_frames = other_iiwa.frameInterpolator["target_pos"] - self.iiwa.frameInterpolator["target_pos"]
            dist_bt_frames = abs(np.linalg.norm(v_bt_frames))
            if dist_bt_frames > 0.48:
                dir_to_other = v_bt_frames/dist_bt_frames
                control[:3] += dir_to_other
                #control[:3] = np.clip(control[:3], -self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction,self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction)
            # if the two robots are too close, move away
            elif dist_bt_frames < 0.2:
                v_bt_frames *= -1.0
                dir_from_other = v_bt_frames / dist_bt_frames
                control[:3] += dir_from_other
        control[:3] = np.clip(control[:3], -self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction,self.env.robot_action_scale[self.iiwa.index*6:self.iiwa.index*6+3] * self.control_fraction)


        return control #+ noiseAddition

    def reset(self):
        self.time_held = 0
        self.proceeding = False

    def draw(self):
        worldTarget = self.env.human_skel.bodynodes[self.target_node].to_world(self.nodeOffset)
        if self.proceeding:
            worldTarget = np.array(self.target_position)
        iiwa_frame_org = self.iiwa.frameInterpolator["target_pos"]
        renderUtils.drawLines(lines=[[worldTarget, iiwa_frame_org]])
        renderUtils.drawSphere(pos=worldTarget)
        if self.dist_to_closest_point > 0:
            renderUtils.drawLines(lines=[[self.closest_point, iiwa_frame_org]])
            renderUtils.drawSphere(pos=self.closest_point)

class IiwaTrackController(IiwaFrameController):
    def __init__(self, env, track_spline, iiwa):
        IiwaFrameController.__init__(self,env)
        self.track_spline = track_spline
        self.iiwa = iiwa
        self.track_time = 0

    def query(self):
        #note:  control[3] = about red (pitch)
        #       control[4] = about blue (yaw)
        #       control[5] = about green (roll)
        return np.zeros(6)

    def reset(self):
        self.track_time = 0

    def draw(self):
        self.track_spline.draw()


class Iiwa:
    #contains all necessary structures to define, control, simulate and reset a single Iiwa robot instance

    def __init__(self, skel, env, index, root_dofs=np.zeros(6), active_compliance=True):
        self.env = env
        self.skel = skel
        self.index = index
        self.active_compliance = active_compliance

        self.frameInterpolator = {"active": True, "target_pos": np.zeros(3), "target_frame": np.identity(3), "speed": 0.5, "aSpeed": 4, "localOffset": np.array([0, 0, 0]), "eulers": np.zeros(3), "distanceLimit": 0.15}
        self.manualFrameTarget = pyutils.ShapeFrame()
        self.manualFrameTarget.setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=np.zeros(3))
        self.frameEulerState = np.array([math.pi/2.0, -0.2, -0.15]) #XYZ used as euler angles to modify orientation base
        self.ik_target = pyutils.ShapeFrame()
        self.previousIKResult = np.zeros(7)
        self.near_collision = 1.0

        self.handle_node = None
        self.handle_node_offset = np.zeros(3)
        self.handle_bodynode = 8
        self.FT_sensor = {"cloth_steps":0, "cloth_F":np.zeros(3), "cloth_T":np.zeros(3), "rigid_steps":0, "rigid_F":np.zeros(3), "rigid_T":np.zeros(3)}
        self.capacitiveSensor = ContinuousCapacitiveSensor(env=self.env, bodynode=self.skel.bodynodes[8])
        self.capacitiveSensor.default2x3setup()

        self.iiwa_frame_controller = None
        #TODO: adjust SPD gains
        self.SPDController = SPDController(self, self.skel, timestep=self.env.dt, ckp=30000.0, ckd=300.0) #computed every simulation timestep (dt)
        #TODO: edit torque limits
        self.torque_limits = np.array([176, 176, 110, 110, 110, 40, 40])
        #self.torque_limits = np.array([176, 176, 110, 110, 110, 40, 40])*0.2

        self.robotEulerStates = []
        self.orientationEulerStateLimits = [np.array([math.pi/2.0-1.5, -1.5, -1.5]), np.array([math.pi/2.0+1.5, 1.5, 1.5])]
        self.root_dofs = np.array(root_dofs)
        self.rest_dofs = np.zeros(7)
        self.skel.set_positions(np.concatenate([np.array(self.root_dofs), self.rest_dofs]))

        #setup the skel
        for ix, bodynode in enumerate(self.skel.bodynodes):
            bodynode.set_gravity_mode(False)
        for ix, joint in enumerate(self.skel.joints):
            joint.set_position_limit_enforced()
        for ix, dof in enumerate(self.skel.dofs):
            dof.set_damping_coefficient(1.0) #TODO: change this?
        self.skel.joints[0].set_actuator_type(Joint.Joint.LOCKED)
        self.skel.set_self_collision_check(True)
        self.skel.set_adjacent_body_check(False)

    def printSkelDetails(self):
        print("Iiwa " + str(self.index) + " details:")
        print(" Bodynodes:")
        for ix, bodynode in enumerate(self.skel.bodynodes):
            print("      " + str(ix) + " : " + bodynode.name)
            print("         mass: " + str(bodynode.mass()))
        print(" Joints: ")
        for ix, joint in enumerate(self.skel.joints):
            print("     " + str(ix) + " : " + joint.name)
        print(" Dofs: ")
        for ix, dof in enumerate(self.skel.dofs):
            print("     " + str(ix) + " : " + dof.name)
            print("         llim: " + str(dof.position_lower_limit()) + ", ulim: " + str(dof.position_upper_limit()))

    def computeIK(self, maxIter=10):
        #compute IK from current ikTarget frame
        #print(self.ik_target.org)
        #print(self.skel.bodynodes[8].to_world(np.zeros(3)))
        tar_pos = np.array(self.ik_target.org)
        tar_quat = self.ik_target.quat
        tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
        self.env.setPosePyBullet(pose=self.skel.q[6:])
        result = p.calculateInverseKinematics(bodyUniqueId=self.env.pyBulletIiwa,
                                              endEffectorLinkIndex=8,
                                              targetPosition=tar_pos - self.root_dofs[3:],
                                              targetOrientation=tar_quat,
                                              # targetOrientation=tar_dir,
                                              lowerLimits=self.env.iiwa_dof_llim.tolist(),
                                              upperLimits=self.env.iiwa_dof_ulim.tolist(),
                                              jointRanges=self.env.iiwa_dof_jr.tolist(),
                                              restPoses=self.skel.q[6:].tolist(),
                                              maxNumIterations=maxIter
                                              )

        #predicted_pose = self.skel.q + (np.concatenate([self.root_dofs, result]) - self.skel.q)*5.0
        predicted_pose = np.concatenate([self.root_dofs, result])
        #if self.env.checkIiwaPose(self.skel, predicted_pose):



        if self.env.checkProxyPose(self.index, predicted_pose):
            self.previousIKResult = np.array(result)
        else:
            #print("collision pose")
            self.env.text_queue.append("MPC COLLISION_ROBOT_WARNING")
            self.near_collision = 0

        if not self.active_compliance:
            self.previousIKResult = np.array(result)
        elif self.near_collision == 0:
            self.skel.dq *= 0.5

    def setIKPose(self, setFrame=True):
        #set the most recent IK solved pose
        self.skel.set_positions(np.concatenate([np.array(self.root_dofs), self.previousIKResult]))
        hn = self.skel.bodynodes[9]  # hand node
        efFrame = pyutils.ShapeFrame()
        efFrame.setTransform(T=hn.world_transform())
        self.robotEulerStates = pyutils.getEulerAngles3(efFrame.orientation)
        self.capacitiveSensor.update()
        if setFrame:
            #also update the frame to end effector location
            self.frameInterpolator["target_pos"] = np.array(efFrame.org)
            self.frameInterpolator["target_frame"] = np.array(efFrame.orientation)
            self.frameInterpolator["eulers"] = np.array(self.robotEulerStates)
            self.frameEulerState = np.array(self.robotEulerStates)

    def setRestPose(self):
        self.skel.set_positions(np.concatenate([np.array(self.root_dofs), np.array(self.rest_dofs)]))

    def computeTorque(self):
        tau_mod = 1.0
        if self.active_compliance:
            tau_mod = LERP(0.001,1.0,min(self.near_collision,0.5)/0.5)
        #print(tau_mod)
        self.env.text_queue.append("tau_mod: " + str(tau_mod))

        #compute motor torque for previousIKResult
        self.SPDController.target = np.array(self.previousIKResult)

        if self.near_collision < 1.0 and self.active_compliance:
            self.SPDController.target = LERP(np.array(self.skel.q[6:]), np.array(self.previousIKResult), min(self.near_collision,0.5)/0.5)

        tau = np.clip(self.SPDController.query(skel=self.skel), -self.torque_limits, self.torque_limits)
        #if self.nearCollision or len(self.env.humanRobotCollisions) > 0:
        tau = np.clip(tau, -self.torque_limits*tau_mod, self.torque_limits*tau_mod)
        tau = np.concatenate([np.zeros(6), tau])

        if self.near_collision < 0.25 and self.active_compliance:
            for t in range(6,len(tau)):
                if abs(self.skel.dq[t]) > 0.1:
                    if tau[t] > 0 == self.skel.dq[t] > 0:
                        tau[t] = 0

        return tau

    def controlFrame(self, control):
        #update the interpolation frame from a control signal (6 dof position and angle update)
        if self.env.manual_robot_control:
            pass
            #self.manualFrameTarget.org += pyutils.sampleDirections(num=1)[0]*0.001
            #self.frameInterpolator["target_pos"] = np.array(self.manualFrameTarget.org)
            #self.frameInterpolator["eulers"] = pyutils.getEulerAngles3(self.manualFrameTarget.orientation)
        else:
            #First excecute scripted changes, then deviations
            if self.iiwa_frame_controller is not None:
                scripted_control = self.iiwa_frame_controller.query()
                self.frameInterpolator["target_pos"] += scripted_control[:3]
                self.frameInterpolator["eulers"] += scripted_control[3:]
            else:
                #neural net policy control
                self.frameInterpolator["target_pos"] += control[:3]
                self.frameInterpolator["eulers"] += control[3:6]

        toRoboEF = self.skel.bodynodes[9].to_world(np.zeros(3)) - self.frameInterpolator["target_pos"]
        distToRoboEF = np.linalg.norm(toRoboEF)
        # clamp interpolation target frame to distance from EF to prevent over commit
        if (distToRoboEF > (self.frameInterpolator["distanceLimit"])):
            # print("clamping frame")
            self.frameInterpolator["target_pos"] = self.skel.bodynodes[9].to_world(np.zeros(3)) + -(toRoboEF / distToRoboEF) * self.frameInterpolator["distanceLimit"]

    def interpolateIKTarget(self, dt=None):
        #if not dt is given, use the env default timestep
        if dt is None:
            dt = self.env.dt*self.env.frame_skip

        self.near_collision = min(self.near_collision+dt, 1.0)

        if self.near_collision < 0.1 and self.active_compliance:
            hn = self.skel.bodynodes[9]
            self.ik_target.setTransform(hn.T)

        # interpolate the target frame at constant speed toward a goal location
        if self.frameInterpolator["active"]:
            targetFrame = pyutils.ShapeFrame()
            targetFrame.orientation = np.array(self.frameInterpolator["target_frame"])
            self.frameInterpolator["eulers"] = np.clip(self.frameInterpolator["eulers"],
                                                                self.orientationEulerStateLimits[0],
                                                                self.orientationEulerStateLimits[1])

            interpFrameEulerState = np.array(self.frameInterpolator["eulers"])

            # interpolate orientation:
            if True:
                self.frameInterpolator["target_frame"] = pyutils.euler_to_matrix(interpFrameEulerState)
                targetFrame.orientation = np.array(self.frameInterpolator["target_frame"])

            targetFrame.org = np.array(self.frameInterpolator["target_pos"])
            targetFrame.updateQuaternion()
            globalOffset = targetFrame.toGlobal(p=self.frameInterpolator["localOffset"])
            targetFrame.org += targetFrame.org - globalOffset
            targetDisp = targetFrame.org - self.ik_target.org
            dispMag = np.linalg.norm(targetDisp)
            travelMag = min(dt * self.frameInterpolator["speed"], dispMag)
            if (travelMag > 0.001):
                self.ik_target.org += (targetDisp / dispMag) * travelMag

            # interpolate the target eulers:
            hn = self.skel.bodynodes[9]
            roboFrame = pyutils.ShapeFrame()
            roboFrame.setTransform(hn.T)

            robo_eulers = pyutils.getEulerAngles3(roboFrame.orientation)
            for i in range(3):
                eDist = interpFrameEulerState[i] - robo_eulers[i]
                eTraverse = min(dt * self.frameInterpolator["aSpeed"], abs(eDist))
                if (eDist != 0):
                    self.frameEulerState[i] = robo_eulers[i] + (eDist / abs(eDist)) * eTraverse

        self.frameEulerState = np.clip(self.frameEulerState, self.orientationEulerStateLimits[0], self.orientationEulerStateLimits[1])

        try:
            self.ik_target.orientation = pyutils.euler_to_matrix(self.frameEulerState)
            self.ik_target.updateQuaternion()
        except:
            print("INVALID TARGET SETTING STATE for IIWA " + str(self.index))
            # print(" targetDir: " + str(targetDir))
            print(" self.ikTarget: " + str(self.ik_target.org))

    def step(self, control):
        #update frame, interpolate ikTarget, compute IK, update internal structures
        self.controlFrame(control)
        self.interpolateIKTarget()
        self.computeIK()

    def updateCapacitiveSensor(self):
        #update capacitive sensor readings
        self.capacitiveSensor.update()
        self.capacitiveSensor.getReadingSpherical()

    def addClothHandle(self, verts, offset, bodynode=8):
        self.handle_bodynode = bodynode
        self.handle_node_offset = np.array(offset)
        hn = self.skel.bodynodes[bodynode]
        self.handle_node = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.handle_node.addVertices(verts=verts)
        self.handle_node.setOrgToCentroid()
        self.handle_node.org = hn.to_world(self.handle_node_offset)
        self.handle_node.setOrientation(R=hn.T[:3, :3])
        self.handle_node.recomputeOffsets()
        self.handle_node.updatePrevConstraintPositions()

    def updateClothHandle(self):
        if self.handle_node is None:
            return
        #update cloth handle and FT sensor reading
        hn = self.skel.bodynodes[self.handle_bodynode]  # end effector (hand) node
        self.handle_node.updatePrevConstraintPositions()
        newOrg = hn.to_world(self.handle_node_offset)
        if not np.isfinite(newOrg).all():
            print("Invalid robot " + str(self.index) + " pose in handle update...")
        else:
            self.handle_node.org = np.array(newOrg)
            self.handle_node.setOrientation(R=hn.T[:3, :3])

    def getFTSensorReading(self):
        #return average FT over previously logged steps
        avg_cloth_FT = np.zeros(6)
        avg_rigid_FT = np.zeros(6)
        if self.FT_sensor["cloth_steps"] > 0:
            avg_cloth_FT = np.concatenate([self.FT_sensor["cloth_F"], self.FT_sensor["cloth_T"]])/self.FT_sensor["cloth_steps"]
        if self.FT_sensor["rigid_steps"] > 0:
            avg_rigid_FT = np.concatenate([self.FT_sensor["rigid_F"], self.FT_sensor["rigid_T"]])/self.FT_sensor["rigid_steps"]

        return avg_cloth_FT + avg_rigid_FT

    def updateClothFTSensorReading(self, clear=False):
        if clear: #clear the previous set of readings
            self.FT_sensor["cloth_steps"] = 0
            self.FT_sensor["cloth_F"] = np.zeros(3)
            self.FT_sensor["cloth_T"] = np.zeros(3)
        if self.handle_node is not None:
            self.FT_sensor["cloth_steps"] += 1
            self.FT_sensor["cloth_F"] += self.handle_node.prev_force
            self.FT_sensor["cloth_T"] += self.handle_node.prev_torque

    def updateRigidFTSensorReading(self, F, T, clear=False):
        if clear: #clear the previous set of readings
            self.FT_sensor["rigid_steps"] = 0
            self.FT_sensor["rigid_F"] = np.zeros(3)
            self.FT_sensor["rigid_T"] = np.zeros(3)
        if self.handle_node is not None:
            self.FT_sensor["rigid_steps"] += 1
            self.FT_sensor["rigid_F"] += F
            self.FT_sensor["rigid_T"] += T


class DressingTarget:
    #defines a dressing target task (e.g. "put limb L into feature F")
    def __init__(self, env, skel, feature, limb_sequence, distal_offset=None):
        self.env = env
        self.skel = skel
        self.feature = feature
        self.limb_sequence = limb_sequence
        self.distal_offset = distal_offset
        self.previous_evaluation = 0

    def getLimbProgress(self):
        self.previous_evaluation = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.skel, nodes=self.limb_sequence, offset=self.distal_offset), feature=self.feature)
        return self.previous_evaluation

class DartClothIiwaEnv(gym.Env):
    """Superclass for all Dart, PhysX Cloth, Human/Iiwa interaction environments.
        """

    def __init__(self, human_world_file=None, proxy_human_world_file=None, robot_file=None, pybullet_robot_file=None, experiment_directory=None, dt=0.0025, frame_skip=4, task_horizon=600, simulate_cloth=True, cloth_mesh_file=None, cloth_mesh_state_file=None, cloth_scale= 1.0, cloth_friction=0.25, robot_root_dofs=[], active_compliance=True, dual_policy=True, is_human=True):
        '''

        :param human_world_file: filename of human skel and world file for dart in assets folder, default if None
        :param proxy_human_world_file: filename of human skel and world file for dart in assets folder, default if None (proxy world is used for collision MPC)
        :param robot_file: filename of (or path to) robot urdf or skel file (no world) from assets folder, default if None
        :param pybullet_robot_file: filename of (or path to) robot urdf from assets folder, default if None
        :param experiment_directory: name of folder in rllab/data/local/experiment/ containing policy file for "other" policy
        :param dt: DART simulation timestep
        :param frame_skip: number of DART simulation steps between control steps (frames)
        :param task_horizon: number of control frames before automatic termination (used to determine minimum reward)
        :param simulate_cloth: boolean, if False, don't run cloth simulation (other cloth setup continues as normal)
        :param cloth_mesh_file: if not None, load a cloth mesh file to simulate
        :param cloth_mesh_state_file: if not None, use the designated file as a mesh state for reset
        :param cloth_scale: (default 1.0): uniform scale of the .obj mesh file
        :param cloth_friction: (default 0.5): cloth/cloth and cloth/rigid friction level [0,1]
        :param robot_root_dofs: list of 6 root dof value lists, one entry per robot to be initialized
        '''

        #setup some flags
        self.dual_policy = dual_policy #if true, expect an action space concatenation of human/robot(s)
        self.dualPolicy = dual_policy
        self.is_human = is_human #(ignore if dualPolicy is True) if true, human action space is active, otherwise robot action space is active.
        self.rendering = False
        self.dart_render = True
        self.proxy_render = False
        self.cloth_render = True
        self.detail_render = False
        self.simulating = False #used to allow simulation freezing while rendering continues
        self.passive_robots = False #if true, no motor torques from the robot
        self.active_compliance = active_compliance
        self.manual_robot_control = False
        self.manual_human_control = False
        self.print_skel_details = True
        self.data_driven_joint_limits = True
        self.screen_size = (720, 720)
        if self.detail_render:
            self.screen_size = (1080,720)
        self.obs_dim = 0 #set later (in subclasses)
        self.act_dim = 0 #set later (if default in this method)
        self.cloth_force_scale = -15.0 #these forces are in the wrong direction

        self.dt = dt
        self.frame_skip = frame_skip
        self.task_horizon = task_horizon

        self.viewer = None
        self.setSeed = None

        self.reset_number = 0
        self.numSteps = 0

        #load other policy file if necessary
        self.prefix = os.path.dirname(__file__)
        self.experiment_prefix = self.prefix + "/../../../../rllab/data/local/experiment/"
        self.experiment_directory = experiment_directory
        if self.experiment_directory is None:
            #default to this experiment
            self.experiment_directory = "experiment_2019_02_13_human_multibot"
        self.otherPolicyFile = self.experiment_prefix + self.experiment_directory + "/policy.pkl"
        self.otherPolicy = None
        if not self.dual_policy:
            try:
                self.otherPolicy = joblib.load(self.otherPolicyFile)
            except:
                print("cannot load the other Policy...")

        #NOTE: should be overridden in subclasses unless default human and robot are expected
        self.action_space = spaces.Box(np.ones(1), np.ones(1)*-1.0)

        # override these for different human skel
        self.human_action_scale = np.ones(1)
        self.initial_human_action_scale = np.array(self.human_action_scale)
        self.human_control_bounds = np.array([np.ones(1), np.ones(1) * -1])

        #6 dof action space for each robot
        self.robot_action_scale = np.ones(6*len(robot_root_dofs))
        self.robot_control_bounds = np.array([np.ones(6*len(robot_root_dofs)), np.ones(6*len(robot_root_dofs)) * -1])

        #NOTE: must override this in subclasses once obs space is defined...
        self.observation_space = spaces.Box(np.inf*np.ones(1)*-1.0, np.inf*np.ones(1))
        self.humanRobotCollisions = [] #list of collisions between human and robots this frame used to inform some observation terms...

        #if no alternative given, load default world
        defaultHuman = False
        if human_world_file is None:
            human_world_file = 'UpperBodyCapsules_datadriven.skel'
            defaultHuman = True

        #convert to full path
        human_world_file = os.path.join(os.path.dirname(__file__), "assets", human_world_file)
        if not path.exists(human_world_file):
            raise IOError("File %s does not exist" % human_world_file)

        # load the DART world
        self.dart_world = pydart.World(dt, human_world_file)
        self.dart_world.set_gravity(np.array([0., -9.8, 0]))
        self.collisionResult = CollisionResult.CollisionResult(self.dart_world)

        #proxy world setup for collision MPC (some more in iiwa initialization)
        self.proxy_dart_world = None
        self.proxy_collisionResult = None
        self.proxy_human_skel = None
        self.initializeProxyWorld(filename=proxy_human_world_file)

        # ----------------------------
        # setup the human
        # ----------------------------
        self.human_skel = self.dart_world.skeletons[-1] #assume the world file last skeleton is the human model
        self.humanSPDController = SPDController(self, self.human_skel, timestep=self.frame_skip * self.dt, startDof=0, ckp=30.0, ckd=0.1)
        self.humanSPDIntperolationTarget = np.zeros(self.human_skel.ndofs)
        self.humanSPDInterpolationRate = 1.0 #defines the maximum joint range percentage change per second of simulation
        self.human_manual_target = np.zeros(self.human_skel.ndofs) #if manual control, set interpolation target here every frame
        self.human_collision_warning = 1.0 #[0,1] seconds since last collision warning

        for jt in range(0, len(self.human_skel.joints)):
            if self.human_skel.joints[jt].has_position_limit(0):
                self.human_skel.joints[jt].set_position_limit_enforced(True)

        # enable DART collision testing
        self.human_skel.set_self_collision_check(True)
        self.human_skel.set_adjacent_body_check(False)

        self.fingertip = np.array([0, -0.085, 0])
        self.limbNodes = [] #list of limb node lists (default: R arm, L arm)
        self.limbDofs = [] #list of limb dof lists (default: R arm, L arm)
        self.oracles = [] #list of oracle vectors filled by an obs term and referenced by reward terms

        # skeleton capsule definitions
        self.skelCapsulesDefined = False
        self.skelCapsules = []  # list of capsule instances with two body nodes, two offset vector and two radii
        self.collisionCapsuleInfo = None  # set in updateClothCollisionStructures(capsules=True)
        self.collisionSphereInfo = None  # set in updateClothCollisionStructures()

        self.collision_filter = self.dart_world.create_collision_filter()
        if defaultHuman:
            # setup collision filtering
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[10],
                                                    self.human_skel.bodynodes[12])  # left forearm to fingers
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[5],
                                                    self.human_skel.bodynodes[7])  # right forearm to fingers
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[1],
                                                    self.human_skel.bodynodes[13])  # torso to neck
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[1],
                                                    self.human_skel.bodynodes[14])  # torso to head
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[1],
                                                    self.human_skel.bodynodes[3])  # torso to right shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[1],
                                                    self.human_skel.bodynodes[8])  # torso to left shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[13],
                                                    self.human_skel.bodynodes[3])  # neck to right shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[13],
                                                    self.human_skel.bodynodes[8])  # neck to left shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[14],
                                                    self.human_skel.bodynodes[3])  # head to right shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[14],
                                                    self.human_skel.bodynodes[8])  # head to left shoulder
            self.collision_filter.add_to_black_list(self.human_skel.bodynodes[3],
                                                    self.human_skel.bodynodes[8])  # right shoulder to left shoulder

            # DART does not automatically limit joints with any unlimited dofs
            self.human_skel.joints[4].set_position_limit_enforced(True)
            self.human_skel.joints[9].set_position_limit_enforced(True)

            #limb definitions
            self.limbNodes.append([3, 4, 5, 6, 7]) #R arm
            self.limbNodes.append([8, 9, 10, 11, 12]) #L arm

            self.limbDofs.append(range(3,11)) #R arm
            self.limbDofs.append(range(11,19)) #L arm

            self.data_driven_constraint_query = []
            self.data_driven_constraints = []
            if self.data_driven_joint_limits:
                leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.human_skel.joint('j_bicep_left'), self.human_skel.joint('elbowjL'), True)
                rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.human_skel.joint('j_bicep_right'), self.human_skel.joint('elbowjR'), False)
                self.data_driven_constraints.append(leftarmConstraint)
                self.data_driven_constraints.append(rightarmConstraint)
                leftarmConstraint.add_to_world(self.dart_world)
                rightarmConstraint.add_to_world(self.dart_world)

            #damping
            for i in range(len(self.human_skel.dofs)):
                self.human_skel.dofs[i].set_damping_coefficient(2.0) #TODO: revisit this?
            self.human_skel.dofs[0].set_damping_coefficient(4.0)
            self.human_skel.dofs[1].set_damping_coefficient(4.0)

            #SPD tuning for upperBodyDatadriven human
            for i in range(2):
                self.humanSPDController.Kp[i][i] = 10000
                self.humanSPDController.Kd[i][i] = 400

            self.humanSPDController.Kp[2][2] = 2000
            self.humanSPDController.Kd[2][2] = 70

            clav_dofs = [3, 4, 11, 12]
            for i in clav_dofs:
                self.humanSPDController.Kp[i][i] = 4000
                self.humanSPDController.Kd[i][i] = 200

            shoulder_dofs = [5, 6, 7, 13, 14, 15]
            for i in shoulder_dofs:
                self.humanSPDController.Kp[i][i] = 5000
                self.humanSPDController.Kd[i][i] = 100

            # elbows
            elbow_dofs = [8, 16]
            for i in elbow_dofs:
                self.humanSPDController.Kp[i][i] = 4000
                self.humanSPDController.Kd[i][i] = 60

            wrist_dofs = [9, 10, 17, 18]
            for i in wrist_dofs:
                self.humanSPDController.Kp[i][i] = 4000
                self.humanSPDController.Kd[i][i] = 30

            # neck
            for i in range(19, 21):
                self.humanSPDController.Kp[i][i] = 1000
                self.humanSPDController.Kd[i][i] = 30

            #self.humanSPDController.Kp *= 5.0

            if self.dual_policy:
                self.action_space = spaces.Box(np.ones(22+6*len(robot_root_dofs))* -1.0, np.ones(22+6*len(robot_root_dofs)))
                self.act_dim = 6 * len(robot_root_dofs) + 22
            elif self.is_human:
                self.action_space = spaces.Box(np.ones(22)* -1.0, np.ones(22))
                self.act_dim = 22

            self.human_action_scale = np.ones(22)*12
            self.human_action_scale[0] = 75
            self.human_action_scale[1] = 75
            self.human_action_scale[2] = 18
            self.human_action_scale[3] = 18
            self.human_action_scale[4] = 18
            self.human_action_scale[11] = 18
            self.human_action_scale[12] = 18
            self.initial_human_action_scale = np.array(self.human_action_scale)

            self.human_control_bounds = np.array([np.ones(22), np.ones(22) * -1])

        if self.print_skel_details:
            for i in range(len(self.human_skel.bodynodes)):
                print(self.human_skel.bodynodes[i])

            for i in range(len(self.human_skel.dofs)):
                print(self.human_skel.dofs[i])

            for i in range(len(self.human_skel.joints)):
                print(self.human_skel.joints[i])

        #----------------------------
        #setup the robots
        #----------------------------

        # if no alternative given, load default iiwa bot
        if robot_file is None:
            robot_file = '/iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf'

        # convert to full path
        robot_file = os.path.dirname(__file__) + "/assets" + robot_file
        if not path.exists(robot_file):
            raise IOError("File %s does not exist" % robot_file)

        # if no alternative given, load default iiwa bot
        if pybullet_robot_file is None:
            pybullet_robot_file = '/iiwa_description/urdf/iiwa7_simplified.urdf'

        # convert to full path
        pybullet_robot_file = os.path.dirname(__file__) + "/assets" + pybullet_robot_file
        if not path.exists(pybullet_robot_file):
            raise IOError("File %s does not exist" % pybullet_robot_file)

        #initialize pybullet singleton
        if self.print_skel_details:
            print("Setting up pybullet")
        self.pyBulletPhysicsClient = p.connect(p.DIRECT)
        self.pyBulletIiwa = p.loadURDF(pybullet_robot_file)
        if self.print_skel_details:
            print("Iiwa bodyID: " + str(self.pyBulletIiwa))
            print("Number of pybullet joints: " + str(p.getNumJoints(self.pyBulletIiwa)))
            for i in range(p.getNumJoints(self.pyBulletIiwa)):
                jinfo = p.getJointInfo(self.pyBulletIiwa, i)
                print(" " + str(jinfo[0]) + " " + str(jinfo[1]) + " " + str(jinfo[2]) + " " + str(jinfo[3]) + " " + str(
                    jinfo[12]))

        #setup dart iiwa skels and Iiwa class objects
        self.iiwas = []
        self.proxy_iiwa_skels = []
        for ix,root_dofs in enumerate(robot_root_dofs):
            #create a robot
            self.dart_world.add_skeleton(filename=robot_file)

            #proxy setup
            self.proxy_dart_world.add_skeleton(filename=robot_file)
            self.proxy_iiwa_skels.append(self.proxy_dart_world.skeletons[-1])

            self.iiwas.append(Iiwa(skel=self.dart_world.skeletons[-1], env=self, index=ix, root_dofs=root_dofs, active_compliance=self.active_compliance))
            if self.print_skel_details:
                self.iiwas[-1].printSkelDetails()
            #modify the action scale #TODO: edit this?
            self.robot_action_scale[6 * ix:6 * ix + 3] = np.ones(3) * 0.01  # position
            self.robot_action_scale[6 * ix + 3:6 * ix + 6] = np.ones(3) * 0.02  # orientation


        #if there is a robot, set the pybullet model to the correct orientation
        if len(self.iiwas) > 0:
            T = self.iiwas[0].skel.bodynodes[0].world_transform()
            tempFrame = pyutils.ShapeFrame()
            tempFrame.setTransform(T)
            root_quat = tempFrame.quat
            root_quat = (root_quat.x, root_quat.y, root_quat.z, root_quat.w)

            p.resetBasePositionAndOrientation(self.pyBulletIiwa, posObj=np.zeros(3), ornObj=root_quat)
            self.setPosePyBullet(self.iiwas[0].skel.q[6:])

        # compute the joint ranges for null space IK
        self.iiwa_dof_llim = np.zeros(7)
        self.iiwa_dof_ulim = np.zeros(7)
        self.iiwa_dof_jr = np.zeros(7)
        if len(robot_root_dofs) > 0:
            for i in range(7): #makes assumption of 7 dof robot (Iiwa)
                self.iiwa_dof_llim[i] = self.dart_world.skeletons[-1].dofs[i + 6].position_lower_limit()
                self.iiwa_dof_ulim[i] = self.dart_world.skeletons[-1].dofs[i + 6].position_upper_limit()
                self.iiwa_dof_jr[i] = self.iiwa_dof_ulim[i] - self.iiwa_dof_llim[i]

        if not self.dual_policy and not self.is_human:
            self.action_space = spaces.Box(np.ones(6*len(robot_root_dofs))* -1.0, np.ones(6*len(robot_root_dofs)))
            self.act_dim = 6*len(robot_root_dofs)
        # ----------------------------
        self._seed()

        #initialize reward and obs structures (to be filled by subclasses)
        self.reward_manager = RewardManager(env=self)
        self.human_obs_manager = ObservationManager()
        self.robot_obs_manager = ObservationManager()

        #initialize cloth simulation etc...
        self.clothScene = None
        self.cloth_features = [] #fill in subclasses
        self.separated_meshes = [] #fill in subclasses
        self.dressing_targets = [] #fill in subclasses
        self.handleNodes = [] #non-robot handle nodes, fill in subclasses
        self.simulateCloth = simulate_cloth
        self.deformation = 0

        if cloth_mesh_file is not None:
            # convert to full path
            cloth_mesh_file = os.path.join(os.path.dirname(__file__), "assets", cloth_mesh_file)
            if not path.exists(cloth_mesh_file):
                raise IOError("File %s does not exist" % cloth_mesh_file)

            cloth_vector_scale = np.array([cloth_scale, cloth_scale, cloth_scale])

            if cloth_mesh_state_file is not None:
                cloth_mesh_state_file = os.path.join(os.path.dirname(__file__), "assets", cloth_mesh_state_file)
                if not path.exists(cloth_mesh_file):
                    raise IOError("File %s does not exist" % cloth_mesh_state_file)
                self.clothScene = pyphysx.ClothScene(step=self.dt*2,
                                                mesh_path=cloth_mesh_file,
                                                state_path=cloth_mesh_state_file,
                                                scale=cloth_vector_scale)
            else:
                self.clothScene = pyphysx.ClothScene(step=self.dt*2,
                                                mesh_path=cloth_mesh_file,
                                                scale=cloth_vector_scale)

            self.clothScene.togglePinned(0, 0)  # turn off auto-pin

            self.clothScene.setFriction(0, cloth_friction)  # reset this anytime as desired


            self.collisionCapsuleInfo = None  # set in updateClothCollisionStructures(capsules=True)
            self.collisionSphereInfo = None  # set in updateClothCollisionStructures()
            #haptic_sensor_data allows average readings over time as well as overloaded haptic sensor locations
            self.haptic_sensor_data = {"num_sensors":22, "cloth_steps":0, "cloth_data":np.zeros(66), "rigid_steps":0, "rigid_data":np.zeros(66)}

            self.updateClothCollisionStructures(capsules=True, hapticSensors=True)

            self.clothScene.setSelfCollisionDistance(distance=0.03)
            self.clothScene.setParticleConstraintMode(mode=1)

            self.clothScene.step()
            self.clothScene.reset()

            if not self.cloth_render:
                self.clothScene.renderClothFill = False

            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.dt))
            }

            #emptied, filled, sequentially drawn each render step
            self.text_queue = []

    def _step(self, a):
        if not self.rendering:
            #clear the text queue to free memory if not actually rendering
            self.text_queue = []

        if not self.simulating:
            if self.dual_policy:
                return np.zeros(self.human_obs_manager.obs_size+self.robot_obs_manager.obs_size), 0, False, {}
            elif self.is_human:
                return np.zeros(self.human_obs_manager.obs_size), 0, False, {}
            else:
                return np.zeros(self.robot_obs_manager.obs_size), 0, False, {}

        human_a = None
        robot_a = None

        # query the "other" policy and set actions
        if self.dual_policy:
            human_a = a[:len(self.human_action_scale)]
            robot_a = a[len(self.human_action_scale):]
        elif self.is_human:
            human_a = a

            try:
                robot_obs = self.robot_obs_manager.getObs()
                robot_a, robot_a_info = self.otherPolicy.get_action(robot_obs)
                robot_a = robot_a_info['mean']
            except:
                #print("robot policy not setup, defaulting zero action")
                robot_a = np.zeros(len(self.robot_action_scale))
        else:
            robot_a = a

            # query the human policy
            try:
                human_obs = self.human_obs_manager.getObs()
                human_a, human_a_info = self.otherPolicy.get_action(human_obs)
                human_a = human_a_info['mean']
            except:
                #print("human policy not setup, defaulting zero action")
                human_a = np.zeros(len(self.human_action_scale))

        robo_action_scaled = None

        #TODO: scripted robot control?

        robo_action_clamped = np.clip(robot_a, self.robot_control_bounds[1], self.robot_control_bounds[0])
        robo_action_scaled = np.multiply(robo_action_clamped, self.robot_action_scale)

        #control, interpolate frame and compute new IK target pose for each robot
        for iiwa_ix,iiwa in enumerate(self.iiwas):
            iiwa.step(robo_action_scaled[iiwa_ix*6:iiwa_ix*6+6])

        #update cloth features
        for feature in self.cloth_features:
            feature.fitPlane()

        #TODO: update human weakness if used (maybe a function call to updateBeforeSimulation implemented by sublcasses still?)

        #update human SPD target and compute motor torques
        human_tau = self.computeHumanControl(human_a=human_a)

        self.do_simulation(human_tau, self.frame_skip)

        #update capacitive sensors
        for iiwa in self.iiwas:
            iiwa.updateCapacitiveSensor()

        reward = self.reward_manager.computeReward()
        ob = self._get_obs()

        #self.updateClothCollisionStructures(hapticSensors=True) #don't need this? It is done in simulation before each cloth sim step

        done, terminationReward = self.checkTermination(human_tau, None)

        if done:
            if math.isfinite(reward):
                reward += terminationReward
            else:
                reward = -100000
                print("caught non finite reward")

        self.numSteps += 1

        return ob, reward, done, {}

    def _get_obs(self):
        # this should return the active observation
        if self.dual_policy:
            return np.concatenate([self.human_obs_manager.getObs(), self.robot_obs_manager.getObs()])
        elif self.is_human:
            return self.human_obs_manager.getObs()
        else:
            return self.robot_obs_manager.getObs()

    def do_simulation(self, tau, n_frames):
        human_tau = np.array(tau)

        if not self.simulating:
            return

        #TODO: this is necessary if we take the reset instead of terminate path...
        #human_pre_q = np.array(self.human_skel.q)
        #human_pre_dq = np.array(self.human_skel.dq)

        #robot_pre_qs = []
        #robot_pre_dqs = []
        #for iiwa_ix, iiwa in enumerate(self.iiwas):
        #    robot_pre_qs.append(np.array(iiwa.skel.q))
        #    robot_pre_dqs.append(np.array(iiwa.skel.dq))

        self.humanRobotCollisions = [] #list of collisions between human and robots this frame

        for i in range(n_frames):
            self.human_skel.set_forces(human_tau)

            for iiwa in self.iiwas:
                if self.passive_robots:  # 0 torques for all robots
                    iiwa.skel.set_forces(np.zeros(13))
                else:
                    iiwa.skel.set_forces(iiwa.computeTorque())

            self.dart_world.step()

            if self.checkInvalidDynamics():
                #we have found simulator instability, cancel simulation and prepare for termination
                #TODO: instead should this be allowed with some reset to previous pose?
                break

            # every other step simulate cloth and update handles
            if (i % 2 == 1 and self.simulateCloth):
                #update robot handles
                for iiwa in self.iiwas:
                    iiwa.updateClothHandle()
                #TODO: update positions of non-robot handles
                #for handle_node in self.handleNodes:
                #    handle_node.

                self.updateClothCollisionStructures(hapticSensors=True)
                self.clothScene.step()

                #step handles
                for iiwa in self.iiwas:
                    if iiwa.handle_node is not None:
                        iiwa.handle_node.step() #handle FT update happens here
                        iiwa.updateClothFTSensorReading(clear=(i == 1)) #clear on the first of each set of updates
                for handle_node in self.handleNodes:
                    handle_node.step()
                #update human/cloth haptic observation
                if i == 1:
                    self.haptic_sensor_data["cloth_steps"] = 0
                    self.haptic_sensor_data["cloth_data"] = np.zeros(self.haptic_sensor_data["num_sensors"] * 3)
                self.haptic_sensor_data["cloth_steps"] += 1
                self.haptic_sensor_data["cloth_data"] += self.clothScene.getHapticSensorObs()*self.cloth_force_scale

            #check robot/human contact and update the following
            #NOTE: getCumulativeHapticForcesFromRigidContacts directly accumulates FT for handlenodes from human contact
            #   so, this is here simply to accumulate steps, clear the sensor and maybe add other stuff later...
            rigid_F = np.zeros(3)
            rigid_T = np.zeros(3)
            for iiwa in self.iiwas:
                iiwa.updateRigidFTSensorReading(F=rigid_F, T=rigid_T, clear=(i == 0))

            # check human/all collision and update the following
            human_rigid_contacts = self.getCumulativeHapticForcesFromRigidContacts()
            if i == 0:
                self.haptic_sensor_data["rigid_steps"] = 0
                self.haptic_sensor_data["rigid_data"] = np.zeros(self.haptic_sensor_data["num_sensors"] * 3)
            self.haptic_sensor_data["rigid_steps"] += 1
            self.haptic_sensor_data["rigid_data"] += human_rigid_contacts

    def reset_model(self):

        #random seeding
        seeds = []

        if False:
            try:
                seed = seeds[self.reset_number]
            except:
                print("out of seeds, exiting")
                exit()
                seed = self.reset_number

            print("Seeding: " + str(seed))
            random.seed(seed)
            self.np_random.seed(seed)
            np.random.seed(seed)
            self.setSeed = seed

        self.dart_world.reset()

        if self.simulateCloth:
            self.clothScene.reset()

        for iiwa in self.iiwas:
            iiwa.setRestPose()
            if iiwa.iiwa_frame_controller is not None:
                iiwa.iiwa_frame_controller.reset()

        self.humanSPDIntperolationTarget = np.zeros(self.human_skel.ndofs)

        self.human_obs_manager.reset()
        self.robot_obs_manager.reset()
        self.reward_manager.reset()

        #reset the MPC collision structures
        self.human_collision_warning = 1.0
        for iiwa in self.iiwas:
            iiwa.near_collision = 1.0

        self.additionalResets()

        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        self.reset_number += 1
        self.numSteps = 0

        return self._get_obs()

    def additionalResets(self):
        #TODO: overwrite this in the subclasses to add specific reset functionality
        pass

    def _reset(self):
        ob = self.reset_model()
        return ob

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'human':
            self._get_viewer().runSingleStep()

    def extraRenderFunction(self):

        #TODO: remove after testing
        #render the offset of the head
        #h_off = np.array([0,0.25,0])
        #h_off = np.array([0,0,0])
        #h_off_p = self.human_skel.bodynodes[13].to_world(h_off)
        #renderUtils.drawArrow(p0=h_off_p+np.array([0.5,0.5,0]),p1=h_off_p)

        #cloth_centroid = self.clothScene.getVertexCentroid(cid=0)
        #renderUtils.drawArrow(p0=cloth_centroid + np.array([0.5, 0.5, 0]), p1=cloth_centroid)


        if self.proxy_render:
            renderUtils.drawLines(pyutils.getRobotLinks(self.proxy_human_skel, pose=self.proxy_human_skel.q))

        if self.manual_human_control:
            renderUtils.drawLines(pyutils.getRobotLinks(self.human_skel,pose=self.human_manual_target))

        if self.manual_robot_control:
            lines = []
            for iiwa in self.iiwas:
                iiwa.manualFrameTarget.drawArrowFrame(size=0.4)
                lines.append([iiwa.manualFrameTarget.org, iiwa.frameInterpolator["target_pos"]])
            renderUtils.drawLines(lines)

        #render a central axis
        renderUtils.drawArrowAxis(org=np.zeros(3), v0=np.array([1.0,0,0]), v1=np.array([0,1.0,0]), v2=np.array([0,0,1.0]), scale=0.5)

        #render the valid initial point box
        if False:
            r_pivot = self.iiwas[0].skel.bodynodes[3].to_world(np.zeros(3))
            depth_offset = 0.15
            c0 = np.array([-0.7, -0.7, -0.7])
            c0 = r_pivot+c0
            c1 = np.array([0.7, 0.7, 0.7])
            c1 = r_pivot+c1
            c1[0] = min((r_pivot[0] - depth_offset), c1[0])
            c1[2] = min((r_pivot[2] - depth_offset), c1[2])
            cen = (c0+c1)/2.0
            dim = c1-c0
            renderUtils.drawBox(cen=cen, dim=dim, fill=False)
            renderUtils.drawSphere(pos=r_pivot,rad=0.7,solid=False)

        #render the SPD target for the human
        if True:
            q = np.array(self.human_skel.q)
            dq = np.array(self.human_skel.dq)

            for i in range(2):
                if i==0:
                    self.human_skel.set_positions(self.humanSPDIntperolationTarget)
                    renderUtils.setColor(color=[0.8, 0.6, 0.6])
                else:
                    self.human_skel.set_positions(self.humanSPDController.target)
                    renderUtils.setColor(color=[0.6, 0.8, 0.6])

                if self.skelCapsulesDefined:
                    #renderUtils.setColor(color=[0.8, 0.6, 0.6])
                    #print(self.skelCapsules)
                    for capsule in self.skelCapsules:
                        p0 = self.human_skel.bodynodes[capsule[0]].to_world(capsule[2])
                        p1 = self.human_skel.bodynodes[capsule[3]].to_world(capsule[5])
                        renderUtils.drawCapsule(p0=p0, p1=p1, r0=capsule[1], r1=capsule[4])

            self.human_skel.set_positions(q)
            self.human_skel.set_velocities(dq)

        for iiwa in self.iiwas:
            # draw robot capactive sensors
            iiwa.capacitiveSensor.draw()

            # draw interpolation frames
            f_int = iiwa.frameInterpolator
            if f_int["active"]:
                renderFrame = pyutils.ShapeFrame()
                renderFrame.setOrg(org=f_int["target_pos"])
                renderFrame.orientation = np.array(f_int["target_frame"])
                renderFrame.updateQuaternion()
                renderFrame.drawArrowFrame(size=0.5)

                renderUtils.setColor(color=[0, 0, 0])
                renderUtils.drawLineStrip(points=[iiwa.ik_target.org, f_int["target_pos"]])
                renderUtils.drawSphere(pos=renderFrame.org + renderFrame.org - renderFrame.toGlobal(p=f_int["localOffset"]), rad=0.02)

            iiwa.ik_target.drawArrowFrame(size=0.2)

            #render the FT sensor reading
            cur_FT = iiwa.getFTSensorReading()
            renderUtils.setColor(color=[1.0, 0, 1.0])
            if len(self.humanRobotCollisions) > 0:
                renderUtils.setColor(color=[1.0,0,0])
            if iiwa.handle_node is not None:
                renderUtils.drawArrow(p0=iiwa.handle_node.org, p1=iiwa.handle_node.org + cur_FT[:3] * 0.1)

            #render manual frame control
            if iiwa.iiwa_frame_controller is not None:
                iiwa.iiwa_frame_controller.draw()

        #draw necessary observation term components
        self.human_obs_manager.draw()
        self.robot_obs_manager.draw()
        self.reward_manager.draw()

        #inner bicep indicator strips
        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.human_skel.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.human_skel.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.human_skel.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.human_skel.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #draw cloth features
        for feature in self.cloth_features:
            feature.drawProjectionPoly(renderNormal=True, renderBasis=False,fill=False)

        # render geodesic
        if False:
            sm_ix = 0 #default left feature geodesic
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separated_meshes[sm_ix].nodes[v + self.separated_meshes[sm_ix].numv].geodesic
                side0geo = self.separated_meshes[sm_ix].nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separated_meshes[sm_ix].maxGeo, value=self.separated_meshes[sm_ix].maxGeo-side0geo))
                renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separated_meshes[sm_ix].maxGeo, value=self.separated_meshes[sm_ix].maxGeo-side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)


        #TEXT
        m_viewport = self.viewer.viewport

        self.clothScene.drawText(x=15., y=15, text="Time = " + str(self.numSteps * self.dt * self.frame_skip), color=(0., 0, 0))

        #draw the text queue
        textHeight = 15
        textLines = 2
        renderUtils.setColor(color=[0., 0, 0])
        for text in self.text_queue:
            self.clothScene.drawText(x=15., y=textLines*textHeight, text=text, color=(0., 0, 0))
            textLines += 1

        #empty the queue
        self.text_queue = []

        if self.detail_render:
            #draw forces
            far_left = 360.0
            far_top = 12
            for d in range(self.human_skel.ndofs):
                self.clothScene.drawText(x=far_left, y=self.viewer.viewport[3] - far_top - 13 - d*20, text="%0.2f" % (-self.human_action_scale[d],), color=(0., 0, 0))
                self.clothScene.drawText(x=far_left+85, y=self.viewer.viewport[3] - far_top - 13 - d*20, text="%0.2f" % (self.human_skel.forces()[d],), color=(0., 0, 0))
                self.clothScene.drawText(x=far_left+185., y=self.viewer.viewport[3] - far_top - 13 - d*20, text="%0.2f" % (self.human_action_scale[d],), color=(0., 0, 0))

                tval = (self.human_skel.forces()[d] + self.human_action_scale[d]) / (self.human_action_scale[d] + self.human_action_scale[d])
                renderUtils.drawProgressBar(topLeft=[far_left+60, self.viewer.viewport[3] - far_top - d * 20], h=16, w=120, progress=tval, origin=0.5, color=[1.0, 0.0, 0])

            #draw pose with target
            renderUtils.renderDofs(self.human_skel,self.humanSPDIntperolationTarget,True,_topLeft=[15, self.viewer.viewport[3] - 12])

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = NoRenderWindow(self.dart_world,title=None)
            if self.rendering:
                self.viewer = StaticClothGLUTWindow(self.dart_world, title=None, clothScene=self.clothScene, extraRenderFunc=self.extraRenderFunction, inputFunc=self.inputFunc, resetFunc=self.reset_model, env=self)
                self.viewer.scene.add_camera(Trackball(theta=-45.0, phi=0.0, zoom=0.1), 'gym_camera')
                self.viewer.scene.set_camera(self.viewer.scene.num_cameras() - 1)
                if not self.dart_render:
                    self.viewer.renderWorld = False
            self.viewer.run(_width=self.screen_size[0], _height=self.screen_size[1], _show_window=self.rendering)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            # recording angle from front of person with robot on left side...
            self._get_viewer().scene.tb._trans = [0.34000000000000019, 0.020000000000000004, -2.0999999999999988]
            rot = [-0.094887037321912837, -0.91548784322523225, -0.071299301298955647, 0.38444098206273875]
            pyutils.setTrackballOrientation(self.viewer.scene.tb, rot)

    def _seed(self, seed=None):
        self.setSeed = seed
        self.np_random, seed = seeding.np_random(seed)
        if seed is None:
            #this ensures that if global numpy was seeded and no seed is provided some seed transfers here (necessary for multi-threaded variation)
            state = np.random.get_state()
            self.np_random.set_state(state)
        return [seed]

    def _getFile(self):
        return __file__

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def checkTermination(self, tau, obs):
        if self.checkInvalidDynamics():
            return True, self.reward_manager.reward_limits[0]*self.task_horizon-self.numSteps
        #override with additional termiantion criteria if necessary
        return False, 0

    def checkInvalidDynamics(self):
        if not np.isfinite(self.human_skel.q).all():
            print("Infinite value detected...")# + str(self.robot_skeleton.q))
            return True
        elif np.amax(np.absolute(self.human_skel.q)) > 5:
            print("Detecting potential instability...")# + str(self.robot_skeleton.q))
            return True
        for iiwa_ix,iiwa in enumerate(self.iiwas):
            if not np.isfinite(iiwa.skel.q).all():
                print("Infinite value detected (robot "+str(iiwa_ix)+")...")# + str(self.dart_world.skeletons[0].q))
                return True
            elif np.amax(np.absolute(iiwa.skel.q)) > 5:
                print("Detecting potential instability (robot "+str(iiwa_ix)+")...")# + str(self.dart_world.skeletons[0].q))
                return True
        return False

    # set a pose in the pybullet simulation env
    def setPosePyBullet(self, pose):
        count = 0
        for i in range(p.getNumJoints(self.pyBulletIiwa)):
            jinfo = p.getJointInfo(self.pyBulletIiwa, i)
            if (jinfo[3] > -1):
                p.resetJointState(self.pyBulletIiwa, i, pose[count])
                count += 1

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        # collision spheres creation (override this for different body interaction with cloth)
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0., 0, 0])
        cs0 = self.human_skel.bodynodes[1].to_world(z)
        cs1 = self.human_skel.bodynodes[2].to_world(z)
        cs2 = self.human_skel.bodynodes[14].to_world(z)
        cs3 = self.human_skel.bodynodes[14].to_world(np.array([0, 0.175, 0]))
        cs4 = self.human_skel.bodynodes[4].to_world(z)
        cs5 = self.human_skel.bodynodes[5].to_world(z)
        cs6 = self.human_skel.bodynodes[6].to_world(z)
        cs7 = self.human_skel.bodynodes[7].to_world(z)
        cs8 = self.human_skel.bodynodes[7].to_world(fingertip)
        cs9 = self.human_skel.bodynodes[9].to_world(z)
        cs10 = self.human_skel.bodynodes[10].to_world(z)
        cs11 = self.human_skel.bodynodes[11].to_world(z)
        cs12 = self.human_skel.bodynodes[12].to_world(z)
        cs13 = self.human_skel.bodynodes[12].to_world(fingertip)
        csVars0 = np.array([0.15, -1, -1, 0, 0, 0])
        csVars1 = np.array([0.07, -1, -1, 0, 0, 0])
        csVars2 = np.array([0.1, -1, -1, 0, 0, 0])
        csVars3 = np.array([0.1, -1, -1, 0, 0, 0])
        csVars4 = np.array([0.065, -1, -1, 0, 0, 0])
        csVars5 = np.array([0.05, -1, -1, 0, 0, 0])
        csVars6 = np.array([0.0365, -1, -1, 0, 0, 0])
        csVars7 = np.array([0.04, -1, -1, 0, 0, 0])
        csVars8 = np.array([0.046, -1, -1, 0, 0, 0])
        csVars9 = np.array([0.065, -1, -1, 0, 0, 0])
        csVars10 = np.array([0.05, -1, -1, 0, 0, 0])
        csVars11 = np.array([0.0365, -1, -1, 0, 0, 0])
        csVars12 = np.array([0.04, -1, -1, 0, 0, 0])
        csVars13 = np.array([0.046, -1, -1, 0, 0, 0])
        collisionSpheresInfo = np.concatenate(
            [cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7,
             csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13,
             csVars13]).ravel()

        sphereToBodynodeMapping = [1, 2, 14, 14, 4, 5, 6, 7, 7, 9, 10, 11, 12, 12]
        offsetsToBodynodeMapping = {3: np.array([0, 0.175, 0]), 8: fingertip, 13: fingertip}

        # inflate collision objects
        # for i in range(int(len(collisionSpheresInfo)/9)):
        #    collisionSpheresInfo[i*9 + 3] *= 1.15

        self.collisionSphereInfo = np.array(collisionSpheresInfo)
        # collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        if np.isnan(np.sum(collisionSpheresInfo)):  # this will keep nans from propagating into PhysX resulting in segfault on reset()
            return
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)

        if capsules is True:
            # collision capsules creation
            collisionCapsuleInfo = np.zeros((14, 14))
            collisionCapsuleInfo[0, 1] = 1
            collisionCapsuleInfo[1, 2] = 1
            collisionCapsuleInfo[1, 4] = 1
            collisionCapsuleInfo[1, 9] = 1
            collisionCapsuleInfo[2, 3] = 1
            collisionCapsuleInfo[4, 5] = 1
            collisionCapsuleInfo[5, 6] = 1
            collisionCapsuleInfo[6, 7] = 1
            collisionCapsuleInfo[7, 8] = 1
            collisionCapsuleInfo[9, 10] = 1
            collisionCapsuleInfo[10, 11] = 1
            collisionCapsuleInfo[11, 12] = 1
            collisionCapsuleInfo[12, 13] = 1
            collisionCapsuleBodynodes = -1 * np.ones((14, 14))
            collisionCapsuleBodynodes[0, 1] = 1
            collisionCapsuleBodynodes[1, 2] = 13
            collisionCapsuleBodynodes[1, 4] = 3
            collisionCapsuleBodynodes[1, 9] = 8
            collisionCapsuleBodynodes[2, 3] = 14
            collisionCapsuleBodynodes[4, 5] = 4
            collisionCapsuleBodynodes[5, 6] = 5
            collisionCapsuleBodynodes[6, 7] = 6
            collisionCapsuleBodynodes[7, 8] = 7
            collisionCapsuleBodynodes[9, 10] = 9
            collisionCapsuleBodynodes[10, 11] = 10
            collisionCapsuleBodynodes[11, 12] = 11
            collisionCapsuleBodynodes[12, 13] = 12
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo, collisionCapsuleBodynodes)
            self.collisionCapsuleInfo = np.array(collisionCapsuleInfo)

        if not self.skelCapsulesDefined:
            self.skelCapsulesDefined = True
            for i in range(len(self.collisionCapsuleInfo)):
                for j in range(len(collisionCapsuleInfo)):
                    if collisionCapsuleInfo[i, j] == 1:
                        offset_i = np.zeros(3)
                        try:
                            offset_i = np.array(offsetsToBodynodeMapping[i])
                        except:
                            pass
                        offset_j = np.zeros(3)
                        try:
                            offset_j = np.array(offsetsToBodynodeMapping[j])
                        except:
                            pass

                        self.skelCapsules.append([sphereToBodynodeMapping[i], collisionSpheresInfo[i*9 + 3], offset_i,
                                                  sphereToBodynodeMapping[j], collisionSpheresInfo[j*9 + 3], offset_j]
                                                 )  # bodynode1, radius1, offset1, bodynode2, radius2, offset2


        if hapticSensors is True:
            # hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            # hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate(
                [cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33),
                 LERP(cs5, cs6, 0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10,
                 LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            hapticSensorRadii = np.array(
                [csVars0[0], csVars1[0], csVars2[0], csVars3[0], csVars4[0], LERP(csVars4[0], csVars5[0], 0.33),
                 LERP(csVars4[0], csVars5[0], 0.66), csVars5[0], LERP(csVars5[0], csVars6[0], 0.33),
                 LERP(csVars5[0], csVars6[0], 0.66), csVars6[0], csVars7[0], csVars8[0], csVars9[0],
                 LERP(csVars9[0], csVars10[0], 0.33), LERP(csVars9[0], csVars10[0], 0.66), csVars10[0],
                 LERP(csVars10[0], csVars11[0], 0.33), LERP(csVars10[0], csVars11[0], 0.66), csVars11[0], csVars12[0],
                 csVars13[0]])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            self.clothScene.setHapticSensorRadii(hapticSensorRadii)

    def computeHumanControl(self, human_a, dt=None, target_dof_limit_percent=0.55):
        '''
        :param human_a:
        :param dt:
        :param target_dof_limit_percent: what percent of the total joint range is the target limited to (prevents over-commit)
        :return: human tau
        '''
        # if not dt is given, use the env default timestep
        if dt is None:
            dt = self.dt * self.frame_skip

        self.human_collision_warning = min(self.human_collision_warning+dt, 1.0)

        human_control = np.zeros(self.human_skel.ndofs)

        pos_upper_lim = self.human_skel.position_upper_limits()
        pos_lower_lim = self.human_skel.position_lower_limits()
        pos_range = pos_upper_lim - pos_lower_lim
        for rix,r in enumerate(pos_range):
            if math.isinf(r):
                pos_range[rix] = 3.0 #default to 2.5 joint range for unlimited joints (should be plenty)

        if self.manual_human_control:
            self.humanSPDIntperolationTarget = np.array(self.human_manual_target)
        else:
            self.humanSPDIntperolationTarget += human_a * 0.1 #control bounds [-1,1]*0.1

        for d in range(self.human_skel.ndofs):
            # limit close to current pose
            qdof = self.human_skel.q[d]
            diff = qdof - self.humanSPDIntperolationTarget[d]
            allowed_diff = target_dof_limit_percent * pos_range[d]
            if abs(diff) > allowed_diff:
                if diff > 0:
                    self.humanSPDIntperolationTarget[d] = qdof - allowed_diff
                else:
                    self.humanSPDIntperolationTarget[d] = qdof + allowed_diff

            # joint limiting
            self.humanSPDIntperolationTarget[d] = min(self.humanSPDIntperolationTarget[d], pos_upper_lim[d])
            self.humanSPDIntperolationTarget[d] = max(self.humanSPDIntperolationTarget[d], pos_lower_lim[d])

        #TODO: check this section - always reset the target to the current pose + interpolated pose deviation
        #target_diff = self.humanSPDIntperolationTarget - self.humanSPDController.target
        target_diff = self.humanSPDIntperolationTarget - self.human_skel.q

        for d in range(len(target_diff)):
            maxChange = self.humanSPDInterpolationRate * dt * pos_range[d]  # defines the maximum joint range percentage change per second of simulation
            maxChange *= 7.0 #TODO: be aware of this
            if abs(target_diff[d]) > maxChange:
                self.humanSPDController.target[d] = self.human_skel.q[d] + (target_diff[d] / abs(target_diff[d])) * maxChange
            else:
                self.humanSPDController.target[d] = self.humanSPDIntperolationTarget[d]

        if not self.checkProxyPose(skel_ix=-1, pose=self.humanSPDController.target):
            #self.humanSPDController.target = np.array(self.human_skel.q)
            self.human_collision_warning = 0.0
            self.text_queue.append("MPC COLLISION_HUMAN_WARNING")

        human_control = self.humanSPDController.query(None)

        human_clamped_control = np.array(human_control)

        for i in range(len(human_clamped_control)):
            if human_clamped_control[i] > self.human_action_scale[i]:
                human_clamped_control[i] = self.human_action_scale[i]
            if human_clamped_control[i] < -self.human_action_scale[i]:
                human_clamped_control[i] = -self.human_action_scale[i]

        return human_clamped_control

    def getCumulativeHapticForcesFromRigidContacts(self):
        #compute contact forces between the body and objects
        #fill the iiwa handle data-structures
        self.collisionResult.update()
        sensor_pos = self.clothScene.getHapticSensorLocations()
        sensor_rad = self.clothScene.getHapticSensorRadii()
        relevant_contacts = []
        for ix, c in enumerate(self.collisionResult.contacts):
            # add a contact if the human skeleton is involved
            if (c.skel_id1 == self.human_skel.id or c.skel_id2 == self.human_skel.id):
                relevant_contacts.append(c)

        forces = []
        for i in range(self.clothScene.getNumHapticSensors()):
            forces.append(np.zeros(3))

        for ix, c in enumerate(relevant_contacts):
            if (c.skel_id1 != c.skel_id2):
                # the contact is between the human skel and another object
                # find the closest sensor to activate
                best_hs = self.clothScene.getClosestNHapticSpheres(n=1, pos=c.point)[0]
                vp = sensor_pos[3 * best_hs: best_hs*3 + 3] - c.point
                vpn = vp / np.linalg.norm(vp)
                fn = c.force / np.linalg.norm(c.force)
                toward_human_f = np.zeros(3)
                if (vpn.dot(fn) > -vpn.dot(fn)):  # force pointing toward the sensor is correct
                    forces[best_hs] += c.force
                    toward_human_f = np.array(c.force)
                else:  # reverse a force pointing away from the sensor
                    forces[best_hs] += -c.force
                    toward_human_f = np.array(-c.force)

                #also check if this is contact with the robot
                for iiwa in self.iiwas:
                    if (c.skel_id1 == iiwa.skel.id or c.skel_id2 == iiwa.skel.id):
                        self.humanRobotCollisions.append(toward_human_f)
                        if iiwa.handle_node is not None:
                            #check if this is is read by ft sensor (contact with the FT bodynode) and if so, add it
                            if c.bodynode_id1 == iiwa.skel.bodynodes[iiwa.handle_bodynode].id or c.bodynode_id2 == iiwa.skel.bodynodes[iiwa.handle_bodynode].id:
                                iiwa.FT_sensor["rigid_F"] += -toward_human_f
                                iiwa.FT_sensor["rigid_T"] += np.cross(c.point-iiwa.handle_node.org, -toward_human_f)
            else:
                # the contact is between the human and itself
                # find the two closest sensors to activate
                best_hs = self.clothScene.getClosestNHapticSpheres(n=2, pos=c.point)
                for i in range(2):
                    vp = sensor_pos[3 * best_hs[i]: best_hs[i]*3 + 3] - c.point
                    vpn = vp / np.linalg.norm(vp)
                    fn = c.force / np.linalg.norm(c.force)
                    if (vpn.dot(fn) > -vpn.dot(fn)):  # force pointing toward the sensor is correct
                        forces[best_hs[i]] += c.force
                    else:  # reverse a force pointing away from the sensor
                        forces[best_hs[i]] += -c.force

        result = np.zeros(len(forces)*3)
        for ix,f in enumerate(forces):
            #f /= mag_scale
            #f_mag = np.linalg.norm(f)
            #if(f_mag > 1.0):
            #    f /= f_mag
            result[ix*3:ix*3+3] = f
        return result

    def getValidRandomPose(self, verbose=True, symmetrical=False):
        #use joint limits and data driven joint limit function to pick a valid random pose
        upper = self.human_skel.position_upper_limits()
        lower = self.human_skel.position_lower_limits()
        for d in range(len(upper)):
            if math.isinf(upper[d]):
                upper[d] = 2.0
            if math.isinf(lower[d]):
                lower[d] = -2.0
        valid = False
        counter = 0
        new_pose = None
        init_q = np.array(self.human_skel.q)
        init_dq = np.array(self.human_skel.dq)
        while (not valid):
            valid = True
            new_pose = np.random.uniform(lower, upper)
            #new_pose[0] = 0
            #new_pose[1] = 0
            #new_pose[2] = 0
            if symmetrical:
                new_pose[11:19] = new_pose[3:11]
                new_pose[13] *= -1
                new_pose[15] *= -1
            self.human_skel.set_positions(new_pose)
            self.dart_world.step()

            efL = self.human_skel.bodynodes[12].to_world(np.zeros(3))
            if efL[2] > -0.2:
                valid = False

            # print("constraint indices" + str(self.dataDrivenConstraints))
            constraintQuery = []
            for constraint in self.data_driven_constraints:
                constraintQuery.append(constraint.query(self.dart_world, False))
            if verbose:
                print("constraint query: " + str(constraintQuery))
            if (constraintQuery[0] < 0 or constraintQuery[1] < 0):
                valid = False
            if valid:
                self.collisionResult.update()
                #check collisions
                for ix, c in enumerate(self.collisionResult.contacts):
                    if (c.skel_id1 == self.human_skel.id and c.skel_id2 == self.human_skel.id):
                        valid = False
            if counter%10 == 0:
                print("tried " + str(counter) + " times...")
            self.dart_world.reset()
            counter += 1
        self.human_skel.set_positions(init_q)
        self.human_skel.set_velocities(init_dq)
        self.human_skel.set_forces(np.zeros(self.human_skel.ndofs))
        if verbose:
            print("found a valid pose in " + str(counter) + " tries.")
        return new_pose

    def checkProxyPose(self, skel_ix, pose):
        #starttime = time.time()
        #check if given the state of other skeletons in the world a given pose is in collision
        skel_poses = []
        skel_vels = []
        for iiwa in self.iiwas:
            skel_poses.append(iiwa.skel.q)
            skel_vels.append(iiwa.skel.dq)
        skel_poses.append(self.human_skel.q)
        skel_vels.append(self.human_skel.dq)

        this_skel = None
        if skel_ix == -1:
            this_skel = self.proxy_human_skel
        else:
            this_skel = self.proxy_iiwa_skels[skel_ix]

        #do the check
        pose_ok = True
        self.proxy_dart_world.reset()
        for iiwa_ix, iiwa in enumerate(self.iiwas):
            self.proxy_iiwa_skels[iiwa_ix].set_positions(skel_poses[iiwa_ix])
        self.proxy_human_skel.set_positions(skel_poses[-1])
        this_skel.set_positions(pose)

        self.proxy_dart_world.step()
        self.proxy_collisionResult.update()
        for ix, c in enumerate(self.proxy_collisionResult.contacts):
            #if (c.skel_id1 == self.proxy_human_skel.id or c.skel_id2 == self.proxy_human_skel.id):
            if (c.skel_id1 == this_skel.id or c.skel_id2 == this_skel.id):
                pose_ok = False

        return pose_ok

    def initializeProxyWorld(self, filename=None):

        if filename is None:
            filename = 'UpperBodyCapsules_datadriven_proxy.skel'

        #convert to full path
        filename = os.path.join(os.path.dirname(__file__), "assets", filename)
        if not path.exists(filename):
            raise IOError("File %s does not exist" % filename)

        self.proxy_dart_world = pydart.World(self.dt, filename)
        self.proxy_collisionResult = CollisionResult.CollisionResult(self.proxy_dart_world)
        self.proxy_human_skel = self.proxy_dart_world.skeletons[-1]

        #self.proxy_dart_world.set_gravity(np.array([0., -9.8, 0]))

        #self.proxy_human_skel.set_self_collision_check(True)
        #self.proxy_human_skel.set_adjacent_body_check(False)

    def saveObjState(self, filename=None):
        print("Trying to save the object state")
        print("filename: " + str(filename))
        if filename is None:
            filename = "objState"
        self.clothScene.saveObjState(filename, 0)

    def saveSkelState(self, skel, filename=None):
        print("saving skel state")
        if filename is None:
            filename = "skelState"
        print("filename " + str(filename))
        f = open(filename, 'w')
        for ix,dof in enumerate(skel.q):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))

        f.write("\n")

        for ix,dof in enumerate(self.skel.dq):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))
        f.close()

    def loadSkelState(self, skel, filename=None):
        openFile = "skelState"
        if filename is not None:
            openFile = filename
        f = open(openFile, 'r')
        qpos = np.zeros(skel.ndofs)
        qvel = np.zeros(skel.ndofs)
        for ix, line in enumerate(f):
            if ix > 1: #only want the first 2 file lines
                break
            words = line.split()
            if(len(words) != skel.ndofs):
                break
            if(ix == 0): #position
                qpos = np.zeros(skel.ndofs)
                for ixw, w in enumerate(words):
                    qpos[ixw] = float(w)
            else: #velocity (if available)
                qvel = np.zeros(skel.ndofs)
                for ixw, w in enumerate(words):
                    qvel[ixw] = float(w)

        skel.set_positions(qpos)
        skel.set_velocities(qvel)
        f.close()

    def saveRandomState(self, directory=None, max_state_number=100):
        state_number = random.randint(0,max_state_number)
        if directory is None:
            directory = self.experiment_prefix + self.experiment_directory + "/states/"
        self.saveSkelState(self.human_skel, directory+"human_%05d" % state_number)
        for iiwa in self.iiwas:
            self.saveSkelState(iiwa.skel, directory + "iiwa"+str(iiwa.index)+"_%05d" % state_number)

        self.saveObjState(directory+"cloth_%05d" % state_number)

    def loadRandomState(self, directory=None, max_state_number=100):
        state_number = random.randint(0, max_state_number)
        if directory is None:
            directory = self.experiment_prefix + self.experiment_directory + "/states/"
        self.loadSkelState(self.human_skel, directory+"human_%05d" % state_number)
        for iiwa in self.iiwas:
            self.loadSkelState(iiwa.skel, directory + "iiwa"+str(iiwa.index)+"_%05d" % state_number)

        #load the cloth
        self.clothScene.loadObjState(filename=directory+"cloth_%05d" % state_number)

    def fillSavedStates(self, directory=None, max_state_number=100):
        if directory is None:
            directory = self.experiment_prefix + self.experiment_directory + "/states/"
        self.saveRandomState(directory, max_state_number=1)
        for s in range(1,max_state_number):
            shutil.copy2( directory+"human_%05d" % 0, directory+"human_%05d" % s)
            for iiwa in self.iiwas:
                shutil.copy2( directory + "iiwa"+str(iiwa.index)+"_%05d" % 0, directory + "iiwa"+str(iiwa.index)+"_%05d" % s)
            shutil.copy2( directory+"cloth_%05d" % 0, directory+"cloth_%05d" % s)