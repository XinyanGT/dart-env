# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_assist_base import *
import random
import time
import math

import pydart2.joint as Joint
import pydart2.collision_result as CollisionResult

import pybullet as p
import pybullet_data
import os
import pickle
import joblib

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.pyutils import LERP
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class Controller(object):
    def __init__(self, env, skel, policyfilename=None, name=None, obs_subset=[]):
        self.env = env #needed to set env state variables on setup for use
        self.skel = skel
        self.name = name
        prefix = os.path.dirname(os.path.abspath(__file__))
        prefix = os.path.join(prefix, '../../../../rllab/data/local/experiment/')
        if name is None:
            self.name = policyfilename
        self.policy = None
        if policyfilename is not None:
            self.policy = pickle.load(open(prefix+policyfilename + "/policy.pkl", "rb"))
        self.obs_subset = obs_subset #list of index,length tuples to slice obs for input

    def query(self, obs):
        obs_subset = np.array([])
        for s in self.obs_subset:
            obs_subset = np.concatenate([obs_subset, obs[s[0]:s[0]+s[1]]]).ravel()
        a, a_info = self.policy.get_action(obs_subset)
        a = a_info['mean']
        return a

    def setup(self):
        print("base setup ... overwrite this for specific control requirements")
        #TODO: subclasses for setup requirements

    def update(self):
        print("default update")
        #TODO: subclasses update targets, etc...

    def transition(self):
        #return true when a controller detects task completion to transition to the next controller
        return False

class SPDController(Controller):
    def __init__(self, env, skel, target=None, timestep=0.01, startDof=6, ckp=30000.0, ckd=100.0):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        self.startDof = startDof
        Controller.__init__(self, env, skel, policyfilename, name, obs_subset)

        self.h = timestep
        #self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs-startDof
        self.qhat = self.skel.q
        #self.Kp = np.diagflat([30000.0] * (ndofs))
        #self.Kd = np.diagflat([100.0] * (ndofs))

        self.Kp = np.diagflat([ckp] * (ndofs))
        self.Kd = np.diagflat([ckd] * (ndofs))

        #self.Kd[0, 6] = 1.0

        #self.Kd[6,6] = 1.0

        #self.Kp[0][0] = 2000.0
        #self.Kd[0][0] = 100.0
        #self.Kp[1][1] = 2000.0
        #self.Kp[2][2] = 2000.0
        #self.Kd[2][2] = 100.0
        #self.Kp[3][3] = 2000.0
        #self.Kp[4][4] = 2000.0

        '''
        for i in range(ndofs):
            if i ==9 or i==10 or i==17 or i==18:
                self.Kd[i][i] *= 0.01
                self.Kp[i][i] *= 0.01
        '''

        #print(self.Kp)
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        #cur_q = np.array(self.skel.q)
        #self.env.loadCharacterState(filename="characterState_regrip")
        self.target = np.array(self.skel.q[self.startDof:])
        #self.env.restPose = np.array(self.target)
        #self.target = np.array(self.skel.q)
        #self.env.robot_skeleton.set_positions(cur_q)

        a=0

    def update(self):
        #if self.env.handle_node is not None:
        #    self.env.handle_node.clearHandles();
        #    self.env.handle_node = None
        a=0

    def transition(self):
        return False

    def query(self, obs, skel=None):

        #override which skeleton is controlled if necessary
        if skel is None:
            skel = self.skel

        #SPD
        self.qhat = self.target
        p = -self.Kp.dot(skel.q[self.startDof:] + skel.dq[self.startDof:] * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq[self.startDof:])
        b = -skel.c[self.startDof:] + p + d + skel.constraint_forces()[self.startDof:]
        A = skel.M[self.startDof:, self.startDof:] + self.Kd * self.h

        #print(np.linalg.cond(A))
        #near singular matrix check ... remove for speed
        #if not np.linalg.cond(A) < 1/sys.float_info.epsilon:
        #    print("Near singular...")

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        #tau = p - self.Kd.dot(skel.dq[6:] + x * self.h)
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class IiwaFrameController:
    def __init__(self, env):
        self.env = env

    def query(self):
        #note:  control[3] = about red (pitch)
        #       control[4] = about blue (yaw)
        #       control[5] = about green (roll)
        return np.zeros(3)

    def reset(self):
        #if any state is held, reset it
        pass

    def draw(self):
        #any necessary visualizations for the controller
        pass

class IiwaApproachHoverProceedAvoidController(IiwaFrameController):
    # This controller approaches the character and hovers at a reasonable range to allow the human to dress the sleeve
    def __init__(self, env, robot_ix, target_node, node_offset, distance, noise=0.0, control_fraction=0.3, slack=(0.1, 0.075), hold_time=1.0, avoidDist=0.1):
        IiwaFrameController.__init__(self, env)
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
        self.robot_ix = robot_ix

    def query(self):
        control = np.zeros(6)
        #noiseAddition = np.random.uniform(-self.env.robot_action_scale, self.env.robot_action_scale) * self.noise
        worldTarget = self.env.robot_skeleton.bodynodes[self.target_node].to_world(self.nodeOffset)
        if self.proceeding:
            worldTarget = np.array(self.target_position)
        iiwaEf = self.env.iiwa_skels[self.robot_ix].bodynodes[8].to_world(np.array([0, 0, 0.05]))
        iiwa_frame_org = self.env.frameInterpolator[self.robot_ix]["target_pos"]
        t_disp = worldTarget - iiwa_frame_org
        t_dist = np.linalg.norm(t_disp)
        t_dir = t_disp / t_dist

        # move down to level with target by default (should counter the avoidance)
        y_disp = worldTarget[1] - iiwa_frame_org[1]
        if y_disp < 0:
            control[1] = y_disp

        limbsContained = True
        for prog in self.env.individualLimbProgress:
            if prog < 0:
                limbsContained = False

        if not limbsContained and not self.proceeding:
            #move toward the target distance
            if t_dist > (self.slack[0] + self.distance):
                control[:3] = t_dir * (t_dist - self.distance + self.slack[0])
            elif t_dist < (self.distance - self.slack[0]):
                control[:3] = t_dir * (t_dist - self.distance - self.slack[0])
        else:
            self.time_held += self.env.dt
            if self.time_held > self.hold_time:
                if not self.proceeding:
                    self.proceeding = True
                    self.target_position = np.array(worldTarget)
                # now begin moving toward the arm again (still using the slack distance)
                # TODO: better way to measure distance at the end?
                control[:3] = t_dir * (t_dist - self.slack[0])

        # angular control: orient toward the shoulder to within some error
        frame = pyutils.ShapeFrame()
        frame.setOrg(org=self.env.frameInterpolator[self.robot_ix]["target_pos"])
        frame.orientation = np.array(self.env.frameInterpolator[self.robot_ix]["target_frame"])
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

        control = np.clip(control, -self.env.robot_action_scale[6*self.robot_ix:6*self.robot_ix+6]* self.control_fraction, self.env.robot_action_scale[6*self.robot_ix:6*self.robot_ix+6]*self.control_fraction)

        #check control frame position against body distance and deflect if necessary

        #1: find the closest point on the body
        projected_frame_center = control[:3] + iiwa_frame_org
        self.dist_to_closest_point = -1
        for cap in self.env.skelCapsules:
            p0 = self.env.robot_skeleton.bodynodes[cap[0]].to_world(cap[2])
            p1 = self.env.robot_skeleton.bodynodes[cap[3]].to_world(cap[5])
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
            control[:3] = np.clip(control[:3], -self.env.robot_action_scale[:3] * self.control_fraction,self.env.robot_action_scale[6*self.robot_ix:6*self.robot_ix+3] * self.control_fraction)

        #if the two robots are too far from one another, move toward the other
        v_bt_frames = self.env.frameInterpolator[0]["target_pos"] - self.env.frameInterpolator[1]["target_pos"]
        dist_bt_frames = abs(np.linalg.norm(v_bt_frames))
        if dist_bt_frames > 0.48:
            if self.robot_ix == 0:
                v_bt_frames *= -1.0
            dir_to_other = v_bt_frames/dist_bt_frames
            control[:3] += dir_to_other
            control[:3] = np.clip(control[:3], -self.env.robot_action_scale[:3] * self.control_fraction,self.env.robot_action_scale[6*self.robot_ix:6*self.robot_ix+3] * self.control_fraction)
        # if the two robots are too close, move away
        elif dist_bt_frames < 0.2:
            if self.robot_ix == 1:
                v_bt_frames *= -1.0
            dir_from_other = v_bt_frames / dist_bt_frames
            control[:3] += dir_from_other
            control[:3] = np.clip(control[:3], -self.env.robot_action_scale[:3] * self.control_fraction,self.env.robot_action_scale[6*self.robot_ix:6*self.robot_ix+3] * self.control_fraction)


        return control

    def reset(self):
        self.time_held = 0
        self.proceeding = False

    def draw(self):
        worldTarget = self.env.robot_skeleton.bodynodes[self.target_node].to_world(self.nodeOffset)
        if self.proceeding:
            worldTarget = np.array(self.target_position)
        iiwa_frame_org = self.env.frameInterpolator["target_pos"]
        renderUtils.drawLines(lines=[[worldTarget, iiwa_frame_org]])
        renderUtils.drawSphere(pos=worldTarget)
        if self.dist_to_closest_point > 0:
            renderUtils.drawLines(lines=[[self.closest_point, iiwa_frame_org]])
            renderUtils.drawSphere(pos=self.closest_point)

class DressingTarget:
    #defines a dressing target task (e.g. "put limb L into feature F")
    def __init__(self, env, skel, feature, limb_sequence, distal_offset=None):
        self.env = env
        self.skel = skel
        self.feature = feature
        self.limb_sequence = limb_sequence
        self.distal_offset = distal_offset

    def getLimbProgress(self):
        return pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.skel, nodes=self.limb_sequence, offset=self.distal_offset), feature=self.feature)

class ContinuousCapacitiveSensor:
    def __init__(self, env, bodynode=None, offset=np.zeros(3), R=np.identity(3)):
        self.env = env
        self.bodynode = bodynode
        self.offset = offset
        self.R = R
        self.frame = pyutils.ShapeFrame()
        self.sensorOffsets = []
        self.sensorGlobals = []
        self.sensorReadings = []
        self.sensorRanges = []
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
        self.sensorGlobals = []
        for off in self.sensorOffsets:
            pos = self.frame.toGlobal(p=off)
            self.sensorGlobals.append(pos)

    def getReading(self):
        self.sensorReadings = []

        #first truncate all capsules to the frame half plane
        norm = self.frame.toGlobal(np.array([0,0,1])) - self.frame.toGlobal(np.zeros(3))
        norm /= np.linalg.norm(norm)
        frame_plane = Plane(org=self.frame.org, normal=norm)
        capsules = []
        #for c in self.env.skelCapsules:
        for capsule in self.env.skelCapsules:
            p0 = self.env.robot_skeleton.bodynodes[capsule[0]].to_world(capsule[2])
            p1 = self.env.robot_skeleton.bodynodes[capsule[3]].to_world(capsule[5])
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
        for ix,point in enumerate(self.sensorGlobals):
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

            if dist_to_closest_point < self.sensorRanges[ix] and dist_to_closest_point>0:
                self.sensorReadings.append([dist_to_closest_point, closest_point])
            else:
                self.sensorReadings.append([self.sensorRanges[ix], None])
        #print("finished getSensorReading")

    def getReadingSpherical(self):
        self.sensorReadings = []

        capsules = []
        for capsule in self.env.skelCapsules:
            p0 = self.env.robot_skeleton.bodynodes[capsule[0]].to_world(capsule[2])
            p1 = self.env.robot_skeleton.bodynodes[capsule[3]].to_world(capsule[5])
            r0 = capsule[1]
            r1 = capsule[4]
            capsules.append([p0, p1, r0, r1])

        #then for each sensor location, find the closest capsule projection
        for ix,point in enumerate(self.sensorGlobals):
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

            if dist_to_closest_point < self.sensorRanges[ix] and dist_to_closest_point>0:
                self.sensorReadings.append([dist_to_closest_point, closest_point])
            else:
                self.sensorReadings.append([self.sensorRanges[ix], None])

    def draw(self, ranges=True):
        norm = self.frame.toGlobal(np.array([0, 0, 1])) - self.frame.toGlobal(np.zeros(3))
        norm /= np.linalg.norm(norm)
        lines = [[self.frame.org, self.frame.org+norm]]
        for ix,p in enumerate(self.sensorGlobals):
            renderUtils.drawSphere(pos=p)
            #renderUtils.drawSphere(pos=p, rad=self.sensor_ranges[ix], solid=False)
            if self.sensorReadings[ix][0] < self.sensorRanges[ix]:
                lines.append([p,self.sensorReadings[ix][1]])
        renderUtils.drawLines(lines)

    def default2x3setup(self, spacing = 0.07):
        self.sensorOffsets = []
        self.sensorGlobals = []
        self.sensorReadings = []
        self.sensorRanges = []

        for ix in range(3):
            for iy in range(2):
                local = np.array([ix*spacing - spacing, iy*spacing - spacing/2.0, 0.065])
                self.sensorOffsets.append(local)
                self.sensorRanges.append(0.15)
                self.sensorReadings.append([0.15,None])
        self.update()
        self.numSensorRayZones = 6

    def getAggregateSensorReading(self):
        sensorReadings = np.zeros(0)
        for reading in self.sensorReadings:
            sensorReadings = np.concatenate([sensorReadings, np.array([reading[0]])])
        return sensorReadings


class DartClothUpperBodyDataDrivenClothIiwaGownMultibotEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):

        self.dualPolicy = True #if true, both human and robot share a (possibly split) policy
        self.isHuman = True #otherwise robot (also turns on some reward terms)
        rendering = True
        self.demoRendering = True #when true, reduce the debugging display significantly
        clothSimulation = True
        self.renderCloth = True
        self.renderSPDGhost = True
        dt = 0.0025
        cloth_dt = 0.005
        frameskip = 4
        self.numRobots = 2

        self.prefix = os.path.dirname(__file__)
        experiment_prefix = self.prefix+"/../../../../rllab/data/local/experiment/"
        #experiment_directory = "experiment_2019_02_14_coopt_multibot"
        experiment_directory = "experiment_2019_02_13_human_multibot"
        self.humanPolicyFile = experiment_prefix + experiment_directory + "/policy.pkl"
        self.robotPolicyFile = experiment_prefix + experiment_directory + "2/policy.pkl"
        self.otherPolicy = None #load this

        #human observation terms
        self.featureInObs   = False  # if true, feature centroid location and displacement from ef are observed
        self.oracleInObs    = True  # if true, oracle vector is in obs
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation
        self.robotJointObs  = True #if true, obs includes robot joint locations in world space
        #self.redundantRoboJoints = [4, 6, 10] #these will be excluded from obs
        self.redundantRoboJoints = [1] #these will be excluded from obs
        self.humanJointObs  = True #if true, obs includes human joint locations
        self.hoopNormalObs  = False #if true, obs includes the normal vector of the hoop
        self.jointLimVarObs = False #if true, constraints are varied in reset and given as NN input
        self.actionScaleVarObs = False #if true, action scales are varied in reset and given as NN input
        self.weaknessScaleVarObs = False #if true, scale torque limits on one whole side with a single value to model unilateral weakness
        self.elbowConVarObs = False  # if true, modify limits of the elbow joint
        self.SPDTargetObs = True  # need this to control this

        #reward flags (robot default)
        self.uprightReward              = False  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadReward           = False  # if True, rewarded for - head/torso angle
        self.elbowFlairReward           = False
        self.limbProgressReward         = True  # if true, the (-inf, 1] limb progress metric is included in reward
        self.oracleDisplacementReward   = False  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward           = False  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenalty         = True
        self.restPoseReward             = False
        self.variationEntropyReward     = False #if true (and variations exist) reward variation in action linearly w.r.t. distance in variation space (via sampling)
        self.shoulderPlaneReward        = False #if true, penalize robot for being "inside" the shoulder plan wrt human
        self.contactPenalty             = False #if true, penalize contact between robot and human

        if self.isHuman:
            self.uprightReward = True
            self.stableHeadReward = True
            self.contactGeoReward = True
            self.restPoseReward = False


        self.uprightRewardWeight              = 10  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadRewardWeight           = 2
        self.elbowFlairRewardWeight           = 1
        self.limbProgressRewardWeight         = 10  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementRewardWeight   = 50  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoRewardWeight           = 2  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenaltyWeight         = 5
        self.restPoseRewardWeight             = 1
        self.variationEntropyRewardWeight     = 1
        self.shoulderPlaneRewardWeight        = 3
        self.contactPenaltyWeight             = 1

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.collarTermination  = False  #if true, rollout terminates when collar is off the head/neck
        self.sleeveEndTerm      = False  #if true, terminate the rollout if the arm enters the end of sleeve feature before the beginning (backwards dressing)
        self.elbowFirstTerm     = False #if true, terminate when any limb enters the feature before the hand

        #other variables
        self.humanSPDController = None
        #self.humanSPDTarget = np.zeros(22)
        self.humanSPDIntperolationTarget = np.zeros(22) #interpolation b/t SPD target and pose
        self.humanSPDInterpolationRate = 10.5
        self.previousAction = np.zeros(22) #TODO: increase if recurrent
        self.prevTau = None
        self.elbowFlairNode = 10
        self.maxDeformation = 30.0
        self.restPose = None
        self.prevOracles = []
        for i in range(self.numRobots):
            self.prevOracles.append(np.zeros(3))
        self.prevAvgGeodesics = [None,None]
        self.localLeftEfShoulder1 = None
        self.localRightEfShoulder1 = None
        self.limbProgress = 0 #sum of all limb progress targets
        self.individualLimbProgress = [0.0,0.0] #two target features
        self.previousDeformationReward = 0
        self.handFirst = False #once the hand enters the feature, switches to true
        self.state_save_directory = "saved_control_states/"
        self.fingertip = np.array([0,-0.085,0])
        self.ef_accuracy_info = {'best':0, 'worst':0, 'total':0, 'average':0 }
        self.collisionResult = None
        self.humanRobotCollision = False #true if robot/human collision this step
        self.haptic_data = {'high':0, 'total':0, 'avg':0, 'var':0, 'instances':[]}
        self.hoop_FT_data = {'max':np.zeros(6), 'total':np.zeros(6), 'avg':np.zeros(6), 'instances':0}
        self.task_data = {'successes':0, 'trials':0, 'avg_limb_prog':0, 'total_limb_prog':0}
        self.initialIiwaEfs = []
        self.initialJointConstraints = None #set on init
        self.jointConstraintVariation = None #set in reset if "jointLimVarObs" is true. [0,1] symmetric scale of joint ranges
        self.initialActionScale = None #set after initialization
        self.weaknessScale = 1.0 #amount of gravity compenstation which is "taxed" from control torques
        self.variationTesting = False
        self.numSeeds = 10
        self.variations = [0.25, 0.5, 0.75, 1.0] #if variationTesting then cycle through these fixed variations
        self.variations = [0.1, 0.4, 0.7, 1.0] #if variationTesting then cycle through these fixed variations
        #self.variations = [0.4]
        self.simpleWeakness = True #if true, 10x torque limits, no gravity comp
        self.rigid_f_history = []
        self.rigid_f_history_avg = np.zeros(66)
        self.cloth_f_history = []
        self.cloth_f_history_avg = np.zeros(66)
        self.maxContactForce = 0  # mganitude of maximum contact force between robot and human this rollout
        self.stepMaxContactForce = 0  # mganitude of maximum contact force between robot and human this rollout

        self.redundantHumanJoints = [] #any joints which we don't want robot to observe

        # iiwa frame interpolator controls
        self.targetCentric = True #if true, robot policy operates on the target, not the current pose
        self.manualTargetControl = False #if true, actions are not considered
        self.frameInterpolator = []
        for r in range(self.numRobots):
            self.frameInterpolator.append({"active":True, "target_pos":np.zeros(3), "target_frame":np.identity(3), "speed":0.75, "aSpeed":5, "localOffset":np.array([0,0,0]), "eulers":np.zeros(3), "distanceLimit":0.15})
        self.scriptedFrameControl = False
        self.iiwaFrameControllers = [IiwaApproachHoverProceedAvoidController(self, 0, 8,np.array([0.21, 0.15, 0]),0.4),
                                     IiwaApproachHoverProceedAvoidController(self, 1, 3, np.array([-0.21, 0.15, 0]), 0.4)]


        self.consecutiveInstabilities = 0
        self.elbow_constraint_range = 0.3  # joint limit symmetrical distance from rest
        self.elbow_rest = 0.2  # drawn at reset
        self.elbow_initial_limits = [0, 0]  # set later in init

        # limb progress tracking
        self.limbProgressGraphing = False
        self.limbProgressGraph = None
        if(self.limbProgressGraphing):
            self.limbProgressGraph = pyutils.LineGrapher(title="Limb Progress")

        # deformation tracking
        self.deformationGraphing = False
        self.deformationGraph = None
        if (self.deformationGraphing):
            self.deformationGraph = pyutils.LineGrapher(title="Deformation")

        # force/torque tracking
        self.FTGraphing = False
        self.FTGraph = None
        if (self.FTGraphing):
            self.FTGraph = pyutils.LineGrapher(title="Force/Torque at EF")

        self.FTavgGraphing = False
        self.FTavgGraph = None
        if (self.FTavgGraphing):
            self.FTavgGraph = pyutils.LineGrapher(title="Force/Torque average at EF")

        # restPose error tracking
        self.restPoseErrorGraphing = False
        self.restPoseErrorGraph = None
        if (self.restPoseErrorGraphing):
            self.restPoseErrorGraph = pyutils.LineGrapher(title="Rest Pose Error")

        self.handleNodes = []
        self.updateHandleNodesFrom = 8  # left/right fingers

        self.human_actuatedDofs = np.arange(22)
        self.human_observation_size = len(self.human_actuatedDofs) * 3 #q(sin,cos), dq
        if self.prevTauObs:
            self.human_observation_size += len(self.human_actuatedDofs)
        if self.hapticsInObs:
            self.human_observation_size += 66
        if self.featureInObs:
            self.human_observation_size += 6 * 2 #2 features
        if self.oracleInObs:
            self.human_observation_size += 3 * 2 #2 features
        if self.contactIDInObs:
            self.human_observation_size += 22
        if self.robotJointObs:
            self.human_observation_size += 27*self.numRobots
        if self.humanJointObs:
            self.human_observation_size += 45
        if self.hoopNormalObs:
            self.human_observation_size += 3
        if self.actionScaleVarObs:
            self.human_observation_size += len(self.human_actuatedDofs)
        if self.jointLimVarObs:
            self.human_observation_size += len(self.human_actuatedDofs)
        if self.weaknessScaleVarObs:
            self.human_observation_size += 1
        if self.elbowConVarObs:
            self.human_observation_size += 1
        if self.SPDTargetObs:
            self.human_observation_size += 22

        # setup robot obs:
        self.robotCapacitiveObs = True  # need this flag for rendering and reading updates

        self.robot_observation_size = ((13 - 6) * 3)*self.numRobots #robot dofs
        self.robot_observation_size += 45 #human joint positions
        self.robot_observation_size += 27*self.numRobots #robot joint posistions
        self.robot_observation_size += 6 #human end effectors
        self.robot_observation_size += 6*self.numRobots #hoop joint resultant forces/torques
        self.robot_observation_size += 6*self.numRobots #end effector position and orientation
        if self.targetCentric:
            self.robot_observation_size += 6*self.numRobots #target frame position and orientation
        if self.robotCapacitiveObs:
            self.robot_observation_size += 6 * self.numRobots

        # initialize the Iiwa variables
        self.SPDController = None #resuse this for all robots
        self.iiwa_skels = []
        self.iiwa_ik_targets = [] #a set of shape frames for IK position and orientation
        self.maxIiwaReach = 1.0  # omni-directional reach (from 2nd dof)
        self.orientationBaseMatrix = np.array([[1, 0, 0],
                                               [0, 0, -1],
                                               [0, 1, 0]])
        self.robotEulerStates = []
        for i in range(self.numRobots):
            self.robotEulerStates.append(np.array([math.pi/2.0, -0.2, -0.15])) #XYZ used as euler angles to modify orientation base
        self.frameEulerStates = []
        for i in range(self.numRobots):
            self.frameEulerStates.append(np.array([math.pi/2.0, -0.2, -0.15])) #XYZ used as euler angles to modify orientation base
        self.orientationEulerStateLimits = [np.array([math.pi/2.0-1.5, -1.5, -1.5]), np.array([math.pi/2.0+1.5, 1.5, 1.5])]
        self.robotPathParams = {'p0_depth_range': 0.05, 'p0_depth_offset': 0.15,
                                'p0_disk_rad': self.maxIiwaReach * 0.7,
                                'p3_el_dim': np.array([0.2, 0.1, 0.1]), 'p3_el_org': np.array([0.15, 0.075, 0]),
                                'b_tan_dot_cone': 0.2, 'b_tan_len': 0.5,
                                'orient_dot_cone': 0.8}
        self.passiveIiwa = False
        self.root_adjustment = False
        self.ikOrientation = True
        self.adaptiveSPD = False
        self.freezeTracking = False #if true, target SPD pose is frozen
        self.previousIKResults = []

        #setup robot base location
        self.dFromRoot = 0.75
        #self.aFrom_invZ = 0.959931
        self.aFrom_invZ = 1.1
        self.iiwa_root_dofs = [] #values for the fixed 6 dof root transformation

        #robots mirrored around z
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot*math.sin(self.aFrom_invZ), -0.2, -self.dFromRoot*math.cos(self.aFrom_invZ)]))
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot*math.sin(-self.aFrom_invZ), -0.2, -self.dFromRoot*math.cos(-self.aFrom_invZ)]))


        #self.iiwa_root_dofs = np.array([-1.2, -1.2, -1.2, 0, -0.1, -0.8]) #values for the fixed 6 dof root transformation
        self.iiwa_rest = np.array([0, 0, 0, 0, 0, 0, 0])
        self.rigidClothFrame = pyutils.BoxFrame(c0=np.array([0.1,0.2,0.001]),c1=np.array([-0.1,0,-0.001]))
        self.rigidClothTargetFrame = pyutils.BoxFrame(c0=np.array([0.1,0.2,0.001]),c1=np.array([-0.1,0,-0.001]))
        self.targetFrameTracking = {"active":False, "history":[], "sampleRate":10}
        self.renderIKGhost = False
        self.renderIiwaReach = False
        self.renderIiwaCollidable = False
        self.renderHapticObs = False
        self.renderOracle = True
        self.print_skel_details = False
        self.posePath = pyutils.Spline()
        self.hoopTorques = None


        # SPD error graphing per dof
        self.graphSPDError = False
        self.SPDErrorGraph = None
        if self.graphSPDError:
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)

        #setup pybullet (use this one pybullet model for all robots...)
        if self.print_skel_details:
            print("Setting up pybullet")
        self.pyBulletPhysicsClient = p.connect(p.DIRECT)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pyBulletIiwa = p.loadURDF(dir_path + '/assets/iiwa_description/urdf/iiwa7_simplified.urdf')
        if self.print_skel_details:
            print("Iiwa bodyID: " + str(self.pyBulletIiwa))
            print("Number of pybullet joints: " + str(p.getNumJoints(self.pyBulletIiwa)))
            for i in range(p.getNumJoints(self.pyBulletIiwa)):
                jinfo = p.getJointInfo(self.pyBulletIiwa, i)
                print(" " + str(jinfo[0]) + " " + str(jinfo[1]) + " " + str(jinfo[2]) + " " + str(jinfo[3]) + " " + str(
                    jinfo[12]))

        screensize = (1280,720)
        if self.demoRendering:
            screensize = (720,720)

        #HERE IN INIT (replaces init in DartClothUpperBodyDataDrivenClothAssistBaseEnv)

        # rendering variables
        self.useOpenGL = rendering
        self.screenSize = screensize
        self.renderDARTWorld = True
        self.renderUI = True
        self.renderRewardsData = True
        self.gravity = True
        self.dataDrivenJointLimts = True

        # skeleton capsule definitions
        self.skelCapsulesDefined = False
        self.skelCapsules = []  # list of capsule instances with two body nodes, two offset vector and two radii

        #rewardsData setup
        self.rewardsData = renderUtils.RewardsData([], [], [], [])

        self.arm = 0  # 0->both, 1->right, 2->left
        self.human_actuatedDofs = np.arange(22)  # full upper body
        self.lockedDofs = []
        self.limbNodesR = [3, 4, 5, 6, 7]
        self.limbNodesL = [8, 9, 10, 11, 12]
        self.efOffset = np.array([0, -0.06, 0])

        if self.arm == 1:
            self.human_actuatedDofs = np.arange(3, 11) # right arm
            self.lockedDofs = np.concatenate([np.arange(3), np.arange(11, 22)])

        elif self.arm == 2:
            self.human_actuatedDofs = np.arange(11, 19) # left arm
            self.lockedDofs = np.concatenate([np.arange(11), np.arange(19, 22)])

        # 22 dof upper body (HUMAN)
        self.human_action_scale = np.ones(len(self.human_actuatedDofs))
        self.actionScaleVariation = np.ones(len(self.human_action_scale))
        self.human_action_scale *= 12
        if 0 in self.human_actuatedDofs:
            self.human_action_scale[self.human_actuatedDofs.tolist().index(0)] = 50
        if 1 in self.human_actuatedDofs:
            self.human_action_scale[self.human_actuatedDofs.tolist().index(1)] = 50

        self.human_control_bounds = np.array([np.ones(len(self.human_actuatedDofs)),
                                              np.ones(len(self.human_actuatedDofs)) * -1])

        #robot dofs and action/control bounds
        self.robot_actuatedDofs = np.arange(6)
        self.robot_action_scale = np.ones(6*2)
        self.robot_control_bounds = np.array(
            [np.ones(len(self.robot_action_scale)), np.ones(len(self.robot_action_scale)) * -1])

        self.reset_number = 0
        self.numSteps = 0

        # DartClothEnv setup
        skelFile = 'UpperBodyCapsules_datadriven.skel'
        obs_size = self.robot_observation_size
        act_bounds = self.robot_control_bounds
        if self.dualPolicy:
            #print("dual policy")
            # concatenate all obs_size and act_bounds
            obs_size = self.robot_observation_size + self.human_observation_size
            print("obs split: " + str(self.human_observation_size))
            act_bounds = np.array([np.ones(len(self.robot_action_scale) + len(self.human_actuatedDofs)),
                                   np.ones(len(self.robot_action_scale) + len(self.human_actuatedDofs)) * -1])
            #print(act_bounds)
        elif self.isHuman:
            obs_size = self.human_observation_size
            act_bounds = self.human_control_bounds


        #clothScene creation (and geodesic/separated mesh)
        clothMeshFile = "fullgown1.obj"
        clothMeshStateFile = None
        #clothMeshStateFile = "hanginggown.obj"
        clothScale = np.array([1.3, 1.3, 1.3])
        clothScene = None

        if clothMeshStateFile is not None:
            clothScene = pyphysx.ClothScene(step=cloth_dt,
                                            mesh_path=self.prefix + "/assets/" + clothMeshFile,
                                            state_path=self.prefix + "/../../../../" + clothMeshStateFile,
                                            scale=clothScale)
        else:
            clothScene = pyphysx.ClothScene(step=cloth_dt,
                                            mesh_path=self.prefix + "/assets/" + clothMeshFile,
                                            scale=clothScale)

        clothScene.togglePinned(0, 0)  # turn off auto-pin

        self.reward = 0
        self.cumulativeReward = 0
        self.deformation = 0

        # intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=frameskip, dt=dt,
                                  observation_size=obs_size, action_bounds=act_bounds,
                                  screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=frameskip, dt=dt,
                                  observation_size=obs_size, action_bounds=act_bounds,
                                  disableViewer=True, visualize=False)

        self.setSeed = -1  # default to no seed

        print("action_space: " + str(self.action_space))
        #print("action_scale: " + str(self.action_scale))
        #print("control_bounds: " + str(self.control_bounds))

        #dataDrivenJointLimts
        if self.dataDrivenJointLimts:
            leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_left'), self.robot_skeleton.joint('elbowjL'), True)
            rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_right'), self.robot_skeleton.joint('elbowjR'), False)
            leftarmConstraint.add_to_world(self.dart_world)
            rightarmConstraint.add_to_world(self.dart_world)

        utils.EzPickle.__init__(self)

        #gravity
        if not self.gravity:
            self.dart_world.set_gravity(np.zeros(3))
        else:
            self.dart_world.set_gravity(np.array([0., -9.8, 0]))

        self.clothScene.setFriction(0, 0.5)  # reset this anytime as desired

        self.collisionCapsuleInfo = None  # set in updateClothCollisionStructures(capsules=True)
        self.collisionSphereInfo = None  # set in updateClothCollisionStructures()
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)

        self.simulateCloth = clothSimulation

        # enable DART collision testing
        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        # setup collision filtering
        self.collision_filter = self.dart_world.create_collision_filter()
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                                self.robot_skeleton.bodynodes[12])  # left forearm to fingers
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[5],
                                                self.robot_skeleton.bodynodes[7])  # right forearm to fingers
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                                self.robot_skeleton.bodynodes[13])  # torso to neck
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                                self.robot_skeleton.bodynodes[14])  # torso to head
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                                self.robot_skeleton.bodynodes[3])  # torso to right shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                                self.robot_skeleton.bodynodes[8])  # torso to left shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                                self.robot_skeleton.bodynodes[3])  # neck to right shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                                self.robot_skeleton.bodynodes[8])  # neck to left shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                                self.robot_skeleton.bodynodes[3])  # head to right shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                                self.robot_skeleton.bodynodes[8])  # head to left shoulder
        self.collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[3],
                                                self.robot_skeleton.bodynodes[8])  # right shoulder to left shoulder

        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])

        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

        for i in range(len(self.dart_world.skeletons[0].dofs)):
            # print(self.dart_world.skeletons[0].dofs[i])
            self.dart_world.skeletons[0].dofs[i].set_damping_coefficient(1.0)

        # enable joint limits
        for i in range(len(self.robot_skeleton.joints)):
            print(self.robot_skeleton.joints[i])

        # DART does not automatically limit joints with any unlimited dofs
        self.robot_skeleton.joints[4].set_position_limit_enforced(True)
        self.robot_skeleton.joints[9].set_position_limit_enforced(True)

        self.clothScene.setSelfCollisionDistance(distance=0.03)
        self.clothScene.step()
        self.clothScene.reset()

        #LATER IN FILE:
        #initialize the Iiwa robot
        #print("loading URDFs")
        self.initialActionScale = np.array(self.human_action_scale)
        self.robot_action_scale = np.ones(6*self.numRobots)
        for rix in range(self.numRobots):
            self.robot_action_scale[6*rix:6*rix+3] = np.ones(3) * 0.01  # position
            self.robot_action_scale[6*rix+3:6*rix+6] = np.ones(3) * 0.02  # orientation
        self.iiwa_torque_limits = np.array([176,176,110,110,110,40,40])
        iiwaFilename = ""
        if self.renderIiwaCollidable:
            iiwaFilename = os.path.join(os.path.dirname(__file__), "assets", 'iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf')
        else:
            iiwaFilename = os.path.join(os.path.dirname(__file__), "assets", 'iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf')

        #add the robots to the scene
        self.dart_world.add_skeleton(filename=iiwaFilename)
        self.dart_world.add_skeleton(filename=iiwaFilename)
        #hoopFilename = os.path.join(os.path.dirname(__file__), "assets", 'sawyer_description/urdf/hoop_weldhang.urdf')
        #self.dart_world.add_skeleton(filename=hoopFilename)
        #self.hoop = self.dart_world.skeletons[3]
        #self.hoopToHandConstraint = None #set in reset on 1st reset
        if self.print_skel_details:
            for s in self.dart_world.skeletons:
                print(s)
        #self.iiwa_skel = self.dart_world.skeletons[2]
        for ix,s in enumerate(self.dart_world.skeletons):
            if ix > 1:
                self.iiwa_skels.append(s)
                self.iiwa_ik_targets.append(pyutils.ShapeFrame())
                self.previousIKResults.append(np.zeros(7))

        if self.print_skel_details:
            print(" Iiwa Skels setup and info: ")
        for rix,r in enumerate(self.iiwa_skels):
            if self.print_skel_details:
                print("Iiwa # " + str(rix))
                print(" Bodynodes:")
            for ix,bodynode in enumerate(r.bodynodes):
                if self.print_skel_details:
                    print("      "+str(ix)+" : " + bodynode.name)
                    print("         mass: " + str(bodynode.mass()))

                bodynode.set_gravity_mode(False)

            if self.print_skel_details:
                print(" Joints: ")
            for ix,joint in enumerate(r.joints):
                if self.print_skel_details:
                    print("     "+str(ix)+" : " + joint.name)
                joint.set_position_limit_enforced()

            if self.print_skel_details:
                print(" Dofs: ")
            for ix,dof in enumerate(r.dofs):
                if self.print_skel_details:
                    print("     "+str(ix)+" : " + dof.name)
                    print("         llim: " + str(dof.position_lower_limit()) + ", ulim: " + str(dof.position_upper_limit()))
                dof.set_damping_coefficient(2.0)

            r.joints[0].set_actuator_type(Joint.Joint.LOCKED)

            # enable DART collision testing
            r.set_self_collision_check(True)
            r.set_adjacent_body_check(False)

        #compute the joint ranges for null space IK
        self.iiwa_dof_llim = np.zeros(7)
        self.iiwa_dof_ulim = np.zeros(7)
        self.iiwa_dof_jr = np.zeros(7)
        for i in range(7):
            self.iiwa_dof_llim[i] = self.iiwa_skels[0].dofs[i + 6].position_lower_limit()
            self.iiwa_dof_ulim[i] = self.iiwa_skels[0].dofs[i + 6].position_upper_limit()
            self.iiwa_dof_jr[i] = self.iiwa_dof_ulim[i] - self.iiwa_dof_llim[i]

        # initialize the controller
        self.SPDController = SPDController(self, self.iiwa_skels[0], timestep=frameskip * dt)
        self.humanSPDController = SPDController(self, self.robot_skeleton, timestep=frameskip * dt, startDof=0, ckp=30.0, ckd=0.1)

        #tune the SPD gains
        for i in range(2):
            self.humanSPDController.Kp[i][i] = 10000
            self.humanSPDController.Kd[i][i] = 400

        self.humanSPDController.Kp[2][2] = 2000
        self.humanSPDController.Kd[2][2] = 70

        clav_dofs = [3,4,11,12]
        for i in clav_dofs:
            self.humanSPDController.Kp[i][i] = 2000
            self.humanSPDController.Kd[i][i] = 100

        shoulder_dofs = [5,6,7,13,14,15]
        for i in shoulder_dofs:
            self.humanSPDController.Kp[i][i] = 5000
            self.humanSPDController.Kd[i][i] = 100

        #elbows
        elbow_dofs = [8,16]
        for i in elbow_dofs:
            self.humanSPDController.Kp[i][i] = 4000
            self.humanSPDController.Kd[i][i] = 60

        wrist_dofs = [9,10,17,18]
        for i in wrist_dofs:
            self.humanSPDController.Kp[i][i] = 500
            self.humanSPDController.Kd[i][i] = 5

        #neck
        for i in range(19,21):
            self.humanSPDController.Kp[i][i] = 500
            self.humanSPDController.Kd[i][i] = 30

        self.capacitiveSensors = []
        self.capacitiveSensors.append(ContinuousCapacitiveSensor(env=self, bodynode=self.iiwa_skels[0].bodynodes[8]))
        self.capacitiveSensors.append(ContinuousCapacitiveSensor(env=self, bodynode=self.iiwa_skels[1].bodynodes[8]))
        for capsen in self.capacitiveSensors:
            capsen.default2x3setup()

        #disable character gravity
        if self.print_skel_details:
            print("!!Disabling character gravity (ie. auto gravity compensation")
        #if(not self.weaknessScaleVarObs):
        #    for ix, bodynode in enumerate(self.robot_skeleton.bodynodes):
        #        bodynode.set_gravity_mode(False)
        self.dart_world.skeletons[0].bodynodes[0].set_gravity_mode(False)

        #initialize initial joint and torque limits
        self.initialJointConstraints = [np.array(self.robot_skeleton.position_lower_limits()),np.array(self.robot_skeleton.position_upper_limits())]

        #clothing features
        #self.sleeveRVerts = [46, 697, 1196, 696, 830, 812, 811, 717, 716, 718, 968, 785, 1243, 783, 1308, 883, 990, 739, 740, 742, 1318, 902, 903, 919, 737, 1218, 736, 1217]
        self.sleeveRVerts = [532, 451, 251, 252, 253, 1334, 1320, 1184, 945, 985, 1062, 1607, 1037, 484, 1389, 679, 1230, 736, 1401, 1155, 486, 1410]
        self.sleeveLVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        #self.sleeveLMidVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        #self.sleeveLEndVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        #self.sleeveRMidVerts = [1054, 1055, 1057, 1058, 1060, 1061, 1063, 1052, 1051, 1049, 1048, 1046, 1045, 1043, 1042, 1040, 1039, 734, 732, 733]
        #self.sleeveREndVerts = [228, 1059, 229, 1062, 230, 1064, 227, 1053, 226, 1050, 225, 1047, 224, 1044, 223, 1041, 142, 735, 141, 1056]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        #self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)
        #self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)


        self.separatedMeshes = []
        self.dressingTargets = []
        self.boundaryAddedVerts = []
        self.boundarySkippedVerts = []
        # setup specific separated meshes and dressing targets for this task
        self.separatedMeshes.append(meshgraph.MeshGraph(clothscene=clothScene))  # left sleeve
        self.separatedMeshes.append(meshgraph.MeshGraph(clothscene=clothScene))  # right sleeve
        self.dressingTargets.append(DressingTarget(env=self, skel=self.robot_skeleton, feature=self.sleeveLSeamFeature, limb_sequence=self.limbNodesL, distal_offset=self.fingertip))
        self.dressingTargets.append(DressingTarget(env=self, skel=self.robot_skeleton, feature=self.sleeveRSeamFeature, limb_sequence=self.limbNodesR, distal_offset=self.fingertip))

        self.simulateCloth = clothSimulation
        #if self.simulateCloth: #these are relocated in reset
        self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))
        self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))

        if not self.renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        for i in range(len(self.robot_skeleton.dofs)):
            self.robot_skeleton.dofs[i].set_damping_coefficient(2.0)
        self.robot_skeleton.dofs[0].set_damping_coefficient(4.0)
        self.robot_skeleton.dofs[1].set_damping_coefficient(4.0)
        #self.robot_skeleton.dofs[19].set_damping_coefficient(10.0)
        #self.robot_skeleton.dofs[20].set_damping_coefficient(10.0)
        self.elbow_initial_limits = [self.robot_skeleton.dofs[16].position_lower_limit(), self.robot_skeleton.dofs[16].position_upper_limit()]

        # load rewards into the RewardsData structure
        if self.uprightReward:
            self.rewardsData.addReward(label="upright", rmin=-2.5, rmax=0, rval=0, rweight=self.uprightRewardWeight)

        if self.stableHeadReward:
            self.rewardsData.addReward(label="stable head",rmin=-1.2,rmax=0,rval=0, rweight=self.stableHeadRewardWeight)

        if self.elbowFlairReward:
            self.rewardsData.addReward(label="elbow flair", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.elbowFlairRewardWeight)

        if self.limbProgressReward:
            self.rewardsData.addReward(label="limb progress", rmin=-2.0, rmax=2.0, rval=0,
                                       rweight=self.limbProgressRewardWeight)

        if self.oracleDisplacementReward:
            self.rewardsData.addReward(label="oracle", rmin=-0.1, rmax=0.1, rval=0,
                                       rweight=self.oracleDisplacementRewardWeight)

        if self.contactGeoReward:
            self.rewardsData.addReward(label="contact geo", rmin=0, rmax=2.0, rval=0,
                                       rweight=self.contactGeoRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.deformationPenaltyWeight)

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0,
                                       rweight=self.restPoseRewardWeight)

        if self.variationEntropyReward:
            self.rewardsData.addReward(label="variation entropy", rmin=0, rmax=1.0, rval=0,
                                       rweight=self.variationEntropyRewardWeight)

        if self.shoulderPlaneReward:
            self.rewardsData.addReward(label="shoulder plane", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.shoulderPlaneRewardWeight)

        if self.contactPenalty:
            self.rewardsData.addReward(label="contact", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.contactPenaltyWeight)

        #self.loadCharacterState(filename="characterState_1starmin")

        if self.simpleWeakness:
            print("simple weakness active...")
            #self.initialActionScale *= 5
            #self.initialActionScale[2] += 65
            self.initialActionScale[2] += 13
            print("initialActionScale: " + str(self.initialActionScale))
        self.action_scale_range = [self.initialActionScale * 0.1, self.initialActionScale * 0.65]
        print(self.action_scale_range)

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #update features
        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.fitPlane()
        if self.sleeveRSeamFeature is not None:
            self.sleeveRSeamFeature.fitPlane()
        #if self.sleeveLEndFeature is not None:
        #    self.sleeveLEndFeature.fitPlane()
        #if self.sleeveLMidFeature is not None:
        #    self.sleeveLMidFeature.fitPlane()

        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # right fingertip in right shoulder local frame
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame

        self.human_action_scale = np.array(self.initialActionScale)
        #compute gravity compenstation and set action scale for the state
        if self.weaknessScaleVarObs:
            if self.simpleWeakness:
                for i in range(3,19):
                    #self.action_scale[i] = self.weaknessScale * self.initialActionScale[i]
                    self.human_action_scale[i] = LERP(self.action_scale_range[0][i], self.action_scale_range[1][i], self.weaknessScale)
            else:
                grav_comp = self.robot_skeleton.coriolis_and_gravity_forces()
                #self.additionalAction = np.array(grav_comp)
                self.supplementalTau = np.array(grav_comp)
                arm_tau = self.supplementalTau[11:19] #human's left arm
                #arm_tau = self.supplementalTau[3:11] #human's right arm
                #print("gravity comp(arm): " + str(arm_tau))
                #max_abs = max(arm_tau.max(), arm_tau.min(), key=abs)
                #print("     max: " + str(max_abs))
                for i in range(len(arm_tau)):
                    self.human_action_scale[i + 11] = self.weaknessScale * self.initialActionScale[i + 11] - abs((1.0 - self.weaknessScale) * arm_tau[i])
                    if(self.human_action_scale[i+11] < 0):
                        if(arm_tau[i] > 0):
                            arm_tau[i] += self.human_action_scale[i + 11]
                        else:
                            arm_tau[i] -= self.human_action_scale[i + 11]
                        self.human_action_scale[i + 11] = 0
                self.supplementalTau[11:19] = arm_tau
                #print(self.action_scale)

        if(self.freezeTracking):
            a=0
        else:
            #iiwa IK
            for iiwa_ix,iiwa in enumerate(self.iiwa_skels):
                tar_pos = np.array(self.iiwa_ik_targets[iiwa_ix].org)
                tar_quat = self.iiwa_ik_targets[iiwa_ix].quat
                tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)

                #IK with joint limits
                result = None
                self.setPosePyBullet(pose=iiwa.q[6:])
                if(self.ikOrientation):
                    result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                          endEffectorLinkIndex=8,
                                                          targetPosition=tar_pos - self.iiwa_root_dofs[iiwa_ix][3:],
                                                          targetOrientation=tar_quat,
                                                          #targetOrientation=tar_dir,
                                                          lowerLimits=self.iiwa_dof_llim.tolist(),
                                                          upperLimits=self.iiwa_dof_ulim.tolist(),
                                                          jointRanges=self.iiwa_dof_jr.tolist(),
                                                          restPoses=iiwa.q[6:].tolist(),
                                                          maxNumIterations=10
                                                          )
                else:
                    result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                          endEffectorLinkIndex=8,
                                                          targetPosition=tar_pos - self.iiwa_root_dofs[iiwa_ix][3:],
                                                          #targetOrientation=tar_quat,
                                                          #targetOrientation=tar_dir,
                                                          lowerLimits=self.iiwa_dof_llim.tolist(),
                                                          upperLimits=self.iiwa_dof_ulim.tolist(),
                                                          jointRanges=self.iiwa_dof_jr.tolist(),
                                                          restPoses=iiwa.q[6:].tolist(),
                                                          maxNumIterations=10
                                                          )
                #print("computed IK result: " + str(result))
                self.previousIKResults[iiwa_ix] = np.array(result)
                #self.setPosePyBullet(result)

    def _step(self, a):
        #print("a: " + str(a))
        startTime = time.time()
        if self.reset_number < 1 or not self.simulating:
            if self.dualPolicy:
                return np.zeros(self.human_observation_size+self.robot_observation_size), 0, False, {}
            elif self.isHuman:
                return np.zeros(self.human_observation_size), 0, False, {}
            else:
                return np.zeros(self.robot_observation_size), 0, False, {}

        startTime2 = time.time()

        human_a = None
        robot_a = None
        robo_action_scaled = None

        #query the "other" policy and set actions
        if self.dualPolicy:
            human_a = a[:len(self.human_action_scale)]
            robot_a = a[len(self.human_action_scale):]
        elif self.isHuman:
            human_a = a

            #query the robot policy
            if not self.scriptedFrameControl:
                try:
                    robot_obs = self._get_robot_obs()
                    robot_a, robot_a_info = self.otherPolicy.get_action(robot_obs)
                    robot_a = robot_a_info['mean']
                except:
                    print("robot policy not setup, defaulting zero action")
                    robot_a = np.zeros(len(self.robot_action_scale))

        else:
            robot_a = a

            #query the human policy
            try:
                human_obs = self._get_human_obs()
                human_a, human_a_info = self.otherPolicy.get_action(human_obs)
                human_a = human_a_info['mean']
            except:
                print("human policy not setup, defaulting zero action")
                human_a = np.zeros(len(self.human_action_scale))


        #override robot frame control
        if self.scriptedFrameControl:
            robo_action_scaled = np.zeros(0)
            for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                robo_action_scaled = np.concatenate([robo_action_scaled, self.iiwaFrameControllers[iiwa_ix].query()])
        else:
            robo_action_clamped = np.clip(robot_a, -np.ones(len(robot_a)), np.ones(len(robot_a)))
            robo_action_scaled = np.multiply(robo_action_clamped, self.robot_action_scale)


        #BEGIN - ROBOT FRAME UPDATE FROM ACTIONS
        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            efpos = iiwa.bodynodes[9].to_world(np.zeros(3))
            efdir = iiwa.bodynodes[9].to_world(np.array([0, 1.0, 0])) - efpos
            efdir = efdir / np.linalg.norm(efdir)

            if(self.targetCentric):
                efpos = np.array(self.iiwa_ik_targets[iiwa_ix])
                efdir = self.iiwa_ik_targets[iiwa_ix].toGlobalVec(np.array([0,1.0,0]))
                efdir = efdir / np.linalg.norm(efdir)

            # use actions
            if (not self.manualTargetControl):
                robo_action_slice = robo_action_scaled[6*iiwa_ix:6*iiwa_ix+6]
                self.frameInterpolator[iiwa_ix]["target_pos"] += robo_action_slice[:3]
                self.frameInterpolator[iiwa_ix]["eulers"] += robo_action_slice[3:6]

            toRoboEF = iiwa.bodynodes[9].to_world(np.zeros(3)) - self.frameInterpolator[iiwa_ix]["target_pos"]
            distToRoboEF = np.linalg.norm(toRoboEF)
            # clamp interpolation target frame to distance from EF to prevent over commit
            if (distToRoboEF > (self.frameInterpolator[iiwa_ix]["distanceLimit"])):
                # print("clamping frame")
                self.frameInterpolator[iiwa_ix]["target_pos"] = iiwa.bodynodes[9].to_world(np.zeros(3)) + -(toRoboEF / distToRoboEF) * self.frameInterpolator[iiwa_ix]["distanceLimit"]

            #interpolate the target frame at constant speed toward a goal location
            if self.frameInterpolator[iiwa_ix]["active"]:
                targetFrame = pyutils.ShapeFrame()
                targetFrame.orientation = np.array(self.frameInterpolator[iiwa_ix]["target_frame"])
                self.frameInterpolator[iiwa_ix]["eulers"] = np.clip(self.frameInterpolator[iiwa_ix]["eulers"], self.orientationEulerStateLimits[0],
                                               self.orientationEulerStateLimits[1])

                interpFrameEulerState = np.array(self.frameInterpolator[iiwa_ix]["eulers"])

                # interpolate orientation:
                if True:
                    self.frameInterpolator[iiwa_ix]["target_frame"] = pyutils.euler_to_matrix(interpFrameEulerState)
                    targetFrame.orientation = np.array(self.frameInterpolator[iiwa_ix]["target_frame"])



                targetFrame.org = np.array(self.frameInterpolator[iiwa_ix]["target_pos"])
                targetFrame.updateQuaternion()
                globalOffset = targetFrame.toGlobal(p=self.frameInterpolator[iiwa_ix]["localOffset"])
                targetFrame.org += targetFrame.org-globalOffset
                targetDisp = targetFrame.org - self.iiwa_ik_targets[iiwa_ix].org
                dispMag = np.linalg.norm(targetDisp)
                travelMag = min(self.dt*self.frameInterpolator[iiwa_ix]["speed"], dispMag)
                if(travelMag > 0.001):
                    self.iiwa_ik_targets[iiwa_ix].org += (targetDisp/dispMag)*travelMag

                #interpolate the target eulers:
                hn = iiwa.bodynodes[8]
                roboFrame = pyutils.ShapeFrame()
                roboFrame.setTransform(hn.T)
                robo_eulers = pyutils.getEulerAngles3(roboFrame.orientation)
                for i in range(3):
                    eDist = interpFrameEulerState[i] - robo_eulers[i]
                    eTraverse = min(self.dt*self.frameInterpolator[iiwa_ix]["aSpeed"], abs(eDist))
                    if(eDist != 0):
                        self.frameEulerStates[iiwa_ix][i] = robo_eulers[i] + (eDist/abs(eDist))*eTraverse
                        #print("eTraverse " + str(i) + ": " + str((eDist/abs(eDist))*eTraverse))



            self.frameEulerStates[iiwa_ix] = np.clip(self.frameEulerStates[iiwa_ix], self.orientationEulerStateLimits[0], self.orientationEulerStateLimits[1])


            try:
                self.iiwa_ik_targets[iiwa_ix].orientation = pyutils.euler_to_matrix(self.frameEulerStates[iiwa_ix])
                self.iiwa_ik_targets[iiwa_ix].updateQuaternion()

            except:
                print("INVALID TARGET SETTING STATE for IIWA " + str(iiwa_ix))
                #print(" targetDir: " + str(targetDir))
                print(" self.ikTarget: " + str(self.iiwa_ik_targets[iiwa_ix].org))
                return np.zeros(self.obs_size), -10000, True, {}

        # END - ROBOT FRAME UPDATE FROM ACTIONS


        self.updateBeforeSimulation()  # any env specific updates before simulation
        # print("updateBeforeSimulation took " + str(time.time() - startTime2))
        try:
            self.avgtimings["updateBeforeSimulation"] += time.time() - startTime2
        except:
            self.avgtimings["updateBeforeSimulation"] = time.time() - startTime2

        human_control = np.zeros(self.robot_skeleton.ndofs)

        # SPD addition for human
        if (self.humanSPDController is not None):
            # print(self.humanSPDController.target)
            pos_upper_lim = self.robot_skeleton.position_upper_limits()
            pos_lower_lim = self.robot_skeleton.position_lower_limits()

            self.humanSPDIntperolationTarget += human_a*0.1
            #clamp humanSPD target

            #q = np.array(self.robot_skeleton.q)
            maxDeviation = 0.55
            for d in range(self.robot_skeleton.ndofs):
                #limit close to current pose
                qdof = self.robot_skeleton.q[d]
                diff = qdof - self.humanSPDIntperolationTarget[d]
                if abs(diff) > maxDeviation:
                    if diff > 0:
                        self.humanSPDIntperolationTarget[d] = qdof - maxDeviation
                    else:
                        self.humanSPDIntperolationTarget[d] = qdof + maxDeviation

                #joint limiting
                self.humanSPDIntperolationTarget[d] = min(self.humanSPDIntperolationTarget[d], pos_upper_lim[d])
                self.humanSPDIntperolationTarget[d] = max(self.humanSPDIntperolationTarget[d], pos_lower_lim[d])

            target_diff = self.humanSPDIntperolationTarget - self.humanSPDController.target
            limit_diff = pos_upper_lim - pos_lower_lim
            # print(target_diff)
            maxChange = self.humanSPDInterpolationRate * self.dt
            # self.humanSPDController.target

            for d in range(len(target_diff)):
                # self.humanSPDController.target[d] = min(maxChange, abs(target_diff[d])) * (target_diff[d]/abs(target_diff[d]))
                if math.isinf(limit_diff[d]):  # shoulders
                    limit_diff[d] = 1.5
                if abs(target_diff[d]) > maxChange * limit_diff[d]:
                    self.humanSPDController.target[d] += (target_diff[d] / abs(target_diff[d])) * maxChange
                else:
                    self.humanSPDController.target[d] = self.humanSPDIntperolationTarget[d]

            human_control = self.humanSPDController.query(None)

        human_clamped_control = np.array(human_control)

        for i in range(len(human_clamped_control)):
            if human_clamped_control[i] > self.human_action_scale[i]:
                human_clamped_control[i] = self.human_action_scale[i]
            if human_clamped_control[i] < -self.human_action_scale[i]:
                human_clamped_control[i] = -self.human_action_scale[i]

        human_tau = np.array(human_clamped_control)

        startTime2 = time.time()

        self.do_simulation(human_tau, self.frame_skip)

        try:
            self.avgtimings["do_simulation"] += time.time() - startTime2
        except:
            self.avgtimings["do_simulation"] = time.time() - startTime2

        #set position and 0 velocity of locked dofs
        qpos = np.array(self.robot_skeleton.q)
        qvel = np.array(self.robot_skeleton.dq)
        for dof in self.lockedDofs:
            qpos[dof] = 0
            qvel[dof] = 0
        self.set_state(qpos, qvel)

        #update capacitive sensor
        if self.robotCapacitiveObs:
            try:
                for capsens in self.capacitiveSensors:
                    capsens.update()
                    capsens.getReadingSpherical()
            except:
                print("Capacitive sensors not setup")

        startTime2 = time.time()
        reward = self.computeReward(tau=human_tau)
        #print("computeReward took " + str(time.time() - startTime2))
        try:
            self.avgtimings["computeReward"] += time.time() - startTime2
        except:
            self.avgtimings["computeReward"] = time.time() - startTime2


        startTime2 = time.time()
        ob = self._get_obs()
        s = self.state_vector()
        #print("obs and state took " + str(time.time() - startTime2))
        try:
            self.avgtimings["obs"] += time.time() - startTime2
        except:
            self.avgtimings["obs"] = time.time() - startTime2


        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)

        done, terminationReward = self.checkTermination(human_tau, s, ob)
        reward += terminationReward
        self.reward = reward
        self.cumulativeReward += self.reward
        self.rewardTrajectory.append(self.reward)

        #if done and terminationReward < 0:
        #    print("terminated negatively. reward trajectory: " + str(self.rewardTrajectory))

        self.numSteps += 1
        #print("_step took " + str(time.time() - startTime))
        try:
            self.avgtimings["_step"] += time.time() - startTime2
        except:
            self.avgtimings["_step"] = time.time() - startTime2
        return ob, self.reward, done, {}
        #except:
        #    print("step " + str(self.numSteps) + " failed")
            #self.step(action=np.zeros(len(a)))

    def do_simulation(self, tau, n_frames):
        human_tau = np.array(tau)

        'Override of DartClothEnv.do_simulation to add cloth simulation stepping in a more intelligent manner without compromising upper body'
        if not self.simulating:
            return

        clothSteps = (n_frames*self.dart_world.time_step()) / self.clothScene.timestep
        #print("cloth steps: " + str(clothSteps))
        #print("n_frames: " + str(n_frames))
        #print("dt: " + str(self.dart_world.time_step()))
        clothStepRatio = self.dart_world.time_step()/self.clothScene.timestep
        clothStepsTaken = 0
        human_pre_q = np.array(self.robot_skeleton.q)
        human_pre_dq = np.array(self.robot_skeleton.dq)

        robot_pre_qs = []
        robot_pre_dqs = []
        for iiwa_ix,iiwa in enumerate(self.iiwa_skels):
            robot_pre_qs.append(np.array(iiwa.q))
            robot_pre_dqs.append(np.array(iiwa.dq))

        self.humanRobotCollision = False
        for i in range(n_frames):
            #print("step " + str(i))
            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)

            if not self.kinematic:
                self.robot_skeleton.set_forces(human_tau)

                if self.passiveIiwa:
                    #zero torque for passive robots
                    a = 0
                    tau = np.zeros(len(self.iiwa_skels[0].q))
                    for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                        iiwa.set_forces(tau)
                else:
                    #iiwa SPD w/ torque limiting
                    for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                        if self.SPDController is not None:
                            self.SPDController.target = np.array(self.previousIKResults[iiwa_ix])
                            tau = np.clip(self.SPDController.query(obs=None, skel=iiwa), -self.iiwa_torque_limits, self.iiwa_torque_limits)
                            tau = np.concatenate([np.zeros(6), tau])
                            iiwa.set_forces(tau)

                #iiwa_control = self.SPDController.query(obs=None)
                #iiwa_control = np.clip(iiwa_control, -self.iiwa_torque_limits, self.iiwa_torque_limits)
                #self.iiwa_skel.set_forces(np.concatenate([np.zeros(6), iiwa_control]))

                self.dart_world.step()
                self.instabilityDetected = self.checkInvalidDynamics()
                if self.instabilityDetected:
                    self.consecutiveInstabilities += 1
                    print("Invalid dynamics detected at step " + str(i)+"/"+str(n_frames))
                    self.set_state(human_pre_q, human_pre_dq)
                    for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                        iiwa.set_positions(robot_pre_qs[iiwa_ix])
                        iiwa.set_velocities(robot_pre_dqs[iiwa_ix])
                    return

                #new handle updates testing
                if(i%2 == 1 and self.simulateCloth):#every other step
                    #print(i)

                    for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                        hn = iiwa.bodynodes[8]  # end effector (hand) node
                        self.handleNodes[iiwa_ix].updatePrevConstraintPositions()
                        newOrg = hn.to_world(np.array([0, 0, 0.05]))
                        if not np.isfinite(newOrg).all():
                            print("Invalid robot pose in handle update...")
                            for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                                iiwa.set_positions(robot_pre_qs[iiwa_ix])
                                iiwa.set_velocities(robot_pre_dqs[iiwa_ix])
                            return
                        self.handleNodes[iiwa_ix].org = np.array(newOrg)
                        self.handleNodes[iiwa_ix].setOrientation(R=hn.T[:3, :3])


                    self.updateClothCollisionStructures(hapticSensors=True)
                    self.clothScene.step()
                    self.updateHandleContactForceTorques()
                    for h in self.handleNodes:
                        h.step()
                    if (i == 1):
                        for h in self.handleNodes:
                            h.prev_avg_force = np.array(h.prev_force)
                            h.prev_avg_torque = np.array(h.prev_torque)
                        if len(self.rigid_f_history) > 5:
                            del self.rigid_f_history[0]
                        if len(self.cloth_f_history) > 5:
                            del self.cloth_f_history[0]
                        self.rigid_f_history.append(self.getCumulativeHapticForcesFromRigidContacts())
                        self.cloth_f_history.append(self.clothScene.getHapticSensorObs())
                    else:
                        for h in self.handleNodes:
                            h.prev_avg_force += np.array(h.prev_force)
                            h.prev_avg_torque += np.array(h.prev_torque)
                        self.rigid_f_history[-1] += self.getCumulativeHapticForcesFromRigidContacts()
                        self.cloth_f_history[-1] += self.clothScene.getHapticSensorObs()

                    if (i == (n_frames - 1)):
                        for h in self.handleNodes:
                            h.prev_avg_force /= n_frames
                            h.prev_avg_torque /= n_frames
                        self.rigid_f_history[-1] /= (n_frames / 2.0)
                        self.cloth_f_history[-1] /= (n_frames / 2.0)
                        self.rigid_f_history_avg = np.array(self.rigid_f_history[0])
                        self.cloth_f_history_avg = np.array(self.cloth_f_history[0])
                        for fix in range(1, len(self.rigid_f_history)):
                            self.rigid_f_history_avg += self.rigid_f_history[fix]
                        for fix in range(1, len(self.cloth_f_history)):
                            self.cloth_f_history_avg += self.cloth_f_history[fix]
                        self.rigid_f_history_avg /= len(self.rigid_f_history)
                        self.cloth_f_history_avg /= len(self.cloth_f_history)
                    clothStepsTaken += 1

                    if self.FTGraphing and self.reset_number > 0: #TODO: update this for multiple or specific handle nodes
                        print("S:" + str(self.numSteps) + "|I:" + str(i))
                        print(self.handleNodes[0].prev_force[1])
                        self.FTGraph.yData[-1][self.numSteps * 5 + i] = self.handleNodes[0].prev_force[0]
                        self.FTGraph.yData[-2][self.numSteps * 5 + i] = self.handleNodes[0].prev_force[1]
                        self.FTGraph.yData[-3][self.numSteps * 5 + i] = self.handleNodes[0].prev_force[2]
                        self.FTGraph.yData[-4][self.numSteps * 5 + i] = self.handleNodes[0].prev_torque[0]
                        self.FTGraph.yData[-5][self.numSteps * 5 + i] = self.handleNodes[0].prev_torque[1]
                        self.FTGraph.yData[-6][self.numSteps * 5 + i] = self.handleNodes[0].prev_torque[2]
                        # print(self.numSteps)
                        # print(i)
                        # print(n_frames)

                        if self.numSteps % 5 == 0 and i == (n_frames - 1):
                            # print("here")
                            self.FTGraph.update()

                    if self.FTavgGraphing and self.reset_number > 0 and i == (n_frames - 1):
                        self.FTavgGraph.yData[-1][self.numSteps] = self.handleNodes[0].prev_avg_force[0]
                        self.FTavgGraph.yData[-2][self.numSteps] = self.handleNodes[0].prev_avg_force[1]
                        self.FTavgGraph.yData[-3][self.numSteps] = self.handleNodes[0].prev_avg_force[2]
                        self.FTavgGraph.yData[-4][self.numSteps] = self.handleNodes[0].prev_avg_torque[0]
                        self.FTavgGraph.yData[-5][self.numSteps] = self.handleNodes[0].prev_avg_torque[1]
                        self.FTavgGraph.yData[-6][self.numSteps] = self.handleNodes[0].prev_avg_torque[2]

                        if self.numSteps % 5 == 0:
                            self.FTavgGraph.update()

        self.consecutiveInstabilities = 0

    def computeReward(self, tau):

        # compute and return reward at the current state
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        localLeftEfShoulder2 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2)  # right fingertip in right shoulder local frame
        localRightEfShoulder2 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2)  # right fingertip in right shoulder local frame

        self.prevTau = tau
        reward_record = []

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))
            reward_record.append(reward_upright)

        reward_stableHead = 0
        if self.stableHeadReward:
            reward_stableHead = max(-1.2, -abs(self.robot_skeleton.q[19]) - abs(self.robot_skeleton.q[20]))
            reward_record.append(reward_stableHead)

        reward_elbow_flair = 0
        if self.elbowFlairReward:
            root = self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
            spine = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
            elbow = self.robot_skeleton.bodynodes[self.elbowFlairNode].to_world(np.zeros(3))
            dist = pyutils.distToLine(p=elbow, l0=root, l1=spine)
            z = 0.5
            s = 16
            l = 0.2
            reward_elbow_flair = -(1 - (z * math.tanh(s * (dist - l)) + z))
            # print("reward_elbow_flair: " + str(reward_elbow_flair))
            reward_record.append(reward_elbow_flair)

        reward_limbprogress = 0
        if self.limbProgressReward:
            if self.simulateCloth:
                self.individualLimbProgress[0] = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                                                      offset=self.fingertip), feature=self.sleeveLSeamFeature)
                self.individualLimbProgress[1] = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR,
                                                      offset=self.fingertip), feature=self.sleeveRSeamFeature)

            for pix,prog in enumerate(self.individualLimbProgress):
                if (math.isnan(prog)):  # catch nan before it gets into the reward computation
                    print("!!! NaN limb progress detected !!!")
                    self.individualLimbProgress[pix] = -2.0
                reward_limbprogress += self.individualLimbProgress[pix]

            self.limbProgress = reward_limbprogress
            reward_record.append(reward_limbprogress)

        #2 geodesic fields...
        avgContactGeodesic = None
        if self.numSteps > 0 and self.simulateCloth:
            sm_ix = 0 #separated mesh index (Left seam feature)
            contactInfo = pyutils.getContactIXGeoSide(sensorix=21, clothscene=self.clothScene,
                                                      meshgraph=self.separatedMeshes[sm_ix])
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)
            self.prevAvgGeodesics[0] = avgContactGeodesic

            sm_ix = 1 #separated mesh index (Left seam feature)
            contactInfo = pyutils.getContactIXGeoSide(sensorix=12, clothscene=self.clothScene,
                                                      meshgraph=self.separatedMeshes[sm_ix])
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)
            self.prevAvgGeodesics[1] = avgContactGeodesic

        reward_oracleDisplacement = 0
        if self.oracleDisplacementReward:
            if np.linalg.norm(self.prevOracles[0]) > 0 and self.localLeftEfShoulder1 is not None:
                relative_displacement = localLeftEfShoulder2 - self.localLeftEfShoulder1
                oracle0 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2 + self.prevOracles[0]) - localLeftEfShoulder2
                reward_oracleDisplacement += relative_displacement.dot(oracle0)
            if np.linalg.norm(self.prevOracles[1]) > 0 and self.localRightEfShoulder1 is not None:
                relative_displacement = localRightEfShoulder2 - self.localRightEfShoulder1
                oracle0 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2 + self.prevOracles[1]) - localRightEfShoulder2
                reward_oracleDisplacement += relative_displacement.dot(oracle0)
            reward_record.append(reward_oracleDisplacement)

        reward_contactGeo = 0
        if self.contactGeoReward:
            if self.simulateCloth:
                if self.individualLimbProgress[0] > 0:
                    reward_contactGeo += 1.0
                elif self.prevAvgGeodesics[0] is not None:
                    reward_contactGeo += 1.0 - (self.prevAvgGeodesics[0] / self.separatedMeshes[0].maxGeo)
                if self.individualLimbProgress[1] > 0:
                    reward_contactGeo += 1.0
                elif self.prevAvgGeodesics[1] is not None:
                    reward_contactGeo += 1.0 - (self.prevAvgGeodesics[1] / self.separatedMeshes[1].maxGeo)

            reward_record.append(reward_contactGeo)

        clothDeformation = 0
        if self.simulateCloth:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(
                0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation
        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        reward_restPose = 0
        restPoseError = 0
        if self.restPoseReward:
            if self.restPose is not None:
                # z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
                # s = 1.0  # steepness (higher is steeper)
                # l = 4.2  # translation
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                restPoseError = dist
                # reward_restPose = -(z * math.tanh(s * (dist - l)) + z)
                reward_restPose = max(-51, -dist)
            # print("distance: " + str(dist) + " -> " + str(reward_restPose))
            reward_record.append(reward_restPose)

        # TODO
        if self.variationEntropyReward:
            a = 0
            reward_record.append(0)

        #TODO: may need to update this...
        reward_shoulderPlane = 0
        if self.shoulderPlaneReward:
            print("shoulderPlaneReward: NOT UPDATED FOR MULTI-BOT")
            efpos = self.iiwa_skels[0].bodynodes[9].to_world(np.zeros(3))
            # construct shoulder plane
            s0 = self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))
            s1 = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
            snorm = s1 - s0
            _snorm = snorm / np.linalg.norm(snorm)
            sb1 = np.cross(_snorm, np.array([0, 1.0, 0]))
            sb2 = np.cross(_snorm, sb1)
            shoulderPlane = Plane(org=s1, normal=_snorm, b1=sb1, b2=sb2)
            proj_ef = shoulderPlane.projectPoint(p=efpos)
            if (proj_ef - efpos).dot(_snorm) > 0:
                reward_shoulderPlane = -min(1.0, np.linalg.norm(proj_ef - efpos))
            else:
                reward_shoulderPlane = 0
            reward_record.append(reward_shoulderPlane)

        reward_contactPenalty = 0
        if self.contactPenalty:
            if self.humanRobotCollision:
                reward_contactPenalty = -1.0
            reward_record.append(reward_contactPenalty)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        # update graphs
        if self.limbProgressGraphing and self.reset_number > 0:
            self.limbProgressGraph.yData[-1][self.numSteps] = self.limbProgress
            if self.numSteps % 5 == 0:
                self.limbProgressGraph.update()

        if self.deformationGraphing and self.reset_number > 0:
            self.deformationGraph.yData[-1][self.numSteps] = self.deformation
            if self.numSteps % 5 == 0:
                self.deformationGraph.update()

        # update graphs
        if self.restPoseErrorGraphing and self.reset_number > 0:
            self.restPoseErrorGraph.yData[self.reset_number - 1][self.numSteps] = restPoseError
            if self.numSteps % 5 == 0:
                self.restPoseErrorGraph.update()

        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight \
                      + reward_stableHead * self.stableHeadRewardWeight \
                      + reward_limbprogress * self.limbProgressRewardWeight \
                      + reward_contactGeo * self.contactGeoRewardWeight \
                      + reward_clothdeformation * self.deformationPenaltyWeight \
                      + reward_oracleDisplacement * self.oracleDisplacementRewardWeight \
                      + reward_elbow_flair * self.elbowFlairRewardWeight \
                      + reward_restPose * self.restPoseRewardWeight \
                      + reward_shoulderPlane * self.shoulderPlaneRewardWeight \
                      + reward_contactPenalty * self.contactPenaltyWeight
        if (not math.isfinite(self.reward)):
            print("Not finite reward...")
            return -500
        return self.reward

    def checkTermination(self, tau, s, obs):
        '''
        #record haptic info
        haptic_forces = self.getCumulativeHapticForcesFromRigidContacts()
        num_new_entries = 0
        for i in range(self.clothScene.getNumHapticSensors()):
            f = haptic_forces[i * 3:i * 3 + 3]
            f_mag = np.linalg.norm(f)
            if(f_mag > 0.001):
                num_new_entries += 1
                self.haptic_data['instances'].append(f_mag)
                self.haptic_data['total'] += f_mag
                if(f_mag > self.haptic_data['high']):
                    self.haptic_data['high'] = f_mag
        if(num_new_entries > 0):
            self.haptic_data['avg'] = self.haptic_data['total'] / len(self.haptic_data['instances'])
            self.haptic_data['var'] = 0
            for i in self.haptic_data['instances']:#compute variance
                dif = i-self.haptic_data['avg']
                self.haptic_data['var'] += dif*dif
            self.haptic_data['var'] /= len(self.haptic_data['instances'])
            print("Haptic_data: high:" + str(self.haptic_data['high']) + " | avg: " + str(self.haptic_data['avg']) + " | var: " + str(self.haptic_data['var']) + " | # samples: " + str(len(self.haptic_data['instances'])))
        '''

        if self.consecutiveInstabilities > 5:
            print("too many consecutive instabilities: " + str(self.consecutiveInstabilities) + "/5")
            return True, -10000

        # check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -10000
        elif not np.isfinite(s).all():
            print("Infinite value detected in s..." + str(s))
            return True, -10000
        else:
            for iiwa_ix,iiwa in enumerate(self.iiwa_skels):
                if not np.isfinite(iiwa.q).all():
                    print("Infinite value detected in iiwa "+str(iiwa_ix)+" state..." + str(s))
                    return True, -10000
        # elif self.sleeveEndTerm and self.limbProgress <= 0 and self.simulateCloth:
        #    limbInsertionError = pyutils.limbFeatureProgress(
        #        limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
        #                                          offset=np.array([0, -0.095, 0])), feature=self.sleeveLEndFeature)
        #    if limbInsertionError > 0:
        #        return True, -500
        # elif self.elbowFirstTerm and self.simulateCloth and not self.handFirst:
        #    if self.limbProgress > 0 and self.limbProgress < 0.14:
        #        self.handFirst = True
        #    else:
        #        limbInsertionError = pyutils.limbFeatureProgress(
        #            limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL[:3]),
        #            feature=self.sleeveLSeamFeature)
        #        if limbInsertionError > 0:
        #            return True, -500

        #pose_error = self.iiwa_skel.q[6:] - self.previousIKResult
        #if self.graphSPDError:
        #    self.SPDErrorGraph.addToLinePlot(data=pose_error.tolist())

        return False, 0

    def _get_human_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.human_actuatedDofs))
        dtheta = np.zeros(len(self.human_actuatedDofs))
        for ix, dof in enumerate(self.human_actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.prevTauObs:
            obs = np.concatenate([obs, self.prevTau])

        if self.hapticsInObs:
            f = None
            # robot and human contact
            #rigid_f = self.getCumulativeHapticForcesFromRigidContacts()
            rigid_f = np.array(self.rigid_f_history_avg)
            if self.simulateCloth and self.hapticsAware:
                #f = self.clothScene.getHapticSensorObs()#get force from simulation
                f = np.array(self.cloth_f_history_avg)
            else:
                f = np.zeros(f_size)
            f += rigid_f
            obs = np.concatenate([obs, f]).ravel()

        if self.featureInObs:
            if self.simulateCloth:
                centroid = np.array(self.sleeveLSeamFeature.plane.org)
                efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                disp = centroid-efL
                obs = np.concatenate([obs, centroid, disp]).ravel()

                centroid = np.array(self.sleeveRSeamFeature.plane.org)
                efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
                disp = centroid-efR
                obs = np.concatenate([obs, centroid, disp]).ravel()

        if self.oracleInObs and self.simulateCloth:
            oracle = np.zeros(3)
            sm_ix = 0 #separated mesh index (Left feature)
            if self.reset_number == 0:
                pass
            elif self.individualLimbProgress[0] > 0:
                oracle = self.sleeveLSeamFeature.plane.normal
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene, meshgraph=self.separatedMeshes[sm_ix],returnOnlyGeo=False)

                if minGeoVix is not None:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMeshes[sm_ix].geoVectorAt(minGeoVix, side=vixSide)

                    if minContactGeodesic == 0:
                        minGeoVix = None
                else:
                    #oracle points to the garment when ef not in contact
                    efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                    centroid = self.sleeveLSeamFeature.plane.org
                    target = np.array(centroid)
                    vec = target - efL
                    oracle = vec/np.linalg.norm(vec)

            self.prevOracles[0] = np.array(oracle)
            obs = np.concatenate([obs, oracle]).ravel()

            oracle = np.zeros(3)
            sm_ix = 1 #separated mesh index (Right feature)
            if self.reset_number == 0:
                pass
            elif self.individualLimbProgress[1] > 0:
                oracle = self.sleeveRSeamFeature.plane.normal
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=12, clothscene=self.clothScene, meshgraph=self.separatedMeshes[sm_ix],returnOnlyGeo=False)

                if minGeoVix is not None:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMeshes[sm_ix].geoVectorAt(minGeoVix, side=vixSide)

                    if minContactGeodesic == 0:
                        minGeoVix = None
                else:
                    #oracle points to the garment when ef not in contact
                    efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
                    centroid = self.sleeveRSeamFeature.plane.org
                    target = np.array(centroid)
                    vec = target - efR
                    oracle = vec/np.linalg.norm(vec)
            self.prevOracles[1] = np.array(oracle)
            obs = np.concatenate([obs, oracle]).ravel()

        #obsolete
        '''
        elif self.oracleInObs: #rigid oracle
            oracle = np.zeros(3)
            if(self.limbProgress > 0):
                oracle = self.rigidClothFrame.toGlobal(np.array([0,0,1]))-self.rigidClothFrame.toGlobal(np.zeros(3))
                oracle /= np.linalg.norm(oracle)
            else:
                efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                oracle = self.rigidClothFrame.getCenter() - efL
                oracle /= np.linalg.norm(oracle)
            self.prevOracle = np.array(oracle)
            obs = np.concatenate([obs, oracle]).ravel()
        '''

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        if self.robotJointObs:  # if true, obs includes robot joint locations in world space
            locs = np.zeros(0)
            for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
                for jix,j in enumerate(iiwa.joints):
                    if(jix in self.redundantRoboJoints):
                        continue
                    locs = np.concatenate([locs, j.position_in_world_frame()])
            obs = np.concatenate([obs, locs]).ravel()


        if self.humanJointObs:
            locs = np.zeros(0)
            for j in self.robot_skeleton.joints:
                locs = np.concatenate([locs, j.position_in_world_frame()])
            obs = np.concatenate([obs, locs]).ravel()
            #print("human joint obs size: " + str(len(self.robot_skeleton.joints)))

        if self.hoopNormalObs:
            hoop_norm = self.rigidClothFrame.toGlobal(np.array([0, 0, -1])) - self.rigidClothFrame.toGlobal(np.zeros(3))
            hoop_norm /= np.linalg.norm(hoop_norm)
            obs = np.concatenate([obs, hoop_norm]).ravel()

        if self.actionScaleVarObs:
            obs = np.concatenate([obs, self.actionScaleVariation]).ravel()

        if self.jointLimVarObs:
            obs = np.concatenate([obs, self.jointConstraintVariation]).ravel()

        if self.weaknessScaleVarObs:
            obs = np.concatenate([obs, np.array([self.weaknessScale])]).ravel()

        if self.elbowConVarObs:
            normalized_elbow_rest = (self.elbow_rest-0.5)/2.35
            obs = np.concatenate([obs, np.array([normalized_elbow_rest])]).ravel()

        if self.SPDTargetObs:
            if self.humanSPDController is None:
                obs = np.concatenate([obs, np.zeros(22)]).ravel()
            else:
                obs = np.concatenate([obs, self.humanSPDController.target]).ravel()


        return obs

    def _get_robot_obs(self):

        obs = np.zeros(0)

        # human joint positions
        locs = np.zeros(0)
        for jix, j in enumerate(self.robot_skeleton.joints):
            if (jix in self.redundantHumanJoints):
                continue
            locs = np.concatenate([locs, j.position_in_world_frame()])
        # human end effectors
        locs = np.concatenate([locs, self.robot_skeleton.bodynodes[7].to_world(self.fingertip),
                               self.robot_skeleton.bodynodes[12].to_world(self.fingertip)])
        obs = np.concatenate([obs, locs]).ravel()

        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            #robot pose
            theta = np.array(iiwa.q)
            dtheta = np.array(iiwa.dq)
            obs = np.concatenate([obs, np.cos(theta[6:]), np.sin(theta[6:]), dtheta[6:]]).ravel()

            #robot joint positions
            rlocs = np.zeros(0)
            for jix, j in enumerate(iiwa.joints):
                if (jix in self.redundantRoboJoints):
                    continue
                rlocs = np.concatenate([rlocs, j.position_in_world_frame()])
            obs = np.concatenate([obs, rlocs]).ravel()

            #include robot end effector and target frame pos and orientation
            hn = iiwa.bodynodes[8]
            roboFrame = pyutils.ShapeFrame()
            roboFrame.setTransform(hn.T)
            self.robotEulerStates[iiwa_ix] = pyutils.getEulerAngles3(roboFrame.orientation)
            obs = np.concatenate([obs, hn.to_world(np.zeros(3)), self.robotEulerStates[iiwa_ix]]).ravel()
            if self.targetCentric:
                #obs = np.concatenate([obs, self.ikTarget, self.frameEulerState])
                obs = np.concatenate([obs, self.frameInterpolator[iiwa_ix]["target_pos"], self.frameInterpolator[iiwa_ix]["eulers"]])

        #force torques
        try:
            resultantFT = np.zeros(0)
            for handleNode in self.handleNodes:
                resultantFT = np.concatenate([resultantFT, handleNode.prev_avg_force * 0.05, handleNode.prev_avg_torque]).ravel()
            normalizedResultantFT = np.clip(resultantFT, -1, 1)

            obs = np.concatenate([obs, normalizedResultantFT]).ravel()
        except:
            print("failed obs FT")
            obs = np.concatenate([obs, np.zeros(6*self.numRobots)]).ravel()

        # capacitive sensor aggregate reading
        if self.robotCapacitiveObs:
            for capsens in self.capacitiveSensors:
                obs = np.concatenate([obs, capsens.getAggregateSensorReading() / 0.15])
            pass

        return obs

    def _get_obs(self):
        #this should return the active observation
        if self.dualPolicy:
            return np.concatenate([self._get_human_obs(), self._get_robot_obs()])
        elif self.isHuman:
            return self._get_human_obs()
        else:
            return self._get_robot_obs()

    def reset_model(self):
        self.rewardsData.reset()

        # reset any robot controller states
        for iiwa_controller in self.iiwaFrameControllers:
            iiwa_controller.reset()

        seeds=[]
        #seeds = [0, 2, 5, 8, 11, 20, 27, 35, 36, 47, 50, 51] #success seeds for stochastic policy
        #seeds = [0, 1, 2, 3, 5, 8, 11, 12, 13, 14, 18, 19, 20, 23, 27, 35, 38, 50] #success seeds for mean policy
        #difficultySeeds = [37, 39, 42]
        #seeds = seeds+difficultySeeds
        #seed = self.reset_number
        seeds = []  # 4 variations  seeding
        for j in range(4):
            for i in range(10):  # number of seed trials
                seeds.append(i + 40)
        seeds = [40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50]

        seeds = [40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                 40, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        #print(seeds)
        if False:
            try:
                seed = seeds[self.reset_number]
            except:
                print("out of seeds, exiting")
                exit()
                seed = self.reset_number
                #print("all given seeds simulated")

            #seed = 8
            #print("rollout: " + str(self.reset_number+1) +", seed: " + str(seed))
            print("Seeding: " + str(seed))
            random.seed(seed)
            self.np_random.seed(seed)
            np.random.seed(seed)
            self.setSeed = seed

        self.rewardTrajectory = []
        startTime = time.time()
        #try:
        #print("reset")
        self.cumulativeReward = 0
        self.dart_world.reset()
        self.clothScene.setSelfCollisionDistance(0.03)
        self.clothScene.reset()
        #self.clothScene.setFriction(0, 0.4)

        self.additionalResets()

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        self.avgtimings = {}
        self.reset_number += 1
        self.numSteps = 0

        #reload the policy file for "other"
        if not self.dualPolicy:
            try:
                if self.isHuman:
                    #print(self.robotPolicyFile)
                    self.otherPolicy = joblib.load(self.robotPolicyFile)
                    #self.otherPolicy = pickle.load(open(self.robotPolicyFile, "rb"))
                else:
                    self.otherPolicy = joblib.load(self.humanPolicyFile)
                    #self.otherPolicy = pickle.load(open(self.humanPolicyFile, "rb"))
            except:
            #    print("no policy file found...")
                pass

        #if self.reset_number == 1:
        #    self.reset()
        #print("now entering rollout reset_number: " + str(self.reset_number))
        #self.reset()
        return self._get_obs()
        #except:
        #    print("Failed on reset " + str(self.reset_number))

    def additionalResets(self):
        self.consecutiveInstabilities = 0
        self.maxContactForce = 0

        if self.targetFrameTracking["active"]:
            self.targetFrameTracking["history"] = []

        if self.collisionResult is None:
            self.collisionResult = CollisionResult.CollisionResult(self.dart_world)

        #vary the "weakness" of the character
        if self.actionScaleVarObs:
            self.actionScaleVariation = self.np_random.uniform(low=0.4, high=1.0, size=len(self.human_action_scale))
            #print("action scale variation: " + str(self.actionScaleVariation))

        if self.jointLimVarObs:
            self.jointConstraintVariation = self.np_random.uniform(low=0.5, high=1.0, size=self.robot_skeleton.ndofs)
            llim = np.multiply(np.array(self.initialJointConstraints[0]), self.jointConstraintVariation)
            ulim = np.multiply(np.array(self.initialJointConstraints[1]), self.jointConstraintVariation)
            #print("lower limits: " + str(llim))
            #print("upper limits: " + str(ulim))
            for dix,d in enumerate(self.robot_skeleton.dofs):
                if(math.isfinite(llim[dix])):
                    d.set_position_lower_limit(llim[dix])
                if(math.isfinite(ulim[dix])):
                    d.set_position_upper_limit(ulim[dix])

        if self.weaknessScaleVarObs:
            #self.weaknessScale = random.random()
            self.weaknessScale = random.uniform(0.0,1.0)
            #self.weaknessScale = 0.5
            #self.weaknessScale = 1.0
            #print("weaknessScale = " + str(self.weaknessScale))

            if self.variationTesting:
                #self.weaknessScale = self.variations[self.reset_number % len(self.variations)]
                self.weaknessScale = self.variations[int(self.reset_number / self.numSeeds)]
                if self.reset_number%self.numSeeds == 0:
                    self.viewer.captureDirectory = "/home/alexander/Documents/frame_capture_output/variations/" + str(int(self.reset_number/self.numSeeds)+1)
                    self.viewer.captureIndex = 0
                    #self.weaknessScale = self.variations[int(self.reset_number/self.numSeeds)]
                    print("reset capture directory to " + self.viewer.captureDirectory)
                    #print(self.weaknessScale)

        if self.elbowConVarObs:
            # sample a rest position
            self.elbow_rest = random.uniform(0.5, 2.85)
            self.elbow_rest = 0.75 #spec (easy)
            #self.elbow_rest = 2.0 #spec hard
            # testing range:
            #self.elbow_rest = 0.25 + (int(self.reset_number/10)/8.0) * 2.6
            print("elbow_rest = " + str(self.elbow_rest))
            # set the joint limits as boundary clamped, symmetrical range around rest
            self.robot_skeleton.dofs[16].set_position_lower_limit(
                max(self.elbow_rest - self.elbow_constraint_range, self.elbow_initial_limits[0]))
            self.robot_skeleton.dofs[16].set_position_upper_limit(
                min(self.elbow_rest + self.elbow_constraint_range, self.elbow_initial_limits[1]))
            # set rest position and spring
            self.robot_skeleton.dofs[16].set_rest_position(self.elbow_rest)
            self.robot_skeleton.dofs[16].set_spring_stiffness(5.0)
            self.robot_skeleton.dofs[16].set_position(self.elbow_rest)

        self.humanSPDController.target = np.array(self.robot_skeleton.q)
        self.humanSPDIntperolationTarget = np.array(self.humanSPDController.target)
        #if(self.reset_number > 0):
        #    print("ef_accuracy_info: " + str(self.ef_accuracy_info))
        #self.ef_accuracy_info = {'best': 0, 'worst': 0, 'total': 0, 'average': 0}

        if self.limbProgressGraphing:
            prefix = ""
            if self.variationTesting:
                prefix = self.viewer.captureDirectory+"/"

            # TODO: elbow variation testing
            if True:
                folderNum = max(int((self.reset_number-1) / 10), 0)
                print("folderNum: " + str(folderNum))
                prefix = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/" + str(
                    folderNum) + "/"
                #prefix = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/baseline/"
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
            # TODO: done - elbow variation testing

            self.limbProgressGraph.update()
            self.limbProgressGraph.save(prefix+"limbProgressGraph", prefix+"limbProgressGraphData")

            if self.variationTesting:
                if self.reset_number % self.numSeeds == 0:
                    self.limbProgressGraph.close()
                    self.limbProgressGraph = pyutils.LineGrapher(title="Limb Progress")

            # TODO: elbow variation testing
            if self.reset_number % 10 == 0:
                self.limbProgressGraph.close()
                self.limbProgressGraph = pyutils.LineGrapher(title="Limb Progress")
            # TODO: done - elbow variation testing

            self.limbProgressGraph.xdata = np.arange(600)
            self.limbProgressGraph.plotData(ydata=np.zeros(600))
            self.limbProgressGraph.update()

        if self.deformationGraphing:
            prefix = ""
            if self.variationTesting:
                prefix = self.viewer.captureDirectory+"/"

            # TODO: elbow variation testing
            if True:
                folderNum = max(int((self.reset_number-1) / 10), 0)
                print("folderNum: " + str(folderNum))
                prefix = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/" + str(folderNum) + "/"
                #prefix = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/baseline/"
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
            # TODO: done - elbow variation testing

            self.deformationGraph.update()
            self.deformationGraph.save(prefix+"deformationGraph", prefix+"deformationGraphData")

            if self.variationTesting:
                if self.reset_number % self.numSeeds == 0:
                    self.deformationGraph.close()
                    self.deformationGraph = pyutils.LineGrapher(title="Deformation")

            # TODO: elbow variation testing
            if self.reset_number % 10 == 0 and self.reset_number > 0:
                self.deformationGraph.close()
                self.deformationGraph = pyutils.LineGrapher(title="Deformation")

                folderNum = max(int((self.reset_number) / 10), 0)
                self.viewer.captureDirectory = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/" + str(folderNum)
                if not os.path.exists(self.viewer.captureDirectory):
                    os.makedirs(self.viewer.captureDirectory)
                self.viewer.captureIndex = 0
                # self.weaknessScale = self.variations[int(self.reset_number/self.numSeeds)]
                print("reset capture directory to " + self.viewer.captureDirectory)
                #exit(0)
            # TODO: done - elbow variation testing

            self.deformationGraph.xdata = np.arange(600)
            self.deformationGraph.plotData(ydata=np.zeros(600))
            self.deformationGraph.update()

        if self.FTGraphing:
            prefix = ""
            if self.variationTesting:
                prefix = self.viewer.captureDirectory+"/"

            self.FTGraph.update()
            self.FTGraph.save(prefix+"FTGraph", prefix+"FTGraphData")

            #if self.variationTesting:
            #    if self.reset_number % self.numSeeds == 0:
            self.FTGraph.close()
            self.FTGraph = pyutils.LineGrapher(title="Force/Torque at EF")

            self.FTGraph.xdata = np.arange(600*self.frame_skip)/self.frame_skip
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.plotData(ydata=np.zeros(600*self.frame_skip))
            self.FTGraph.update()

        if self.FTavgGraphing:
            prefix = ""
            if self.variationTesting:
                prefix = self.viewer.captureDirectory+"/"

            self.FTavgGraph.update()
            self.FTavgGraph.save(prefix+"FTavgGraph", prefix+"FTavgGraphData")

            #if self.variationTesting:
            #    if self.reset_number % self.numSeeds == 0:
            self.FTavgGraph.close()
            self.FTavgGraph = pyutils.LineGrapher(title="Force/Torque average at EF")

            self.FTavgGraph.xdata = np.arange(600)
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.plotData(ydata=np.zeros(600))
            self.FTavgGraph.update()

        if self.restPoseErrorGraphing:
            self.restPoseErrorGraph.save("restPoseErrorGraph", "restPoseErrorGraphData")
            self.restPoseErrorGraph.xdata = np.arange(250)
            self.restPoseErrorGraph.plotData(ydata=np.zeros(250))
            self.restPoseErrorGraph.update()

        if self.graphSPDError:
            self.SPDErrorGraph.close()
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)

        #do any additional resetting here
        self.handFirst = False
        #print(self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)))

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #qpos[16] = 1.9
        #qpos[1] = -0.5
        '''qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        '''
        self.set_state(qpos, qvel)
        #self.loadCharacterState(filename="characterState_1starmin")
        self.restPose = qpos

        #set iiwa states
        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            iiwa_pose = np.array(iiwa.q)
            iiwa_pose[:6] = np.array(self.iiwa_root_dofs[iiwa_ix])
            iiwa_pose[6:] = np.array(self.iiwa_rest)
            iiwa.set_positions(iiwa_pose)

        #setting the orientation of the pyBulletIiwa, other settings are unnecessary as we give rest poses for IK
        T = self.iiwa_skels[0].bodynodes[0].world_transform()
        tempFrame = pyutils.ShapeFrame()
        tempFrame.setTransform(T)
        root_quat = tempFrame.quat
        root_quat = (root_quat.x, root_quat.y, root_quat.z, root_quat.w)
        p.resetBasePositionAndOrientation(self.pyBulletIiwa, posObj=np.zeros(3), ornObj=root_quat)
        #self.setPosePyBullet(self.iiwa_skel.q[6:])

        #For now, this is manual for 2 mirrored robots
        #pick 2 points with rejection sampling on local reachability and max distance
        handleMaxDistance = 0.5
        handleMinXDistance = 0.1
        diskRad = self.robotPathParams['p0_disk_rad']
        p0L = self.iiwa_skels[0].bodynodes[3].to_world(np.zeros(3)) + pyutils.sampleDirections(num=1)[0]* diskRad
        p0R = self.iiwa_skels[1].bodynodes[3].to_world(np.zeros(3)) + pyutils.sampleDirections(num=1)[0]* diskRad
        good = False
        r_pivotL = self.iiwa_skels[0].bodynodes[3].to_world(np.zeros(3))
        r_pivotR = self.iiwa_skels[1].bodynodes[3].to_world(np.zeros(3))
        while(not good):
            good = True
            #xz0 = np.array([p0[0], p0[2]])

            #cut off points too close or behind the robot in x (side)
            if p0L[0] > (r_pivotL[0] - self.robotPathParams['p0_depth_offset']):
                good = False

            if p0R[0] < (r_pivotR[0] + self.robotPathParams['p0_depth_offset']):
                good = False

            # cut off points too close or behind the robots in z
            if p0L[2] > (r_pivotL[2] - self.robotPathParams['p0_depth_offset']) or p0R[2] > (r_pivotR[2] - self.robotPathParams['p0_depth_offset']):
                good = False

            if np.linalg.norm(p0L-p0R) > handleMaxDistance:
                good = False

            if abs(p0L[0] - p0R[0]) < handleMinXDistance:
                good = False

            if not good:
                p0L = self.iiwa_skels[0].bodynodes[3].to_world(np.zeros(3)) + pyutils.sampleDirections(num=1)[0] * diskRad
                p0R = self.iiwa_skels[1].bodynodes[3].to_world(np.zeros(3)) + pyutils.sampleDirections(num=1)[0] * diskRad


        self.iiwa_ik_targets[0].setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=p0L)
        self.iiwa_ik_targets[1].setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=p0R)

        #now IK to find robot poses
        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            tar_quat = self.iiwa_ik_targets[iiwa_ix].quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            result = None
            self.setPosePyBullet(pose=iiwa.q[6:])
            if (self.ikOrientation):
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                      endEffectorLinkIndex=8,
                                                      targetPosition=self.iiwa_ik_targets[iiwa_ix].org - self.iiwa_root_dofs[iiwa_ix][3:],
                                                      targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.iiwa_dof_llim.tolist(),
                                                      upperLimits=self.iiwa_dof_ulim.tolist(),
                                                      jointRanges=self.iiwa_dof_jr.tolist(),
                                                      restPoses=iiwa.q[6:].tolist(),
                                                      maxNumIterations=300 #NEED the extra iterations...
                                                      )
            else:
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                      endEffectorLinkIndex=8,
                                                      targetPosition=self.iiwa_ik_targets[iiwa_ix].org - self.iiwa_root_dofs[iiwa_ix][3:],
                                                      # targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.iiwa_dof_llim.tolist(),
                                                      upperLimits=self.iiwa_dof_ulim.tolist(),
                                                      jointRanges=self.iiwa_dof_jr.tolist(),
                                                      restPoses=iiwa.q[6:].tolist(),
                                                      maxNumIterations=10
                                                      )

            self.previousIKResults[iiwa_ix] = np.array(result)
            #self.setPosePyBullet(result)
            iiwa.set_velocities(np.zeros(len(iiwa.dq)))
            iiwa.set_positions(np.concatenate([np.array(self.iiwa_root_dofs[iiwa_ix]), result]))

            hn = iiwa.bodynodes[8]  # hand node

            current_state_frame = pyutils.ShapeFrame()
            current_state_frame.setTransform(hn.world_transform())

            self.robotEulerStates[iiwa_ix] = pyutils.getEulerAngles3(current_state_frame.orientation)
            self.frameEulerStates[iiwa_ix] = pyutils.getEulerAngles3(self.iiwa_ik_targets[iiwa_ix].orientation)
            #print("Iiwa " + str(iiwa_ix) + " euler states:")
            #print(" robot: " + str(self.robotEulerStates[iiwa_ix]))
            #print(" frame: " + str(self.frameEulerStates[iiwa_ix]))

            #self.frameInterpolator["target_pos"] = self.iiwa_skel.bodynodes[8].to_world(np.zeros(3))
            self.frameInterpolator[iiwa_ix]["target_pos"] = np.array(self.iiwa_ik_targets[iiwa_ix].org)
            self.frameInterpolator[iiwa_ix]["target_frame"] = np.array(self.iiwa_ik_targets[iiwa_ix].orientation)
            self.frameInterpolator[iiwa_ix]["eulers"] = np.array(self.frameEulerStates[iiwa_ix])

        #setup the handle nodes
        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            if len(self.handleNodes) > iiwa_ix:
                handleNode = self.handleNodes[iiwa_ix]
                hn = iiwa.bodynodes[8]
                handleNode.clearHandles()
                self.clothScene.setParticleConstraintMode(1)
                #self.handle_node.addVertices(verts=[727, 138, 728, 1361, 730, 961, 1213, 137, 724, 1212, 726, 960, 964, 729, 155, 772])
                if iiwa_ix == 0:
                    handleNode.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])
                else:
                    handleNode.addVertices(verts=[1502, 808, 554, 1149, 819, 1619, 674, 1918, 1528, 1654, 484, 1590, 1802, 1924])

                handleNode.setOrgToCentroid()
                handleNode.setOrientation(R=hn.T[:3, :3])

                # self.handle_node.setTransform(self.iiwa_skel.bodynodes[8].T)
                #self.handle_node.setOrgToCentroid()

                #if self.updateHandleNodeFrom >= 0:
                #    self.handle_node.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
                handleNode.recomputeOffsets()
                handleNode.updatePrevConstraintPositions()

        clothFrame = pyutils.ShapeFrame(org=self.clothScene.getVertexCentroid())
        relativeHandlePosition = self.handleNodes[0].org - self.handleNodes[1].org

        clothCentroid = self.clothScene.getVertexCentroid()
        handleCentroid = (self.handleNodes[0].org + self.handleNodes[1].org)/2.0

        #first translate the handleCentroid to the origin
        self.clothScene.translateCloth(0, -1.0*handleCentroid)
        #then rotate the cloth about that point
        self.clothScene.rotateCloth(cid=0, R=pyutils.rotateZ(math.pi))
        self.clothScene.rotateCloth(cid=0, R=pyutils.rotateY(math.pi))

        #now translate to desired location
        self.clothScene.translateCloth(0, (p0L+p0R)/2.0)

        for handleNode in self.handleNodes:
            handleNode.setOrgToCentroid()
            handleNode.recomputeOffsets()
            handleNode.updatePrevConstraintPositions()

        #now simulate the cloth while interpolating the handle node positions
        simFrames = 100
        hL_init = np.array(self.handleNodes[0].org)
        hR_init = np.array(self.handleNodes[1].org)
        for frame in range(simFrames):
            self.handleNodes[0].org = hL_init + (p0L-hL_init)*(frame/(simFrames-1))
            self.handleNodes[1].org = hR_init + (p0R-hR_init)*(frame/(simFrames-1))
            self.handleNodes[0].updateHandles()
            self.handleNodes[1].updateHandles()
            self.clothScene.step()


        if self.sleeveLSeamFeature is not None:
            # the normal should always point toward the origin in this task
            self.sleeveLSeamFeature.fitPlane()
            toward_origin = self.sleeveLSeamFeature.plane.org
            toward_origin = toward_origin / np.linalg.norm(toward_origin)
            self.sleeveLSeamFeature.fitPlane(normhint=toward_origin)

        if self.sleeveRSeamFeature is not None:
            self.sleeveRSeamFeature.fitPlane()
            toward_origin = self.sleeveRSeamFeature.plane.org
            toward_origin = toward_origin / np.linalg.norm(toward_origin)
            self.sleeveRSeamFeature.fitPlane(normhint=toward_origin)

        #setup the geodesic fields
        if self.reset_number == 0:
            for sm in self.separatedMeshes:
                sm.initSeparatedMeshGraph()
                sm.updateWeights()
            #Manually script this for each task
            self.separatedMeshes[0].computeGeodesic(feature=self.sleeveLSeamFeature, oneSided=True, side=0, normalSide=1)
            vertex_blacklist = [2055]
            self.boundaryAddedVerts,self.boundarySkippedVerts = self.separatedMeshes[1].computeGeodesic(feature=self.sleeveRSeamFeature, oneSided=True, side=0, normalSide=1, boundary_skip_vertices=vertex_blacklist)
            #print("boundary added vertices: " + str(self.boundaryAddedVerts))
            #print("boundary skipped vertices: " + str(skipped))

        if self.simulateCloth:
            #sum of limb progress
            if self.limbProgressReward:
                self.limbProgress = 0.0
                for tix,d_tar in enumerate(self.dressingTargets):
                    self.individualLimbProgress[tix] = d_tar.getLimbProgress()
                    self.limbProgress += self.individualLimbProgress[tix]
                #pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL, offset=self.fingertip), feature=self.sleeveLSeamFeature)

        a=0

    def extraRenderFunction(self):

        # render capacitive sensor readings
        if self.robotCapacitiveObs:
            for capsens in self.capacitiveSensors:
                capsens.draw()

        #draw shoulder plane
        if False:
            s0 = self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))
            s1 = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
            snorm = s1-s0
            _snorm = snorm/np.linalg.norm(snorm)
            sb1 = np.cross(_snorm, np.array([0,1.0,0]))
            sb2 = np.cross(_snorm, sb1)
            shoulderPlane = Plane(org=s1, normal=_snorm, b1=sb1, b2=sb2)
            shoulderPlane.draw()

        #draw interpolation target frame
        for fix,f_int in enumerate(self.frameInterpolator):
            if f_int["active"]:
                renderFrame = pyutils.ShapeFrame()
                renderFrame.setOrg(org=f_int["target_pos"])
                renderFrame.orientation = np.array(f_int["target_frame"])
                renderFrame.updateQuaternion()
                # renderFrame.drawFrame(size=1.1)
                v0 = renderFrame.toGlobal(p=[1, 0, 0]) - renderFrame.org
                v1 = renderFrame.toGlobal(p=[0, 1, 0]) - renderFrame.org
                v2 = renderFrame.toGlobal(p=[0, 0, 1]) - renderFrame.org
                renderUtils.drawArrowAxis(org=f_int["target_pos"], v0=v0, v1=v1, v2=v2, scale=0.5)
                renderUtils.setColor(color=[0, 0, 0])
                renderUtils.drawLineStrip(points=[self.iiwa_ik_targets[fix].org, f_int["target_pos"]])
                renderUtils.drawSphere(
                    pos=renderFrame.org + renderFrame.org - renderFrame.toGlobal(p=f_int["localOffset"]),
                    rad=0.02)

        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()


        if False:
            for handleNode in self.handleNodes:
                handleNode.draw()
                if self.humanRobotCollision:
                    handleNode.drawForce(color=(1.0,0.2,0.2))
                else:
                    handleNode.drawForce()


        #draw initial ef locations
        renderUtils.setColor(color=[1,0,1])
        for p in self.initialIiwaEfs:
            renderUtils.drawSphere(pos=p)

        #TODO: more than one oracle
        renderUtils.setColor(color=[0.0, 0.0, 0])
        if(self.renderOracle):
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            renderUtils.drawArrow(p0=efL, p1=efL+self.prevOracles[0]*0.2)
            renderUtils.drawArrow(p0=efR, p1=efR+self.prevOracles[1]*0.2)

        if self.renderHapticObs:
            self.collisionResult.update()
            #render haptic readings
            haptic_pos = self.clothScene.getHapticSensorLocations()
            haptic_radii = self.clothScene.getHapticSensorRadii()
            haptic_forces = self.getCumulativeHapticForcesFromRigidContacts()
            for h in range(self.clothScene.getNumHapticSensors()):
                renderUtils.setColor(color=[1, 1, 0])
                f = haptic_forces[h*3:h*3+3]
                f_mag = np.linalg.norm(f)
                if(f_mag > 0.001):
                    renderUtils.setColor(color=[0, 1, 0])
                renderUtils.drawSphere(pos=haptic_pos[h*3:h*3+3], rad=haptic_radii[h]*1.1, solid=False)
                if (f_mag > 0.001):
                    renderUtils.drawArrow(p0=haptic_pos[h*3:h*3+3], p1=haptic_pos[h*3:h*3+3]+f)

        if True: #draw IK
            for iiwa_ix,iiwa in enumerate(self.iiwa_skels):
                renderUtils.setColor(color=[1.0, 0, 0])
                renderUtils.drawSphere(self.iiwa_ik_targets[iiwa_ix].org)
                renderUtils.setColor(color=[0, 1.0, 0])
                efpos = iiwa.bodynodes[9].to_world(np.zeros(3))
                #renderUtils.drawLines(lines=[[np.zeros(3), self.sawyer_skel.bodynodes[3].to_world(np.zeros(3))]])
                renderUtils.drawSphere(pos=efpos)

                efdir = iiwa.bodynodes[9].to_world(np.array([0, 1.0, 0])) - efpos
                efdir /= np.linalg.norm(efdir)
                renderUtils.drawArrow(p0=efpos, p1=efpos+efdir*0.2)
                renderUtils.setColor(color=[0, 0, 0])
                renderUtils.drawLines(lines=[
                    [efpos, iiwa.bodynodes[9].to_world(np.array([0, 1.0, 0]))],
                    [efpos, iiwa.bodynodes[9].to_world(np.array([1.0, 0, 0]))],
                    [efpos, iiwa.bodynodes[9].to_world(np.array([0, 0, 1.0]))]
                ])

                #if (self.targetCentric):
                if True:
                    renderUtils.setColor(color=[1.0,0,0])
                    efpos = np.array(self.iiwa_ik_targets[iiwa_ix].org)
                    #efdir = self.orientationTarget.toGlobalVec(np.array([0.0, 1.0, 0]))
                    efdir = self.iiwa_ik_targets[iiwa_ix].toGlobal(p=np.array([0.0, 1.0, 0])) - self.iiwa_ik_targets[iiwa_ix].toGlobal(p=np.zeros(3))
                    efdir = efdir / np.linalg.norm(efdir)
                    renderUtils.drawArrow(p0=efpos, p1=efpos + efdir * 0.2)


        #draw pybullet sawyer body positions
        if False:
            for i in range(13):
                #print(p.getLinkState(self.pyBulletSawyer, i))
                pybullet_state = p.getLinkState(self.pyBulletIiwa, i)[0]
                renderUtils.setColor(color=[0, 0.0, 0])
                renderUtils.drawSphere(pybullet_state)

        #inner bicep indicator strips
        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])


        #renderUtils.drawBox(cen=self.iiwa_root_dofs[3:], dim=np.array([0.2, 0.05, 0.2]))


        if(self.renderCloth):
            if self.sleeveLSeamFeature is not None:
                self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveRSeamFeature is not None:
                self.sleeveRSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                #self.sleeveRSeamFeature.drawVertices(highlight_edges=True)
        #    if self.sleeveLEndFeature is not None:
        #        self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        #    if self.sleeveLMidFeature is not None:
        #        self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)


        #render the added or skipped geodesic verts from first wave
        if False:
            #render added and skipped boundary vertices

            renderUtils.setColor(color=[0, 1, 0])
            for a in self.boundaryAddedVerts:
                #added_positions.append(self.clothScene.getVertexPos(a))
                renderUtils.drawSphere(pos=self.clothScene.getVertexPos(vid=a),rad=0.002)
            renderUtils.setColor(color=[1, 0, 0])
            for s in self.boundarySkippedVerts:
                #skipped_positions.append(self.clothScene.getVertexPos(s))
                renderUtils.drawSphere(pos=self.clothScene.getVertexPos(vid=s), rad=0.002)

        # render geodesic
        if False:
            sm_ix = 1 #default left feature geodesic
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separatedMeshes[sm_ix].nodes[v + self.separatedMeshes[sm_ix].numv].geodesic
                side0geo = self.separatedMeshes[sm_ix].nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMeshes[sm_ix].maxGeo, value=self.separatedMeshes[sm_ix].maxGeo-side0geo))
                renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMeshes[sm_ix].maxGeo, value=self.separatedMeshes[sm_ix].maxGeo-side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)


        m_viewport = self.viewer.viewport
        # print(m_viewport)

        self.clothScene.drawText(x=15., y=self.viewer.viewport[3] - 80, text="Max Contact Force = " + str(self.maxContactForce), color=(0., 0, 0))
        self.clothScene.drawText(x=15., y=15, text="Time = " + str(self.numSteps * self.dt), color=(0., 0, 0))


        if self.variationTesting:
            #print(self.setSeed)
            #print(self.weaknessScale)
            self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - 60, text="(Seed, Variation): (%i, %0.2f)" % (self.setSeed,self.weaknessScale), color=(0., 0, 0))
            #self.clothScene.drawText(x=15., y=15, text="Time = " + str(self.numSteps * self.dt), color=(0., 0, 0))
            self.clothScene.drawText(x=15., y=30, text="Steps = " + str(self.numSteps) + ", dt = " + str(self.dt) + ", frameskip = " + str(self.frame_skip), color=(0., 0, 0))
        #elif self.demoRendering:
        #    #still want time displayed
        #    self.clothScene.drawText(x=15., y=15, text="Time = " + str(self.numSteps * self.dt), color=(0., 0, 0))

        if self.robotCapacitiveObs:
            for capsens in self.capacitiveSensors:
                robo_ef = capsens.bodynode.to_world(np.zeros(3))
                windowef = self.viewer.project(robo_ef)
                sensorReading = capsens.getAggregateSensorReading()
                for s in range(capsens.numSensorRayZones):
                    self.clothScene.drawText(x=windowef[0], y=windowef[1]+16*s,
                                         text="Z["+str(s)+"]: %0.4f" % sensorReading[s],
                                         color=(1.0, 0., 0.))

        if self.weaknessScaleVarObs:
            self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - 60,
                                     text="(Seed, Variation): (%i, %0.2f)" % (self.setSeed, self.weaknessScale),
                                     color=(0., 0, 0))

        if self.elbowConVarObs:
            # render elbow stiffness variation
            if not self.weaknessScaleVarObs:
                self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - 60,
                                         text="Seed: (%i)" % self.setSeed,
                                         color=(0., 0, 0))

            self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - 80,
                                     text="Elbow Rest Value [0,1] = %0.2f" % ((self.elbow_rest - 0.25)/2.6),
                                     color=(0., 0, 0))

        if self.renderUI and not self.demoRendering:
            if self.renderRewardsData:
                self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                        dimensions=[400, -m_viewport[3] + 30])

            textHeight = 15
            textLines = 2

            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Seed = " + str(self.setSeed), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps) + ", dt = " + str(self.dt) + ", frameskip = " + str(self.frame_skip), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Time = " + str(self.numSteps*self.dt), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Path Time = " + str(self.numSteps*self.ikPathTimeScale), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Previous Avg Geodesic = " + str(self.prevAvgGeodesic), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Limb Progress = " + str(self.limbProgress), color=(0., 0, 0))
            textLines += 1

            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)

            #render the constraint and action_scale variations
            if self.jointLimVarObs:
                self.clothScene.drawText(x=360., y=self.viewer.viewport[3]-13, text="J_var", color=(0., 0, 0))
            if self.actionScaleVarObs:
                self.clothScene.drawText(x=410., y=self.viewer.viewport[3]-13, text="A_scale", color=(0., 0, 0))

            for d in range(self.robot_skeleton.ndofs):
                if self.jointLimVarObs:
                    self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - d*20 - 23, text="%0.2f" % self.jointConstraintVariation[d], color=(0., 0, 0))
                if self.actionScaleVarObs:
                    self.clothScene.drawText(x=410., y=self.viewer.viewport[3] - d*20 - 23, text="%0.2f" % self.actionScaleVariation[d], color=(0., 0, 0))

            #render unilateral weakness variation
            self.clothScene.drawText(x=360., y=self.viewer.viewport[3] - 60, text="Weakness Scale Value = %0.2f" % self.weaknessScale, color=(0., 0, 0))


            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])

            #draw Iiwa positions vs. limits
            iiwa_ix = 0
            for d in range(7):
                self.clothScene.drawText(x=15., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.dofs[6 + d].position_lower_limit(),), color=(0., 0, 0))
                self.clothScene.drawText(x=100., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.q[6 + d],), color=(0., 0, 0))
                self.clothScene.drawText(x=200., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.dofs[6 + d].position_upper_limit(),), color=(0., 0, 0))

                val = (self.iiwa_skels[iiwa_ix].q[6 + d] - self.iiwa_skels[iiwa_ix].dofs[6 + d].position_lower_limit()) / (self.iiwa_skels[iiwa_ix].dofs[6 + d].position_upper_limit() - self.iiwa_skels[iiwa_ix].dofs[6 + d].position_lower_limit())
                tar = (self.previousIKResults[iiwa_ix][d] - self.iiwa_skels[iiwa_ix].dofs[6 + d].position_lower_limit()) / (self.iiwa_skels[iiwa_ix].dofs[6 + d].position_upper_limit() - self.iiwa_skels[iiwa_ix].dofs[6 + d].position_lower_limit())
                renderUtils.drawProgressBar(topLeft=[75, self.viewer.viewport[3] - 450 - d*20], h=16, w=120, progress=val, origin=0.5, features=[tar], color=[1.0, 0.0, 0])


                self.clothScene.drawText(x=250., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.force_lower_limits()[6 + d],), color=(0., 0, 0))
                self.clothScene.drawText(x=335., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.forces()[6 + d],), color=(0., 0, 0))
                self.clothScene.drawText(x=435., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.iiwa_skel.force_upper_limits()[6 + d],), color=(0., 0, 0))

                tval = (self.iiwa_skels[iiwa_ix].forces()[6 + d] - self.iiwa_skels[iiwa_ix].force_lower_limits()[6 + d]) / (self.iiwa_skels[iiwa_ix].force_upper_limits()[6 + d] - self.iiwa_skels[iiwa_ix].force_lower_limits()[6 + d])
                renderUtils.drawProgressBar(topLeft=[310, self.viewer.viewport[3] - 450 - d * 20], h=16, w=120, progress=tval, origin=0.5, color=[1.0, 0.0, 0])

        # render target pose (NOTE: not updated for multibot)
        '''
        if self.viewer is not None and self.renderIKGhost:
            q = np.array(self.iiwa_skel.q)
            dq = np.array(self.iiwa_skel.dq)
            self.iiwa_skel.set_positions(np.concatenate([np.array(self.iiwa_root_dofs), self.previousIKResult]))
            # self.viewer.scene.render(self.viewer.sim)
            self.iiwa_skel.render()
            self.iiwa_skel.set_positions(q)
            self.iiwa_skel.set_velocities(dq)
        '''

        # render SPD target ghost
        if self.renderSPDGhost:
            q = np.array(self.robot_skeleton.q)
            dq = np.array(self.robot_skeleton.dq)
            self.robot_skeleton.set_positions(self.humanSPDController.target)
            #self.robot_skeleton.render_with_color(color=[0.6,0.8,0.6])

            self.robot_skeleton.set_positions(self.humanSPDIntperolationTarget)
            #self.robot_skeleton.render_with_color(color=[0.8, 0.6, 0.6])

            if self.skelCapsulesDefined:
                renderUtils.setColor(color=[0.8, 0.6, 0.6])
                #print(self.skelCapsules)
                for capsule in self.skelCapsules:
                    p0 = self.robot_skeleton.bodynodes[capsule[0]].to_world(capsule[2])
                    p1 = self.robot_skeleton.bodynodes[capsule[3]].to_world(capsule[5])
                    renderUtils.drawCapsule(p0=p0, p1=p1, r0=capsule[1], r1=capsule[4])

            #self.robot_skeleton.render_with_color(color=[0.6, 0.6, 0.7+additional_color])

            self.robot_skeleton.set_positions(q)
            self.robot_skeleton.set_velocities(dq)

    def saveGripperState(self, filename=None):
        print("saving gripper state")
        if filename is None:
            filename = "gripperState"
        print("filename " + str(filename))
        f = open(filename, 'w')
        for ix, dof in enumerate(self.dart_world.skeletons[0].q):
            if ix > 0:
                f.write(" ")
            if ix < 3:
                f.write(str(self.handleNode.org[ix]))
            else:
                f.write(str(dof))

        f.write("\n")

        for ix, dof in enumerate(self.dart_world.skeletons[0].dq):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))
        f.close()

    # set a pose in the pybullet simulation env
    def setPosePyBullet(self, pose):
        count = 0
        for i in range(p.getNumJoints(self.pyBulletIiwa)):
            jinfo = p.getJointInfo(self.pyBulletIiwa, i)
            if (jinfo[3] > -1):
                p.resetJointState(self.pyBulletIiwa, i, pose[count])
                count += 1

    def checkIKSplineValidity(self):
        steps = 1.0/self.ikPathTimeScale #number of steps to complete the path
        results = []
        for i in range(math.ceil(steps)):
            t = i*self.ikPathTimeScale
            ikTarget = self.ikPath.pos(t)
            rigidClothTargetFrame = pyutils.BoxFrame()
            rigidClothTargetFrame.setFromDirectionandUp(dir=-ikTarget, up=np.array([0, -1.0, 0]),org=ikTarget)
            tar_quat = rigidClothTargetFrame.quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            pose = []
            for i in range(p.getNumJoints(self.pyBulletIiwa)):
                jinfo = p.getJointInfo(self.pyBulletIiwa, i)
                if (jinfo[3] > -1):
                    pose.append(p.getJointState(self.pyBulletIiwa, i)[0])

            result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                  endEffectorLinkIndex=12,
                                                  targetPosition=ikTarget - self.iiwa_root_dofs[3:],
                                                  targetOrientation=tar_quat,
                                                  lowerLimits=self.iiwa_dof_llim.tolist(),
                                                  upperLimits=self.iiwa_dof_ulim.tolist(),
                                                  jointRanges=self.iiwa_dof_jr.tolist(),
                                                  restPoses=pose
                                                  )
            results.append(np.array(result))
            self.setPosePyBullet(result)

        vels = []
        invalid_count = 0
        for r in range(1,len(results)):
            #print(results[r])
            vels.append((results[r]-results[r-1])/self.dt)
            for d in range(7):
                if(abs(vels[-1][d]) > self.iiwa_skel.dofs[d+6].velocity_upper_limit()):
                    invalid_count += 1
        print("Spline checked with " + str(invalid_count) + " invalid IK velocities.")
        self.setPosePyBullet(np.zeros(7))

    def checkPoseSplineValidity(self):
        steps = 1.0 / self.ikPathTimeScale  # number of steps to complete the path
        results = []
        for i in range(math.ceil(steps)):
            t = i * self.ikPathTimeScale
            results.append(self.posePath.pos(t=t))
        vels = []
        invalid_count = 0
        for r in range(1, len(results)):
            # print(results[r])
            vels.append((results[r] - results[r - 1]) / self.dt)
            for d in range(7):
                if (abs(vels[-1][d]) > self.iiwa_skel.dofs[d + 6].velocity_upper_limit()):
                    invalid_count += 1
        print("Spline checked with " + str(invalid_count) + " invalid pose velocities.")

    def getCumulativeHapticForcesFromRigidContacts(self, mag_scale=40.0):
        #force magnitudes are clamped to mag_scale and then normalized by it to [0,1]
        self.collisionResult.update()
        sensor_pos = self.clothScene.getHapticSensorLocations()
        sensor_rad = self.clothScene.getHapticSensorRadii()
        relevant_contacts = []
        for ix, c in enumerate(self.collisionResult.contacts):
            # add a contact if the human skeleton is involved
            if (c.skel_id1 == self.robot_skeleton.id or c.skel_id2 == self.robot_skeleton.id):
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
                if (vpn.dot(fn) > -vpn.dot(fn)):  # force pointing toward the sensor is correct
                    forces[best_hs] += c.force
                else:  # reverse a force pointing away from the sensor
                    forces[best_hs] += -c.force
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
            f /= mag_scale
            f_mag = np.linalg.norm(f)
            if(f_mag > 1.0):
                f /= f_mag
            result[ix*3:ix*3+3] = f
        return result

    def getHoopForcesFromRigidContacts(self):
        #force magnitudes are clamped to mag_scale and then normalized by it to [0,1]
        self.collisionResult.update()
        relevant_forces = []
        relevant_points = []
        hoop_bodynodes = [16, 17, 18, 19]
        for ix, c in enumerate(self.collisionResult.contacts):
            # add a contact if the human skeleton is involved
            if(c.skel_id2 == self.iiwa_skel.id and c.skel_id1 == self.iiwa_skel.id):
                a=0 #don't coutn self-collision forces
            if (c.skel_id2 == self.iiwa_skel.id):
                if(hoop_bodynodes.__contains__(c.bodynode_id2)):
                    relevant_forces.append(-c.force)
                    relevant_points.append(c.point)
            elif (c.skel_id1 == self.iiwa_skel.id):
                if(hoop_bodynodes.__contains__(c.bodynode_id1)):
                    relevant_forces.append(c.force)
                    relevant_points.append(c.point)

        return relevant_forces, relevant_points

    def updateHandleContactForceTorques(self, maxClamp=10.0):
        self.stepMaxContactForce = 0
        self.collisionResult.update()
        for iiwa_ix, iiwa in enumerate(self.iiwa_skels):
            handleNode = self.handleNodes[iiwa_ix]
            if handleNode is not None:
                handleNode.contactForce = np.zeros(3)
                handleNode.contactTorque = np.zeros(3)
                for ix, c in enumerate(self.collisionResult.contacts):
                    # add a contact if the human skeleton is involved with the robot EF (FT sensor)
                    if (c.skel_id1 == iiwa.id and c.skel_id2 == self.robot_skeleton.id):
                        self.humanRobotCollision = True
                        self.maxContactForce = max(self.maxContactForce, np.linalg.norm(c.force))
                        self.stepMaxContactForce = max(self.stepMaxContactForce, np.linalg.norm(c.force))
                        if(c.bodynode_id1 == iiwa.bodynodes[9].id):
                            force = np.array(c.force)
                            mag = np.linalg.norm(force)
                            if(mag > maxClamp):
                                force /= mag
                                force *= maxClamp
                                handleNode.contactForce += force
                                handleNode.contactTorque += np.cross(c.point-handleNode.org, force)
                    if (c.skel_id2 == iiwa.id and c.skel_id1 == self.robot_skeleton.id):
                        self.humanRobotCollision = True
                        self.maxContactForce = max(self.maxContactForce, np.linalg.norm(c.force))
                        self.stepMaxContactForce = max(self.stepMaxContactForce, np.linalg.norm(c.force))
                        if (c.bodynode_id2 == iiwa.bodynodes[9].id):
                            force = np.array(c.force)
                            mag = np.linalg.norm(force)
                            if (mag > maxClamp):
                                force /= mag
                                force *= maxClamp
                            handleNode.contactForce -= force
                            handleNode.contactTorque -= np.cross(c.point - handleNode.org, force)

    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None, resetFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc, self.reset_model)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            #default setup (in front of person)
            self._get_viewer().scene.tb.trans[2] = -3.5
            #self._get_viewer().scene.tb._set_theta(180)
            #self._get_viewer().scene.tb._set_phi(180)
            self._get_viewer().scene.tb._set_orientation(180,180)

            #recording angle rigid frame (side)
            #self._get_viewer().scene.tb._trans = [-0.40000000000000019, 0.0, -2.0999999999999988]
            #rot = [-0.078705687066204968, 0.5423547110155762, 0.067527388204703831, 0.83372467524051252]
            #pyutils.setTrackballOrientation(self.viewer.scene.tb, rot)

            # recording angle from front of person with robot on left side...
            self._get_viewer().scene.tb._trans = [0.34000000000000019, 0.020000000000000004, -2.0999999999999988]
            rot = [-0.094887037321912837, -0.91548784322523225, -0.071299301298955647, 0.38444098206273875]
            pyutils.setTrackballOrientation(self.viewer.scene.tb, rot)

            #self._get_viewer().scene.tb._set_orientation(-8.274256683701712,2.4687256068775723)
            #render side view

        self.track_skeleton_id = 0
        if not self.renderDARTWorld:
            self.viewer.renderWorld = False
        self.clothScene.renderCollisionCaps = True
        self.clothScene.renderCollisionSpheres = True

    def add_external_step_forces(self):
        #for d in range(19,20):
        #    self.sawyer_skel.bodynodes[d].add_ext_force(_force=np.array([0, -9.8, 0]))
        a=0

    def set_param_values(self, params):
        print("setting param values: " + str(params))

    def print_iiwa_bodynode_transforms(self):
        for b in self.iiwa_skel.bodynodes:
            print(b.name)
            print(b.world_transform())

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a = 0
        # collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0., 0, 0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.175, 0]))
        cs4 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs5 = self.robot_skeleton.bodynodes[5].to_world(z)
        cs6 = self.robot_skeleton.bodynodes[6].to_world(z)
        cs7 = self.robot_skeleton.bodynodes[7].to_world(z)
        cs8 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        cs9 = self.robot_skeleton.bodynodes[9].to_world(z)
        cs10 = self.robot_skeleton.bodynodes[10].to_world(z)
        cs11 = self.robot_skeleton.bodynodes[11].to_world(z)
        cs12 = self.robot_skeleton.bodynodes[12].to_world(z)
        cs13 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
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
        if np.isnan(np.sum(
                collisionSpheresInfo)):  # this will keep nans from propagating into PhysX resulting in segfault on reset()
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
            for i in range(len(collisionCapsuleInfo)):
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

    def checkInvalidDynamics(self):
        if not np.isfinite(self.robot_skeleton.q).all():
            print("Infinite value detected...")# + str(self.robot_skeleton.q))
            return True
        elif np.amax(np.absolute(self.robot_skeleton.q)) > 5:
            print("Detecting potential instability...")# + str(self.robot_skeleton.q))
            return True
        for iiwa_ix,iiwa in enumerate(self.iiwa_skels):
            if not np.isfinite(iiwa.q).all():
                print("Infinite value detected (robot "+str(iiwa_ix)+")...")# + str(self.dart_world.skeletons[0].q))
                return True
            elif np.amax(np.absolute(iiwa.q)) > 5:
                print("Detecting potential instability (robot "+str(iiwa_ix)+")...")# + str(self.dart_world.skeletons[0].q))
                return True

def LERP(p0, p1, t):
    return p0 + (p1 - p0) * t
