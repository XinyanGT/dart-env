# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
import datetime
import math
import pickle

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
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        self.env = env #needed to set env state variables on setup for use
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
        #print("now using mean action...")
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

class JacketRController(Controller):
    def __init__(self, env):
        obs_subset = [(0, 163)]
        name = "JacketR"
        #policyfilename = "experiment_2018_01_06_jacketR2"
        policyfilename = "experiment_2018_05_20_jacketr"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0.0, -0.085, 0.0])

        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        # geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveRMidFeature, oneSided=True, side=0, normalSide=0)

        # feature/oracle setup
        self.env.focusFeature = self.env.sleeveRMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 7  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveRSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 12

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesR,offset=np.array([0,-0.085,0])), feature=self.env.sleeveRSeamFeature)

    def transition(self):
        if self.env.limbProgress > 0.7:
            return True
        return False

class JacketLController(Controller):
    def __init__(self, env):
        obs_subset = [(0, 163)]
        name = "JacketL"
        #policyfilename = "experiment_2018_01_12_jacketL_dist_warm"
        policyfilename = "experiment_2018_01_13_jacketL_dist_warm_curriculum"
        #policyfilename = "experiment_2018_05_21_jacketL_restpose"
        #policyfilename = "experiment_2018_05_21_jacketL"
        #policyfilename = "experiment_2018_05_22_jacketL_warm"
        #policyfilename = "experiment_2018_05_22_jacketL_warm_rest"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_lsleeve")
        self.env.fingertip = np.array([0.0, -0.095, 0.0])

        # geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveLMidFeature, oneSided=True, side=0, normalSide=0)

        # feature/oracle setup
        self.env.focusFeature = self.env.sleeveLMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 12  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveLSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 21

        a=0

    def update(self):
        #self.env._reset()
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesL,offset=np.array([0,-0.085,0])), feature=self.env.sleeveLSeamFeature)
        a=0

    def transition(self):
        #print(self.env.limbProgress)
        if self.env.limbProgress > 0.7:
            return True
        if self.env.stepsSinceControlSwitch > 150:
            self.env.successRecord.append((False, self.env.numSteps, 2))
            self.env._reset()
        return False

class PhaseInterpolate1Controller(Controller):
    def __init__(self, env):
        obs_subset = [(0, 132), (141, 40)]
        name = "PhaseInterpolate"
        #policyfilename = "experiment_2018_01_11_phaseinterpolatejacket"
        #policyfilename = "experiment_2018_01_13_phaseinterpolatejacket_clothplace_warm"
        policyfilename = "experiment_2018_05_21_jackettransition"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_transition")

        self.env.fingertip = np.array([0.0, -0.085, 0.0])
        self.env.handleNode.clearHandles()
        self.env.rightTarget = np.array([0.25825808, 0.52266715, 0.29517964])
        self.env.leftTarget = np.array([0.16762201, -0.5902483, 0.05456982])
        self.env.restPose = np.array(
            [-0.40701562, 0.09377379, 0.72335076, -0.09763787, 0.24877598, 1.52214491, -3.4660477, -1.70725895,
             1.60078166, -0.49294963, 0.08305921, -0.25002647, 0.23895835, 0.20541631, 1.07185287, 0.80736386,
             1.52424401, -0.52905266, -0.52166824, 0.22250483, 0.23721241, -0.51535106])

        #self.env.rightTarget = np.array([ 0.24299472,  0.51030614,  0.29895259])
        #self.env.leftTarget = np.array([ 0.17159357, -0.57064675,  0.05449197])

    def update(self):
        #self.env._reset()
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)
        efL = self.env.robot_skeleton.bodynodes[12].to_world(self.env.fingertip)
        distR = np.linalg.norm(efR-self.env.rightTarget)
        distL = np.linalg.norm(efL-self.env.leftTarget)
        distP = np.linalg.norm(self.env.restPose - self.env.robot_skeleton.q)
        #print("distR " + str(distR) + " | distR " + str(distL) + " | distP " + str(distP))
        if distR < 0.05 and distL < 0.05 and distP < 2.0:
            return True
        if self.env.stepsSinceControlSwitch > 100:
            self.env.successRecord.append((False, self.env.numSteps, 1))
            self.env._reset()
        return False

class PhaseInterpolate2Controller(Controller):
    def __init__(self, env):
        obs_subset = [(0, 132), (141, 40)]
        name = "PhaseInterpolate2"
        policyfilename = "experiment_2018_05_21_jackettransition" #TODO
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_transition2")

        self.env.fingertip = np.array([0.0, -0.085, 0.0])
        self.env.restPose = np.zeros(len(self.env.robot_skeleton.q))
        self.env.restPose[8] = 0.21
        self.env.restPose[16] = 0.21
        #self.env.rightTarget = np.array([ 0.24299472,  0.51030614,  0.29895259])
        #self.env.leftTarget = np.array([ 0.17159357, -0.57064675,  0.05449197])

    def update(self):
        #self.env._reset()
        a=0

    def transition(self):
        return False

class SPDController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, policyfilename, name, obs_subset)
        self.phase = 0

        self.h = 0.002
        self.h = 0.01
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([5.0] * (ndofs))
        self.Kd = np.diagflat([0.2] * (ndofs))
        self.Kp[0][0] *= 5.0
        self.Kp[1][1] *= 5.0
        self.Kd[0][0] *= 100.0
        self.Kd[1][1] *= 100.0
        '''
        for i in range(ndofs):
            if i ==9 or i==10 or i==17 or i==18:
                self.Kd[i][i] *= 0.01
                self.Kp[i][i] *= 0.01
        '''

        #print(self.Kp)
        self.preoffset = 0.0

    def setup(self):
        self.phase = 0
        #self.env.saveState(name="enter_seq_final")
        self.env.frameskip = 1
        self.env.dart_world.set_time_step(0.01)
        self.env.robot_skeleton.set_velocities(self.env.robot_skeleton.dq*0.2) #1/5 of current velocities?
        self.env.SPDTorqueLimits = True
        #reset the target
        #cur_q = np.array(self.skel.q)
        #self.env.loadCharacterState(filename="characterState_regrip")
        self.target = np.array([ 0., 0., 0., -0.25, 0., 0., 0., 0., 1.0, 0., 0., -0.25, 0., 0., 0., 0., 1.0, 0., 0., 0., 0., 0.])
        self.target = np.array([ 0., 0., 0., -0.25, 0., 0., -1., 0., 1.0, 0., 0., -0.25, 0., 0., -1., 0., 1.0, 0., 0., 0., 0., 0.])
        self.env.restPose = np.array(self.target)
        #self.target = np.array(self.skel.q)
        #self.env.robot_skeleton.set_positions(cur_q)

        #clear the handles
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
            self.env.handleNode = None

        #self.skel.joint(6).set_damping_coefficient(0, 5)
        #self.skel.joint(6).set_damping_coefficient(1, 5)
        #self.skel.joint(11).set_damping_coefficient(0, 5)
        #self.skel.joint(11).set_damping_coefficient(1, 5)

        a=0

    def update(self):
        #if self.env.handle_node is not None:
        #    self.env.handle_node.clearHandles();
        #    self.env.handle_node = None
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.skel.q - self.env.restPose)
        #print(pDist)
        if pDist < 0.4 and self.phase == 0:
            self.target = np.array([ 0., 0., 0., 0., 0., 0., 0., 0., 0.21, 0., 0., 0., 0., 0., 0., 0., 0.21, 0., 0., 0., 0., 0.])
            self.env.restPose = np.array(self.target)
            self.phase = 1
            #return True
        elif pDist < 0.2 and self.phase == 1:
            self.env.successRecord.append((True, self.env.numSteps, 3))
            self.env._reset()
        return False

    def query(self, obs):
        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        b = -skel.c + p + d + skel.constraint_forces()
        A = skel.M + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        tau = p - self.Kd.dot(skel.dq + x * self.h)
        return tau

class DartClothUpperBodyDataDrivenClothJacketMasterEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True
        dt = 0.002
        frameskip = 5
        self.renderDetails = False

        #observation terms
        self.featureInObs   = True  # if true, feature centroid location and displacement from ef are observed
        self.oracleInObs    = True  # if true, oracle vector is in obs
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True  #if true, rewarded for 0 torso angle from vertical
        self.elbowFlairReward           = False
        self.limbProgressReward         = True  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementReward   = True  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward           = True  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenalty         = True
        self.restPoseReward             = False
        self.rightTargetReward          = True
        self.leftTargetReward           = True

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.collarTermination  = False  #if true, rollout terminates when collar is off the head/neck
        self.sleeveEndTerm      = True  #if true, terminate the rollout if the arm enters the end of sleeve feature before the beginning (backwards dressing)
        self.elbowFirstTerm     = True #if true, terminate when any limb enters the feature before the hand

        #other variables
        self.state_save_directory = "saved_control_states_jacket/"
        self.prevTau = None
        self.elbowFlairNode = 10
        self.maxDeformation = 30.0
        self.restPose = None
        self.prevOracle = np.zeros(3)
        self.prevAvgGeodesic = None
        self.localRightEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.limbProgress = 0
        self.previousDeformationReward = 0
        self.handFirst = False #once the hand enters the feature, switches to true
        self.fingertip = np.array([0.0, -0.085, 0.0])

        self.handleNode = None
        self.updateHandleNodeFrom = 12  # left fingers

        #success tracking
        self.successRecord = [] #contains a list of tuples, one for each rollout: (success/failure, steps excecuted, furthest active controller)
        self.successTrackingFile = "jacket_success_tracking"

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 66
        if self.featureInObs:
            observation_size += 6
        if self.oracleInObs:
            observation_size += 3
        if self.contactIDInObs:
            observation_size += 22
        if self.rightTargetReward:
            observation_size += 9
        if self.leftTargetReward:
            observation_size += 9


        #observation format
        # (0,66) - pose
        # (66,66) - haptics
        # (132,6) - feature
        # (138,3) - oracle
        # (141,22) - contact IDs
        # (163, 9) - right target
        # (172, 9) - left target
        # 181 - total

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="jacketmedium.obj",
                                                          #clothMeshStateFile = "tshirt_regrip5.obj",
                                                          #clothMeshStateFile = "objFile_1starmin.obj",
                                                          clothScale=np.array([0.7,0.7,0.5]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          dt=dt,
                                                          frameskip=frameskip)

        #clothing features
        self.sleeveRVerts = [46, 697, 1196, 696, 830, 812, 811, 717, 716, 718, 968, 785, 1243, 783, 1308, 883, 990, 739, 740, 742, 1318, 902, 903, 919, 737, 1218, 736, 1217]
        self.sleeveRMidVerts = [1054, 1055, 1057, 1058, 1060, 1061, 1063, 1052, 1051, 1049, 1048, 1046, 1045, 1043, 1042, 1040, 1039, 734, 732, 733]
        self.sleeveREndVerts = [228, 1059, 229, 1062, 230, 1064, 227, 1053, 226, 1050, 225, 1047, 224, 1044, 223, 1041, 142, 735, 141, 1056]
        self.sleeveLVerts = [848, 648, 649, 652, 980, 1380, 861, 860, 862, 1369, 92, 1344, 940, 941, 953, 561, 559, 560, 788, 789, 814, 1261, 537, 1122, 535, 1121, 536, 1277, 831, 1278, 834, 1287]
        self.sleeveLMidVerts = [1033, 1013, 1014, 1016, 1035, 1017, 1018, 1020, 1031, 629, 630, 633, 1028, 1030, 1037, 1025, 1026, 1021, 1022, 1414, 1024, 1422]
        self.sleeveLEndVerts = [1015, 216, 1036, 217, 1019, 218, 1032, 72, 632, 74, 222, 1038, 221, 1027, 219, 1023, 220, 1034, 215]

        self.targetGripVertices = [727, 138, 728, 1361, 730, 961, 1213, 137, 724, 1212, 726, 960, 964, 729, 155, 772]
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.sleeveREndFeature = ClothFeature(verts=self.sleeveREndVerts, clothScene=self.clothScene)
        self.sleeveRMidFeature = ClothFeature(verts=self.sleeveRMidVerts, clothScene=self.clothScene)
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)

        # variables for returning feature obs
        self.focusFeature = None  # if set, this feature centroid is used to get the "feature" obs
        self.focusFeatureNode = None  # if set, this body node is used to fill feature displacement obs
        self.progressFeature = None  # if set, this feature is used to fill oracle normal and check arm progress
        self.contactSensorIX = None  # if set, used to compute oracle

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        # controller initialzation
        self.controllers = [
            JacketRController(self),
            PhaseInterpolate1Controller(self),
            JacketLController(self),
            #PhaseInterpolate2Controller(self)
            SPDController(self)
        ]
        self.currentController = 0
        self.stepsSinceControlSwitch = 0

        for i in range(len(self.robot_skeleton.dofs)):
            self.robot_skeleton.dofs[i].set_damping_coefficient(3.0)

        #self.loadCharacterState(filename="characterState_1starmin")

    def _getFile(self):
        return __file__

    def saveState(self, name="unnamed"):
        fname = self.state_save_directory + name
        print(fname)
        count = 0
        objfname_ix = fname + "%05d" % count
        charfname_ix = fname + "_char%05d" % count
        while os.path.isfile(objfname_ix + ".obj"):
            count += 1
            objfname_ix = fname + "%05d" % count
            charfname_ix = fname + "_char%05d" % count
        print(objfname_ix)
        self.saveObjState(filename=objfname_ix)
        self.saveCharacterState(filename=charfname_ix)

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #update features
        if self.sleeveRSeamFeature is not None:
            self.sleeveRSeamFeature.fitPlane()
        if self.sleeveREndFeature is not None:
            self.sleeveREndFeature.fitPlane()
        if self.sleeveRMidFeature is not None:
            self.sleeveRMidFeature.fitPlane()
        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.fitPlane()
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.fitPlane()
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.fitPlane()

        self.limbProgress = -1  # reset this in controller

        self.additionalAction = np.zeros(22)  # reset control
        # update controller specific variables and produce
        if self.currentController is not None:
            self.controllers[self.currentController].update()
            if self.controllers[self.currentController].transition():
                changed = self.currentController
                self.currentController = min(len(self.controllers) - 1, self.currentController + 1)
                changed = (changed != self.currentController)
                if changed:
                    self.controllers[self.currentController].setup()
                    self.controllers[self.currentController].update()
                    self.stepsSinceControlSwitch = 0
            obs = self._get_obs()
            self.additionalAction = self.controllers[self.currentController].query(obs)

        self.stepsSinceControlSwitch += 1

        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500
        '''elif self.sleeveEndTerm and self.limbProgress <= 0 and self.simulateCloth:
            limbInsertionError = pyutils.limbFeatureProgress(
                limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR,
                                                  offset=np.array([0, -0.095, 0])), feature=self.sleeveREndFeature)
            if limbInsertionError > 0:
                return True, -500
        elif self.elbowFirstTerm and self.simulateCloth and not self.handFirst:
            if self.limbProgress > 0 and self.limbProgress < 0.14:
                self.handFirst = True
            else:
                limbInsertionError = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR[:3]),
                    feature=self.sleeveRFeature)
                if limbInsertionError > 0:
                    return True, -500'''
        return False, 0

    def computeReward(self, tau):
        # compute and return reward at the current state
        # unnecessary for master control env... not meant for training
        return 0

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.prevTauObs:
            obs = np.concatenate([obs, self.prevTau])

        if self.hapticsInObs:
            f = None
            if self.simulateCloth and self.hapticsAware:
                f = self.clothScene.getHapticSensorObs()#get force from simulation
            else:
                f = np.zeros(f_size)
            obs = np.concatenate([obs, f]).ravel()

        # feature
        if self.focusFeature is None or self.focusFeatureNode is None:
            obs = np.concatenate([obs, np.zeros(6)]).ravel()
        else:
            centroid = self.focusFeature.plane.org
            ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
            disp = centroid - ef
            obs = np.concatenate([obs, centroid, disp]).ravel()

        # oracle
        oracle = np.zeros(3)
        if self.reset_number == 0:
            a = 0  # nothing
        elif self.limbProgress > 0:
            oracle = self.progressFeature.plane.normal
        else:
            minGeoVix = None
            minContactGeodesic = None
            _side = None
            if self.contactSensorIX is not None:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=self.contactSensorIX,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                     returnOnlyGeo=False)
            if minGeoVix is None:
                if self.focusFeatureNode is not None:
                    # oracle points to the garment when ef not in contact
                    ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
                    # closeVert = self.clothScene.getCloseVertex(p=efR)
                    # target = self.clothScene.getVertexPos(vid=closeVert)

                    centroid = self.focusFeature.plane.org

                    target = centroid
                    vec = target - ef
                    oracle = vec / np.linalg.norm(vec)
            else:
                vixSide = 0
                if _side:
                    vixSide = 1
                if minGeoVix >= 0:
                    oracle = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
        self.prevOracle = oracle
        obs = np.concatenate([obs, oracle]).ravel()

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            obs = np.concatenate([obs, self.rightTarget, efR, self.rightTarget-efR]).ravel()

        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            obs = np.concatenate([obs, self.leftTarget, efL, self.leftTarget-efL]).ravel()


        return obs

    def additionalResets(self):
        if self.reset_number > 6:
            exit()
        #do any additional resetting here
        print("Success Record so far: " + str(self.successRecord))
        print("numTrials = " + str(len(self.successRecord)))
        if len(self.successRecord) > 0:
            successfulCount = 0
            failCounter = np.zeros(4)
            for r in self.successRecord:
                if r[0]:
                    successfulCount+=1
                else:
                    failCounter[r[2]] += 1
            print("Success Rate: " + str(successfulCount/len(self.successRecord)))
            print("Failure Counter: " + str(failCounter))
            self.saveSuccessData()

        self.dart_world.skeletons[0].set_positions(np.ones(6)*-1)
        self.dart_world.set_time_step(0.002)
        self.frame_skip = 5
        self.SPDTorqueLimits = False
        self.currentController = 0
        self.handFirst = False
        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0.125, -0.27, -0.6]))
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos[16] = 1.9
        '''qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        '''
        self.set_state(qpos, qvel)
        #self.loadCharacterState(filename="characterState_1starmin")
        self.restPose = qpos

        # update features
        if self.sleeveRSeamFeature is not None:
            self.sleeveRSeamFeature.fitPlane(normhint=np.array([1.0, 0, 0]))
        if self.sleeveREndFeature is not None:
            self.sleeveREndFeature.fitPlane(normhint=np.array([-1.0,0,0]))
        if self.sleeveRMidFeature is not None:
            self.sleeveRMidFeature.fitPlane(normhint=np.array([-1.0,0,0]))
        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.fitPlane(normhint=np.array([1.0, 0, 0]))
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.fitPlane(normhint=np.array([1.0,0,0]))
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.fitPlane(normhint=np.array([-1.0,0,0]))

        # confirm relative normals
        # ensure relative correctness of normals
        CP2_CP1R = self.sleeveREndFeature.plane.org - self.sleeveRMidFeature.plane.org
        CP2_CP0R = self.sleeveRSeamFeature.plane.org - self.sleeveRMidFeature.plane.org

        # if CP2 normal is not facing the sleeve end invert it
        if CP2_CP1R.dot(self.sleeveRMidFeature.plane.normal) < 0:
            self.sleeveRMidFeature.plane.normal *= -1.0

        # if CP1 normal is facing the sleeve middle invert it
        if CP2_CP1R.dot(self.sleeveREndFeature.plane.normal) < 0:
            self.sleeveREndFeature.plane.normal *= -1.0

        # if CP0 normal is not facing sleeve middle invert it
        if CP2_CP0R.dot(self.sleeveRSeamFeature.plane.normal) > 0:
            self.sleeveRSeamFeature.plane.normal *= -1.0

        CP2_CP1L = self.sleeveLEndFeature.plane.org - self.sleeveLMidFeature.plane.org
        CP2_CP0L = self.sleeveLSeamFeature.plane.org - self.sleeveLMidFeature.plane.org

        # if CP2 normal is not facing the sleeve end invert it
        if CP2_CP1L.dot(self.sleeveLMidFeature.plane.normal) < 0:
            self.sleeveLMidFeature.plane.normal *= -1.0

        # if CP1 normal is facing the sleeve middle invert it
        if CP2_CP1L.dot(self.sleeveLEndFeature.plane.normal) < 0:
            self.sleeveLEndFeature.plane.normal *= -1.0

        # if CP0 normal is not facing sleeve middle invert it
        if CP2_CP0L.dot(self.sleeveLSeamFeature.plane.normal) > 0:
            self.sleeveLSeamFeature.plane.normal *= -1.0

        self.controllers[self.currentController].setup()

        a=0

    def extraRenderFunction(self):

        renderUtils.setColor([0.7, 0.4, 0.2])
        renderUtils.drawSphere(pos=self.robot_skeleton.bodynodes[14].to_world(np.array([0.0, 0.05, -0.09])), rad=0.05)

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        if self.renderDetails:
            renderUtils.setColor(color=[0.0, 0.0, 0])
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(0, 0, 0)
            GL.glVertex3d(-1, 0, 0)
            GL.glEnd()
            renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[7].to_world(np.array([0.0, -0.075, 0.0])),
                                              self.prevOracle + self.robot_skeleton.bodynodes[7].to_world(
                                                  np.array([0.0, -0.075, 0.0])),
                                              ])
            renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[12].to_world(np.array([0.0, -0.075, 0.0])),
                                              self.prevOracle + self.robot_skeleton.bodynodes[12].to_world(
                                                  np.array([0.0, -0.075, 0.0])),
                                              ])

            if self.sleeveRSeamFeature is not None:
                self.sleeveRSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveREndFeature is not None:
                self.sleeveREndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveRMidFeature is not None:
                self.sleeveRMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveLSeamFeature is not None:
                self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveLEndFeature is not None:
                self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveLMidFeature is not None:
                self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)

            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            renderUtils.setColor(color=[1.0, 0, 0])
            renderUtils.drawSphere(pos=self.rightTarget, rad=0.02)
            renderUtils.drawLineStrip(points=[self.rightTarget, efR])

            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            renderUtils.drawSphere(pos=self.leftTarget, rad=0.02)
            renderUtils.drawLineStrip(points=[self.leftTarget, efL])

        # render geodesic
        '''
        for v in range(self.clothScene.getNumVertices()):
            side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
            side0geo = self.separatedMesh.nodes[v].geodesic

            pos = self.clothScene.getVertexPos(vid=v)
            norm = self.clothScene.getVertNormal(vid=v)
            renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side0geo))
            renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
            renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side1geo))
            renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)
        '''

        textHeight = 15
        textLines = 2

        if self.renderUI and self.renderDetails:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
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
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])

            if self.stepsSinceControlSwitch < 50 and len(self.controllers) > 0:
                label = self.controllers[self.currentController].name
                self.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60,
                                         text="Active Controller = " + str(label), color=(0., 0, 0))

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -2.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
        self.track_skeleton_id = 0
        if not self.renderDARTWorld:
            self.viewer.renderWorld = False
        self.clothScene.renderCollisionCaps = True
        self.clothScene.renderCollisionSpheres = True

    def saveSuccessData(self):
        f = open(self.successTrackingFile, 'w')

        successfulCount = 0
        failCounter = np.zeros(4)
        if len(self.successRecord) > 0:
            for r in self.successRecord:
                if r[0]:
                    successfulCount += 1
                else:
                    failCounter[r[2]] += 1

        f.write("Data Collected: " + str(datetime.datetime.today().strftime('%Y-%m-%d')) + "\n")
        f.write("Number of Trials: " + str(len(self.successRecord)) + "\n")
        f.write("Success rate: " + str(successfulCount / len(self.successRecord)) + "\n")
        f.write("Failure Counter: " + str(failCounter) + "\n")

        for r in self.successRecord:
            f.write(str(r) + "\n")

        f.close()