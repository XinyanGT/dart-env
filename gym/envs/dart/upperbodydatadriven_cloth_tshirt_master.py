# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
import datetime
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.pyutils import LERP
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

from scipy.optimize import minimize

import pickle

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

class DropGripController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,154), (163,9), (181,6)]
        name = "DropGrip"
        #policyfilename = "experiment_2018_01_02_dropgrip2"
        #policyfilename = "experiment_2018_02_16_dropgrip_alignspecific" #1st iteration
        policyfilename = "experiment_2018_04_18_dropgrip_stablehead" #2nd iteration

        Controller.__init__(self, env, policyfilename, name, obs_subset)
        self.prevOrientationError = 0
        self.prevPositionError = 0

    def setup(self):
        self.env.fingertip = np.array([0, -0.085, 0])
        self.env.renderContainmentTriangle  = False
        self.env.renderGeodesic             = False
        self.env.renderOracle               = False
        self.env.renderRightTarget          = False
        self.env.renderLeftTarget           = True
        self.env.renderRestPose             = False
        a=0

    def update(self):
        self.env.leftTarget = pyutils.getVertCentroid(verts=self.env.targetGripVerticesL, clothscene=self.env.clothScene) + pyutils.getVertAvgNorm(verts=self.env.targetGripVerticesL, clothscene=self.env.clothScene)*0.03
        #self.env.leftTarget = self.env.gripFeatureL.plane.org + self.env.gripFeatureL.plane.normal * 0.03
        self.env.leftOrientationTarget = self.env.gripFeatureL.plane.toWorld(self.env.localLeftOrientationTarget)

        wLFingertip2 = self.env.robot_skeleton.bodynodes[12].to_world(self.env.fingertip)
        lDist = np.linalg.norm(self.env.leftTarget - wLFingertip2)
        self.prevPositionError = lDist

        wLZero = self.env.robot_skeleton.bodynodes[12].to_world(np.zeros(3))
        efLV = wLFingertip2 - wLZero
        efLDir = efLV / np.linalg.norm(efLV)
        reward_leftOrientationTarget = -(1 - self.env.leftOrientationTarget.dot(efLDir)) / 2.0  # [-1,0]
        self.prevOrientationError = -reward_leftOrientationTarget

    def transition(self):
        efL = self.env.robot_skeleton.bodynodes[12].to_world(np.array([0,-0.065,0]))
        dist = np.linalg.norm(self.env.leftTarget - efL)
        #print("dist: " + str(dist))

        #if dist < 0.035:
        if self.prevPositionError < 0.05:
            if self.prevOrientationError < 0.15: #slightly wider acceptance than training ...
                return True
        if self.env.stepsSinceControlSwitch > 100:
            self.env.successRecord.append((False, self.env.numSteps, 0))
            self.env._reset()
        return False

class RightTuckController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,154)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_toR3_cont"
        #policyfilename = "experiment_2018_01_08_distribution_rightTuck_warm"
        #policyfilename = "experiment_2018_03_05_tuckR_triangle_forward" #sequence iteration 1
        policyfilename = "experiment_2018_04_18_rtuck"
        name="Right Tuck"
        Controller.__init__(self, env, policyfilename, name, obs_subset)
        self.framesContained = 0

    def setup(self):
        if self.env.recordingActionTrajectory:
            self.env.actionTrajectoryGripEvents.append((self.env.numSteps, 1))

        #self.env.saveState(name="enter_seq_rtuck")
        self.env.fingertip = np.array([0, -0.085, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesL)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()
        #setup ef targets
        self.rightTarget = np.array([-0.09569568,  0.14657749, -0.17376665])
        self.leftTarget = np.array([-0.07484753,  0.21876009, -0.26294776])
        self.env.contactSensorIX = None
        self.env.focusGripFeature = self.env.gripFeatureL
        self.env.renderContainmentTriangle = True
        self.env.renderGeodesic = False
        self.env.renderOracle = False
        self.env.renderRightTarget = False
        self.env.renderLeftTarget = False
        self.env.renderRestPose = True
        self.framesContained = 0

        self.env.perturbationNode = 5
        self.env.perturbationActive = True
        self.env.curPerDir = 0
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()

        #print(self.env.stepsSinceControlSwitch)
        if self.env.testingPerturbation:
            if self.env.stepsSinceControlSwitch < self.env.perturbationSteps and self.env.reset_number > 0:
                self.env.perturbationActive = True
                self.env.robot_skeleton.bodynodes[self.env.perturbationNode].add_ext_force(self.env.perturbationDirections[0]*self.env.perturbationMagnitude, self.env.perturbationOffset)
            else:
                #print("here")
                self.env.perturbationActive = False
        else:
            self.env.perturbationActive = False
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)
        efL = self.env.robot_skeleton.bodynodes[12].to_world(self.env.fingertip)
        shoulderR0 = self.env.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
        shoulderR1 = self.env.robot_skeleton.bodynodes[3].to_world(np.zeros(3))
        shoulderR = (shoulderR0+shoulderR1)/2.0
        #TODO: transition criteria

        lines = [
            [self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip),
             self.env.robot_skeleton.bodynodes[6].to_world(np.zeros(3))],
            [self.env.robot_skeleton.bodynodes[6].to_world(np.zeros(3)),
             self.env.robot_skeleton.bodynodes[5].to_world(np.zeros(3))]
        ]
        intersect = False
        aligned = False
        for l in lines:
            hit, dist, pos, aligned = pyutils.triangleLineSegmentIntersection(self.env.previousContainmentTriangle[0],
                                                                              self.env.previousContainmentTriangle[1],
                                                                              self.env.previousContainmentTriangle[2],
                                                                              l0=l[0], l1=l[1])
            if hit:
                # print("hit: " + str(dist) + " | " + str(pos) + " | " + str(aligned))
                intersect = True
                aligned = aligned
                break
        if intersect:
            if aligned:
                self.framesContained += 1

        if self.framesContained > 15:
            return True

        if self.env.stepsSinceControlSwitch > 100:
            self.env.successRecord.append((False, self.env.numSteps, 1))
            self.env._reset()

        '''
        v0 = efR-shoulderR
        v1 = efL-shoulderR
        v0Len = np.linalg.norm(v0)
        v1Len = np.linalg.norm(v1)
        v0 = v0/v0Len
        v1 = v1/v1Len
        CIDS = self.env.clothScene.getHapticSensorContactIDs()
        print("v0.dot(v1): " + str(v0.dot(v1)) + " | v1Len > v2Len: " + str(v1Len > v0Len) + " | CID: " + str(CIDS[12]))
        if v0.dot(v1) > 0.9:
            if v1Len > v0Len+0.05:
                if CIDS[12] >= 0.8:
                    return True
        '''
        return False

class LeftTuckController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,154)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_toL_cont"
        #policyfilename = "experiment_2018_03_26_ltuck_wide"
        #policyfilename = "experiment_2018_04_11_ltuck_seq_velwarm" #seq iter 1

        #policyfilename = "experiment_2018_04_20_ltuck"
        #policyfilename = "experiment_2018_04_21_ltuck" #Works well but maybe too shallow?
        #policyfilename = "experiment_2018_05_04_ltuck_403"

        policyfilename = "experiment_2018_05_22_tuckL"
        policyfilename = "experiment_2018_05_23_ltuck_warm"


        name="Left Tuck"
        Controller.__init__(self, env, policyfilename, name, obs_subset)
        self.framesContained = 0

    def setup(self):
        if self.env.recordingActionTrajectory:
            self.env.actionTrajectoryGripEvents.append((self.env.numSteps, 4))

        #self.env.saveState(name="enter_seq3_ltuck")
        self.framesContained = 0
        self.env.contactSensorIX = None
        #self.env.fingertip = np.array([0, -0.085, 0])
        self.fingertip = np.array([0, -0.075, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 7
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesR)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()
        #setup ef targets
        #self.rightTarget = np.array([ 0.07219914,  0.02462782, -0.37559271])
        #self.leftTarget = np.array([ 0.06795917, -0.02272099, -0.12309984])
        self.env.focusGripFeature = self.env.gripFeatureR
        self.env.renderContainmentTriangle = True
        self.env.renderGeodesic = False
        self.env.renderOracle = False
        self.env.renderRightTarget = False
        self.env.renderLeftTarget = False
        #self.env.contactSensorIX = 21

        self.env.perturbationNode = 10
        self.env.perturbationActive = True
        self.env.curPerDir = 1
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()

        if self.env.testingPerturbation:
            if self.env.stepsSinceControlSwitch < self.env.perturbationSteps and self.env.reset_number > 0:
                self.env.perturbationActive = True
                self.env.robot_skeleton.bodynodes[self.env.perturbationNode].add_ext_force(self.env.perturbationDirections[1]*self.env.perturbationMagnitude, self.env.perturbationOffset)
            else:
                #print("here")
                self.env.perturbationActive = False
        else:
            self.env.perturbationActive = False
        #self.env._reset()
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)
        efL = self.env.robot_skeleton.bodynodes[12].to_world(self.env.fingertip)
        shoulderR0 = self.env.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
        shoulderR1 = self.env.robot_skeleton.bodynodes[3].to_world(np.zeros(3))
        shoulderR = (shoulderR0+shoulderR1)/2.0
        extended_fingertip = np.array([0, -0.09, 0])
        #TODO: transition criteria

        lines = [
            [self.env.robot_skeleton.bodynodes[12].to_world(extended_fingertip),
             self.env.robot_skeleton.bodynodes[11].to_world(np.zeros(3))],
            [self.env.robot_skeleton.bodynodes[11].to_world(np.zeros(3)),
             self.env.robot_skeleton.bodynodes[10].to_world(np.zeros(3))]
        ]
        intersect = False
        aligned = False
        for l in lines:
            hit, dist, pos, aligned = pyutils.triangleLineSegmentIntersection(self.env.previousContainmentTriangle[0],
                                                                              self.env.previousContainmentTriangle[1],
                                                                              self.env.previousContainmentTriangle[2],
                                                                              l0=l[0], l1=l[1])
            if hit:
                # print("hit: " + str(dist) + " | " + str(pos) + " | " + str(aligned))
                intersect = True
                aligned = aligned
                break
        if intersect:
            if aligned:
                self.framesContained += 1

        CID = self.env.clothScene.getHapticSensorContactIDs()[21]
        # -1.0 for full outside, 1.0 for full inside

        if self.framesContained > 20 and CID == 1.0:
            return True

        if self.env.stepsSinceControlSwitch > 100:
            self.env.successRecord.append((False, self.env.numSteps, 4))
            self.env._reset()

        return False

class MatchGripController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,163)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_matchgrip3_cont"
        #policyfilename = "experiment_2018_01_14_matchgrip_dist_lowpose"
        #policyfilename = "experiment_2018_03_23_matchgrip_reducedwide"
        #policyfilename = "experiment_2018_04_09_match_seqwarm"
        #policyfilename = "experiment_2018_04_10_match_seq_veltask" #seq iter 1

        #policyfilename = "experiment_2018_04_20_matchgrip_warm" #seq iter 2
        policyfilename = "experiment_2018_05_17_matchgrip" #seq iter 2: elbow down

        name="Match Grip"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_match")

        self.env.fingertip = np.array([0, -0.09, 0])
        # setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesL)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        self.env.renderContainmentTriangle = False
        self.env.renderGeodesic = False
        self.env.renderOracle = False
        self.env.renderRightTarget = True
        self.env.renderLeftTarget = False
        a=0

    def update(self):
        self.env.rightTarget = pyutils.getVertCentroid(verts=self.env.targetGripVerticesR, clothscene=self.env.clothScene) + pyutils.getVertAvgNorm(verts=self.env.targetGripVerticesR, clothscene=self.env.clothScene)*0.03
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)

        shoulderR = self.env.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
        shoulderL = self.env.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
        elbow = self.env.robot_skeleton.bodynodes[5].to_world(np.zeros(3))
        shoulderR_torso = self.env.robot_skeleton.bodynodes[1].to_local(shoulderR)
        shoulderL_torso = self.env.robot_skeleton.bodynodes[1].to_local(shoulderL)
        elbow_torso = self.env.robot_skeleton.bodynodes[1].to_local(elbow)

        elevation = 0
        if elbow_torso[1] > shoulderL_torso[1] or elbow[1] > shoulderR_torso[1]:
            elevation = max(elbow_torso[1] - shoulderL_torso[1], elbow_torso[1] - shoulderR_torso[1])
        #print(elevation)

        if np.linalg.norm(efR-self.env.rightTarget) < 0.05 and elevation < 0.1:
            return True

        if self.env.stepsSinceControlSwitch > 125:
            self.env.successRecord.append((False, self.env.numSteps, 3))
            self.env._reset()

        return False
        a=0

class MatchGripTransitionController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,163)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_matchgrip3_cont"
        #policyfilename = "experiment_2018_05_20_matchgrip_pose2"
        #policyfilename = "experiment_2018_05_21_match_warm"
        policyfilename = "experiment_2018_05_22_match_warm_rest"

        name="Match Grip Transition"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_match")

        self.env.fingertip = np.array([0, -0.09, 0])
        # setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesL)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        self.env.renderContainmentTriangle = False
        self.env.renderGeodesic = False
        self.env.renderOracle = False
        self.env.renderRightTarget = True
        self.env.renderLeftTarget = False

        self.restPose = np.array(
            [-0.210940942604, -0.0602436241858, 0.785540563981, 0.132571030392, -0.25, -0.580739841458, -0.803858324899,
             -1.472, 1.27301394196, -0.295286198863, 0.611311245326, 0.245333463513, 0.225511476131, 1.20063053643,
             -0.0501794921426, 1.19122509695, 1.97519722198, -0.573360432341, 0.321222466527, 0.580323061076,
             -0.422112755785, -0.997819593165])

        a=0

    def update(self):
        self.env.rightTarget = pyutils.getVertCentroid(verts=self.env.targetGripVerticesR, clothscene=self.env.clothScene) + pyutils.getVertAvgNorm(verts=self.env.targetGripVerticesR, clothscene=self.env.clothScene)*0.03
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)

        shoulderR = self.env.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
        shoulderL = self.env.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
        elbow = self.env.robot_skeleton.bodynodes[5].to_world(np.zeros(3))
        shoulderR_torso = self.env.robot_skeleton.bodynodes[1].to_local(shoulderR)
        shoulderL_torso = self.env.robot_skeleton.bodynodes[1].to_local(shoulderL)
        elbow_torso = self.env.robot_skeleton.bodynodes[1].to_local(elbow)

        elevation = 0
        if elbow_torso[1] > shoulderL_torso[1] or elbow[1] > shoulderR_torso[1]:
            elevation = max(elbow_torso[1] - shoulderL_torso[1], elbow_torso[1] - shoulderR_torso[1])
        #print(elevation)
        #print(np.linalg.norm(efR-self.env.rightTarget))
        if np.linalg.norm(efR-self.env.rightTarget) < 0.05 and elevation < 0.1:
            return True

        if self.env.stepsSinceControlSwitch > 125:
            self.env.successRecord.append((False, self.env.numSteps, 3))
            self.env._reset()

        return False
        a=0

class RightSleeveController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,132), (172, 9), (132, 22)]
        #policyfilename = "experiment_2017_12_12_1sdSleeve_progressfocus_cont2"
        #policyfilename = "experiment_2018_01_09_tshirtR_dist_warm"
        #policyfilename = "experiment_2018_03_19_sleeveR_narrow2wide_lowdef_trpo"
        #policyfilename = "experiment_2018_03_29_rsleeve_seq"
        #policyfilename = "experiment_2018_04_03_rsleeve_seq_highdefwarm" #*** seq iteration 1 controller
        #policyfilename = "experiment_2018_04_03_rsleeve_seq_highdef"

        #policyfilename = "experiment_2018_04_10_rsleeve_seq_highdef_velwarm"

        #seq iteration 2
        #policyfilename = "experiment_2018_04_19_rsleeve"
        policyfilename = "experiment_2018_04_19_rsleeve_warm"

        #policyfilename = "experiment_2018_05_25_tshirtR_ablationbaseline_cont" #evaluation sequencing


        name="Right Sleeve"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_rsleeve")
        self.env.fingertip = np.array([0, -0.08, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesL)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        #geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveRMidFeature, oneSided=True, side=0, normalSide=1)

        #check a relative geodesic value and invert the geodesic if necessary
        vn1 = self.env.separatedMesh.nodes[2492 + self.env.separatedMesh.numv * 1]
        vGeo1 = vn1.geodesic
        vn2 = self.env.separatedMesh.nodes[2668 + self.env.separatedMesh.numv * 1]
        vGeo2 = vn2.geodesic

        if(vGeo1 < vGeo2): #backwards
            print("inverting geodesic")
            self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveRMidFeature, oneSided=True, side=0, normalSide=0)


        print("geos: [" + str(vGeo1) +","+str(vGeo2)+"]")

        #feature/oracle setup
        self.env.focusFeature = self.env.sleeveRMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 7  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveRSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 12

        self.env.renderContainmentTriangle = False
        self.env.renderGeodesic = False
        self.env.renderOracle = True
        self.env.renderRightTarget = False
        self.env.renderLeftTarget = False

        self.env.perturbationActive = True
        self.env.curPerDir = 2
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        #limb progress
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesR,offset=np.array([0,-0.065,0])), feature=self.env.sleeveRSeamFeature)
        a=0

        if self.env.testingPerturbation:
            if self.env.stepsSinceControlSwitch < self.env.perturbationSteps and self.env.reset_number > 0:
                self.env.perturbationActive = True
                self.env.robot_skeleton.bodynodes[self.env.perturbationNode].add_ext_force(self.env.perturbationDirections[2]*self.env.perturbationMagnitude, self.env.perturbationOffset)
            else:
                #print("here")
                self.env.perturbationActive = False
        else:
            self.env.perturbationActive = False

    def transition(self):
        if self.env.limbProgress > 0.6:
            return True
        if self.env.stepsSinceControlSwitch > 100:
            self.env.successRecord.append((False, self.env.numSteps, 2))
            self.env._reset()
        return False

class LeftSleeveController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,132), (172, 9), (132, 22)]
        #policyfilename = "experiment_2017_12_12_1sdSleeve_progressfocus_cont2"
        #policyfilename = "experiment_2017_12_08_2ndSleeve_cont"
        #policyfilename = "experiment_2018_03_26_lsleeve_narrow"
        #policyfilename = "experiment_2018_03_27_lsleeve_widewarm_highdef"
        #policyfilename = "experiment_2018_04_13_lsleeve_seq_velwarm" #seq iter 1
        #policyfilename = "experiment_2018_04_15_lsleeve_seq_velwarm_cont"

        #policyfilename = "experiment_2018_04_23_lsleeve"
        #policyfilename = "experiment_2018_04_23_lsleeve_warm"

        #policyfilename = "experiment_2018_04_30_sleeveL_wide_highdef"
        #policyfilename = "experiment_2018_05_01_sleevel_widewarm"
        #policyfilename = "experiment_2018_05_06_lsleeve2_wide"
        #policyfilename = "experiment_2018_05_09_lsleeve2_wide_warmhighdef"

        policyfilename = "experiment_2018_05_23_lsleeve_warm"

        name="Left Sleeve"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        #self.env.saveState(name="enter_seq_3lsleeve")
        self.env.fingertip = np.array([0, -0.08, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 7
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVerticesR)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        #geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveLMidFeature, oneSided=True, side=0, normalSide=1)

        # check a relative geodesic value and invert the geodesic if necessary
        vn1 = self.env.separatedMesh.nodes[ 2358 + self.env.separatedMesh.numv * 1]
        vGeo1 = vn1.geodesic
        vn2 = self.env.separatedMesh.nodes[ 2230 + self.env.separatedMesh.numv * 1]
        vGeo2 = vn2.geodesic

        if (vGeo1 < vGeo2):  # backwards
            print("inverting geodesic")
            self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveLMidFeature, oneSided=True, side=0, normalSide=0)

        print("geos: [" + str(vGeo1) + "," + str(vGeo2) + "]")

        #feature/oracle setup
        self.env.focusFeature = self.env.sleeveLMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 12  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveLSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 21

        self.env.renderContainmentTriangle = False
        self.env.renderGeodesic = False
        self.env.renderOracle = True
        self.env.renderRightTarget = False
        self.env.renderLeftTarget = False

        self.env.perturbationActive = True
        self.env.curPerDir = 3
        a=0

    def update(self):
        #self.env._reset()
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        #limb progress
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesL,offset=np.array([0,-0.065,0])), feature=self.env.sleeveLSeamFeature)

        if self.env.testingPerturbation:
            if self.env.stepsSinceControlSwitch < self.env.perturbationSteps and self.env.reset_number > 0:
                self.env.perturbationActive = True
                self.env.robot_skeleton.bodynodes[self.env.perturbationNode].add_ext_force(self.env.perturbationDirections[3]*self.env.perturbationMagnitude, self.env.perturbationOffset)
            else:
                #print("here")
                self.env.perturbationActive = False
        else:
            self.env.perturbationActive = False
        a=0

    def transition(self):
        if self.env.limbProgress > 0.5:
            return True
        if self.env.stepsSinceControlSwitch > 200:
            self.env.successRecord.append((False, self.env.numSteps, 5))
            self.env._reset()
        return False

class FinalTransitionController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0, 154)]

        policyfilename = "experiment_2018_05_24_tshirt_final"

        name = "Final Transition"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        if self.env.recordingActionTrajectory:
            self.env.actionTrajectoryGripEvents.append((self.env.numSteps, 6))

        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.renderOracle = False
        self.env.restPose = np.array([ 0., 0., 0., 0., 0., 0., 0., 0., 0.21, 0., 0., 0., 0., 0., 0., 0., 0.21, 0., 0., 0., 0., 0.])

    def update(self):
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.env.robot_skeleton.q - self.env.restPose)
        if self.env.stepsSinceControlSwitch > 125:
            self.env.successRecord.append((False, self.env.numSteps, 6))
            self.env._reset()
        if pDist < 0.5:
            self.env.successRecord.append((True, self.env.numSteps, 6))
            self.env._reset()
        return False

class SPDController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target #overriden often
        self.endPose = np.array(target) #stable target pose for interp
        self.initialPose = np.array(target) #stable start pose for interp
        self.interpTime = 100
        self.steps = 0
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.002
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        p = 200
        d = p*self.h*8
        self.Kp = np.diagflat([p] * (ndofs))
        self.Kd = np.diagflat([d] * (ndofs))

        self.Kd[0][0] *= 8
        self.Kp[0][0] *= 32
        self.Kd[1][1] *= 8
        self.Kp[1][1] *= 32

        '''
        for i in range(ndofs):
            if i ==9 or i==10 or i==17 or i==18:
                self.Kd[i][i] *= 0.01
                self.Kp[i][i] *= 0.01
        '''

        #print(self.Kp)
        self.preoffset = 0.0

    def setup(self):
        #self.env.saveState(name="enter_seq_final")
        #self.env.frameskip = 1
        #reset the target
        self.steps = 0
        self.target = np.array([ 0., 0., 0., 0., 0., 0., 0., 0., 0.302, 0., 0., 0., 0., 0., 0., 0., 0.302, 0., 0., 0., 0., 0.])
        self.endPose = np.array(self.target)
        self.initialPose = np.array(self.env.robot_skeleton.q)
        #self.env.restPose = np.array(self.target)

        #clear the handles
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
            self.env.handleNode = None

        '''self.env.SPDTorqueLimits = False
        self.env.SPD = self
        self.env.SPDTarget = np.array(self.initialPose)#self.target
        self.env.SPDPerFrame = True'''

    def update(self):
        #if self.env.handle_node is not None:
        #    self.env.handle_node.clearHandles();
        #    self.env.handle_node = None
        #self.env._reset()
        '''self.steps += 1
        fraq = float(self.steps)/self.interpTime
        self.env.SPDTarget = self.endPose*fraq + self.initialPose*(1.0-fraq)
        self.target = np.array(self.env.SPDTarget)
        if self.steps > self.interpTime:
            self.env.SPDTarget = np.array(self.endPose)
            self.target = np.array(self.env.SPDTarget)
        self.env.restPose = np.array(self.env.SPDTarget)'''
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.skel.q - self.env.restPose)
        #print(pDist)
        #if pDist < 0.1:
        #    return True
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

class SPDIKController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPDIK"
        self.target = target
        self.efL_target = None
        self.efR_target = None
        self.efL_start = None
        self.efR_start = None
        self.referencePose = None
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.02
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([400.0] * (ndofs))
        self.Kd = np.diagflat([40.0] * (ndofs))
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        cur_q = np.array(self.skel.q)
        self.env.loadCharacterState(filename="characterState_regrip")
        self.env.restPose = np.array(self.skel.q)
        self.target = np.array(self.skel.q)
        self.referencePose = np.array(self.skel.q)
        self.efL_target = self.skel.bodynodes[12].to_world(self.env.fingertip)
        self.efR_target = self.skel.bodynodes[7].to_world(self.env.fingertip)
        self.env.robot_skeleton.set_positions(cur_q)
        self.efL_start = self.skel.bodynodes[12].to_world(self.env.fingertip)
        self.efR_start = self.skel.bodynodes[7].to_world(self.env.fingertip)
        a=0

    def f(self, x):
        self.skel.set_positions(x)

        lhsL = self.skel.bodynodes[12].to_world(self.env.fingertip)
        rhsL = self.efL_start + (self.efL_target - self.efL_start) * min(self.env.stepsSinceControlSwitch / 100.0, 1.0)
        self.env.leftTarget = rhsL

        lhsR = self.skel.bodynodes[7].to_world(self.env.fingertip)
        rhsR = self.efR_start + (self.efR_target - self.efR_start) * min(self.env.stepsSinceControlSwitch / 100.0, 1.0)
        self.env.rightTarget = rhsR

        return 0.5 * np.linalg.norm(lhsL - rhsL) ** 2 + 0.5 * np.linalg.norm(lhsR - rhsR) ** 2

    def g(self, x):
        self.skel.set_positions(x)

        lhsL = self.skel.bodynodes[12].to_world(self.env.fingertip)
        rhsL = self.efL_target
        JL = self.skel.bodynodes[12].linear_jacobian()
        gL = (lhsL - rhsL).dot(JL)

        lhsR = self.skel.bodynodes[7].to_world(self.env.fingertip)
        rhsR = self.efR_target
        JR = self.skel.bodynodes[7].linear_jacobian()
        gR = (lhsR - rhsR).dot(JR)

        return gR+gL

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()

        c_q = np.array(self.skel.q)
        #TODO: IK to set self.target
        res = minimize(self.f,
                       x0=self.skel.positions(),
                       jac=self.g,
                       method="SLSQP")

        self.target = np.array(self.skel.q)
        self.target[:2] = self.referencePose[:2]
        self.env.restPose = np.array(self.skel.q)
        self.skel.set_positions(c_q)
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.skel.q - self.env.restPose)
        #print(pDist)
        #if pDist < 0.1:
        #    return True
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
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class DartClothUpperBodyDataDrivenClothTshirtMasterEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True
        self.simpleUI = False
        self.renderDetails = False

        #other flags
        self.collarTermination = True  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 10 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.resetTime = 0
        self.save_state_on_control_switch = False #if true, the cloth and character state is saved when controllers are switched
        self.state_save_directory = "saved_control_states/"
        self.renderContainmentTriangle  = False
        self.renderGeodesic             = False
        self.renderOracle               = False
        self.renderLeftTarget           = False
        self.renderRightTarget          = False
        self.renderRestPose             = False

        #other variables
        self.simStepsInReset = 90  # 90
        self.initialPerturbationScale = 0.35  # 0.35 #if >0, accelerate cloth particles in random direction with this magnitude
        self.simStepsAfterPerturbation = 60  # 60
        self.restPose = None
        self.localRightEfShoulder1 = None
        self.localLeftEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.prevErrors = None #stores the errors taken from DART each iteration
        self.limbProgress = -1
        self.fingertip = np.array([0, -0.085, 0])
        self.previousContainmentTriangle = [np.zeros(3), np.zeros(3), np.zeros(3)]

        #perturbation force
        self.testingPerturbation = False
        self.perturbationDirections = []
        self.perturbationMagnitude = 100.0
        self.perturbationSteps = 25
        self.perturbationActive = False
        self.perturbationNode = 5
        self.perturbationOffset = np.zeros(3)
        self.curPerDir = 0

        self.actionTrajectoryTesting = False

        # success tracking
        self.successRecord = []  # contains a list of tuples, one for each rollout: (success/failure, steps excecuted, furthest active controller)
        self.successTrackingFile = "tshirt_success_tracking"

        self.localLeftOrientationTarget = np.array([-0.5, -1.0, -1.0])
        self.localLeftOrientationTarget /= np.linalg.norm(self.localLeftOrientationTarget)

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq #(0,66)
        observation_size += 66 #haptics                             #(66,66)
        observation_size += 22 #contact IDs                         #(132,22)
        observation_size += 9 #right target                         #(154,9)
        observation_size += 9 #left target                          #(163,9)
        observation_size += 6 #feature                              #(172,6)
        observation_size += 3 #oracle                               #(178,3)
        observation_size += 6 #left orientation target (dropgrip)   #(181,6)

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        # clothing features
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        #self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        self.targetGripVerticesL = [46, 437, 955, 1185, 47, 285, 711, 677, 48, 905, 1041, 49, 741, 889, 45]
        self.targetGripVerticesR = [905, 1041, 49, 435, 50, 570, 992, 1056, 51, 676, 283, 52, 489, 892, 362, 53]
        self.sleeveRVerts = [2580, 2495, 2508, 2586, 2518, 2560, 2621, 2529, 2559, 2593, 272, 2561, 2658, 2582, 2666, 2575, 2584, 2625, 2616, 2453, 2500, 2598, 2466]
        self.sleeveRMidVerts = [2556, 2646, 2641, 2574, 2478, 2647, 2650, 269, 2630, 2528, 2607, 2662, 2581, 2458, 2516, 2499, 2555, 2644, 2482, 2653, 2507, 2648, 2573, 2601, 2645]
        self.sleeveREndVerts = [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 10, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251]
        self.sleeveLVerts = [211, 2305, 2364, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2276, 2326, 2334, 2288, 2346, 2314, 2251, 2347, 2304, 2245, 2376, 2315]
        self.sleeveLMidVerts = [2379, 2357, 2293, 2272, 2253, 214, 2351, 2381, 2300, 2352, 2236, 2286, 2430, 2263, 2307, 2387, 2232, 2390, 2277, 2348, 2382, 2227, 2296, 2425]
        self.sleeveLEndVerts = [232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 9, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231]

        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene, b1spanverts=[889,1041], b2spanverts=[47,677])
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene, b1spanverts=[362,889], b2spanverts=[51,992])
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.sleeveRMidFeature = ClothFeature(verts=self.sleeveRMidVerts, clothScene=self.clothScene)
        self.sleeveREndFeature = ClothFeature(verts=self.sleeveREndVerts, clothScene=self.clothScene)
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)

        #variables for returning feature obs
        self.focusFeature = None        #if set, this feature centroid is used to get the "feature" obs
        self.focusFeatureNode = None    #if set, this body node is used to fill feature displacement obs
        self.progressFeature = None     #if set, this feature is used to fill oracle normal and check arm progress
        self.contactSensorIX = None     #if set, used to compute oracle
        self.focusGripFeature = None    #if set, this points to the currently gripped feature

        self.simulateCloth = clothSimulation
        self.handleNode = None
        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        #controller initialzation
        self.controllers = [
            #SPDController(self),
            DropGripController(self),
            RightTuckController(self),
            RightSleeveController(self),
            MatchGripTransitionController(self),
            #MatchGripController(self),
            #SPDController(self),
            #SPDIKController(self),
            #MatchGripController(self),
            LeftTuckController(self),
            LeftSleeveController(self),
            FinalTransitionController(self)
            #SPDController(self)
        ]
        self.currentController = None
        self.stepsSinceControlSwitch = 0

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
        #fingertip = np.array([0.0, -0.065, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame
        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVertices, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVertices, clothscene=self.clothScene)*0.03
        self.limbProgress = -1 #reset this in controller

        #update feature planes
        self.collarFeature.fitPlane()
        self.sleeveRSeamFeature.fitPlane()
        self.sleeveRMidFeature.fitPlane()
        self.sleeveREndFeature.fitPlane()
        self.sleeveLSeamFeature.fitPlane()
        self.sleeveLMidFeature.fitPlane()
        self.sleeveLEndFeature.fitPlane()
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()

        if self.focusGripFeature:
            self.previousContainmentTriangle = [
                self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)),
                self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)),
                self.focusGripFeature.plane.org
            ]

        self.additionalAction = np.zeros(22) #reset control
        #update controller specific variables and produce
        if self.replayingActionTrajectory:
            #self.additionalAction = self.actionTrajectory[self.numSteps]
            if len(self.actionTrajectoryGripEvents) > self.nextTrajectoryEvent:
                if self.actionTrajectoryGripEvents[self.nextTrajectoryEvent][0] == self.numSteps:
                    self.currentController = self.actionTrajectoryGripEvents[self.nextTrajectoryEvent][1]
                    self.controllers[self.currentController].setup()
                    self.controllers[self.currentController].update()
                    self.nextTrajectoryEvent += 1
            if self.currentController > 0:
                self.controllers[self.currentController].update()

        else:
            if self.currentController is not None:
                self.controllers[self.currentController].update()
                if self.controllers[self.currentController].transition():
                    changed = self.currentController
                    self.currentController = min(len(self.controllers)-1, self.currentController+1)
                    changed = (changed != self.currentController)
                    if changed:
                        self.controllers[self.currentController].setup()
                        self.controllers[self.currentController].update()
                        self.stepsSinceControlSwitch = 0
                obs = self._get_obs()
                self.additionalAction = self.controllers[self.currentController].query(obs)

        self.stepsSinceControlSwitch += 1
        a=0

    def checkTermination(self, tau, s, obs):

        if self.recordForRendering:
            fname = self.recordForRenderingOutputPrefix
            perturbationfname_ix = fname + "_perturbation%05d" % self.renderSaveSteps
            self.savePerturbationDir(perturbationfname_ix)

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
        if self.collarTermination and self.simulateCloth and self.collarTerminationCD < self.numSteps:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                        self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                print("collar term")
                return True, -500

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        #unnecessary for master control env... not meant for training
        return 0

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        #fingertip = np.array([0.0, -0.065, 0.0])

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        #haptics
        f = np.zeros(f_size)
        if self.simulateCloth and self.hapticsAware:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        obs = np.concatenate([obs, f]).ravel()

        #contactIDs
        HSIDs = self.clothScene.getHapticSensorContactIDs()
        obs = np.concatenate([obs, HSIDs]).ravel()

        #right ef target
        efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        obs = np.concatenate([obs, self.rightTarget, efR, self.rightTarget-efR]).ravel()

        #left ef target
        efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        obs = np.concatenate([obs, self.leftTarget, efL, self.leftTarget-efL]).ravel()

        #feature
        if self.focusFeature is None or self.focusFeatureNode is None:
            obs = np.concatenate([obs, np.zeros(6)]).ravel()
        else:
            centroid = self.focusFeature.plane.org
            ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
            disp = centroid - ef
            obs = np.concatenate([obs, centroid, disp]).ravel()

        #oracle
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
                    '''
                    # oracle points to the garment when ef not in contact
                    ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
                    # closeVert = self.clothScene.getCloseVertex(p=efR)
                    # target = self.clothScene.getVertexPos(vid=closeVert)

                    centroid = self.focusFeature.plane.org

                    target = centroid
                    vec = target - ef
                    oracle = vec / np.linalg.norm(vec)
                    '''

                    # new: oracle points to the tuck triangle centroid when not in contact with cloth
                    target = np.zeros(3)
                    for c in self.previousContainmentTriangle:
                        target += c
                    target /= 3.0
                    ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
                    #efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
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

        #left orientation target
        if(self.leftOrientationTarget is not None):
            efLV = self.robot_skeleton.bodynodes[12].to_world(self.fingertip) - self.robot_skeleton.bodynodes[12].to_world(np.zeros(3))
            efLDir = efLV / np.linalg.norm(efLV)
            obs = np.concatenate([obs, self.leftOrientationTarget, efLDir]).ravel()

        return obs

    def additionalResets(self):

        self.nextTrajectoryEvent = 0
        if self.actionTrajectoryTesting:
            self.recordingActionTrajectory = True
            if self.reset_number > 0:
                self.recordingActionTrajectory = False
                self.replayingActionTrajectory = True
        # do any additional resetting here
        self.stepsSinceControlSwitch = 0
        print("Success Record so far: " + str(self.successRecord))
        print("numTrials = " + str(len(self.successRecord)))
        if len(self.successRecord) > 0:
            successfulCount = 0
            failCounter = np.zeros(7)
            for r in self.successRecord:
                if r[0]:
                    successfulCount += 1
                else:
                    failCounter[r[2]] += 1
            print("Success Rate: " + str(successfulCount / len(self.successRecord)))
            print("Failure Counter: " + str(failCounter))
            self.saveSuccessData()

        self.frameskip = 4
        self.SPDTorqueLimits = False
        self.SPDPerFrame = False
        self.SPD = None

        #reset these in case the SPD controller has already run...
        self.robot_skeleton.joint(6).set_damping_coefficient(0, 1)
        self.robot_skeleton.joint(6).set_damping_coefficient(1, 1)
        self.robot_skeleton.joint(11).set_damping_coefficient(0, 1)
        self.robot_skeleton.joint(11).set_damping_coefficient(1, 1)

        '''count = 0
        recordForRenderingDirectory = "saved_render_states/tshirtseq3" + str(count)
        while(os.path.exists(recordForRenderingDirectory)):
            count += 1
            recordForRenderingDirectory = "saved_render_states/tshirtseq3" + str(count)
        self.recordForRenderingOutputPrefix = recordForRenderingDirectory+"/tshirtseq"
        if self.recordForRendering:
            if not os.path.exists(recordForRenderingDirectory):
                os.makedirs(recordForRenderingDirectory)'''

        self.resetTime = time.time()
        #do any additional resetting here
        #fingertip = np.array([0, -0.065, 0])
        self.currentController = 0
        '''
        if self.simulateCloth:
            up = np.array([0,1.0,0])
            varianceR = pyutils.rotateY(((random.random()-0.5)*2.0)*0.3)
            adjustR = pyutils.rotateY(0.2)
            R = self.clothScene.rotateTo(v1=np.array([0,0,1.0]), v2=up)
            self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
            self.clothScene.rotateCloth(cid=0, R=R)
            self.clothScene.rotateCloth(cid=0, R=adjustR)
            self.clothScene.rotateCloth(cid=0, R=varianceR)
        '''

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q

        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode = None

        repeat = True
        if self.simulateCloth:
            while repeat:
                repeat = False

                # update physx capsules
                self.updateClothCollisionStructures(hapticSensors=True)
                self.clothScene.clearInterpolation()

                self.clothScene.reset()
                up = np.array([0, 1.0, 0])
                varianceAngle = ((random.random() - 0.5) * 2.0) * 0.3
                varianceAngle = (((self.reset_number / 25.0) - 0.5) * 2.0) * 0.3
                varianceAngle = 0.05
                #varianceAngle += random.random()*0.05
                # varianceAngle = -0.23856667449388155
                # varianceAngle = 0.27737702150062965 #chosen for good state
                # varianceAngle = -0.2620526592974481
                varianceR = pyutils.rotateY(varianceAngle)
                # print("varianceAngle: " + str(varianceAngle))
                adjustR = pyutils.rotateY(0.2)
                R = self.clothScene.rotateTo(v1=np.array([0, 0, 1.0]), v2=up)
                self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
                self.clothScene.rotateCloth(cid=0, R=R)
                self.clothScene.rotateCloth(cid=0, R=adjustR)
                self.clothScene.rotateCloth(cid=0, R=varianceR)

                self.collarFeature.fitPlane()
                self.gripFeatureL.fitPlane(normhint=np.array([0, 0, -1.0]))
                self.gripFeatureR.fitPlane(normhint=np.array([0, 0, -1.0]))
                self.localLeftOrientationTarget = np.array([0.0, -1.0, 0.0])
                self.localLeftOrientationTarget = np.array([-0.5, -1.0, -1.0])
                self.localLeftOrientationTarget /= np.linalg.norm(self.localLeftOrientationTarget)

                for i in range(self.simStepsInReset):
                    self.clothScene.step()
                    self.collarFeature.fitPlane()
                    self.gripFeatureL.fitPlane()
                    self.gripFeatureR.fitPlane()

                # check for inverted frames
                if self.gripFeatureL.plane.normal.dot(np.array([0, 0, -1.0])) < 0:
                    # TODO: trash these and reset again
                    print("INVERSION 1")
                    #repeat = True
                    #continue

                # apply random force to the garment to introduce initial state variation
                if self.initialPerturbationScale > 0:
                    force = np.random.uniform(low=-1, high=1, size=3)
                    force /= np.linalg.norm(force)
                    force *= 0.5  # scale
                    numParticles = self.clothScene.getNumVertices(cid=0)
                    forces = np.zeros(numParticles * 3)
                    for i in range(numParticles):
                        forces[i * 3] = force[0]
                        forces[i * 3 + 1] = force[1]
                        forces[i * 3 + 2] = force[2]
                    self.clothScene.addAccelerationToParticles(cid=0, a=forces)
                    # print("applied " + str(force) + " force to cloth.")

                    for i in range(self.simStepsAfterPerturbation):
                        self.clothScene.step()
                        self.collarFeature.fitPlane()
                        self.gripFeatureL.fitPlane()
                        self.gripFeatureR.fitPlane()
                        # check for inverted frames

                self.collarFeature.fitPlane()
                self.sleeveRSeamFeature.fitPlane()
                self.sleeveRMidFeature.fitPlane()
                self.sleeveREndFeature.fitPlane()
                self.sleeveLSeamFeature.fitPlane()
                self.sleeveLMidFeature.fitPlane()
                self.sleeveLEndFeature.fitPlane()
                self.gripFeatureL.fitPlane()
                self.gripFeatureR.fitPlane()

                #ensure feature normals are correct (right sleeve)
                CP2_CP1 = self.sleeveREndFeature.plane.org - self.sleeveRMidFeature.plane.org
                CP2_CP0 = self.sleeveRSeamFeature.plane.org - self.sleeveRMidFeature.plane.org

                # if CP2 normal is not facing the sleeve end invert it
                if CP2_CP1.dot(self.sleeveRMidFeature.plane.normal) < 0:
                    self.sleeveRMidFeature.plane.normal *= -1.0

                # if CP1 normal is facing the sleeve middle invert it
                if CP2_CP1.dot(self.sleeveREndFeature.plane.normal) < 0:
                    self.sleeveREndFeature.plane.normal *= -1.0

                # if CP0 normal is not facing sleeve middle invert it
                if CP2_CP0.dot(self.sleeveRSeamFeature.plane.normal) > 0:
                    self.sleeveRSeamFeature.plane.normal *= -1.0

                # ensure feature normals are correct (left sleeve)
                CP2_CP1 = self.sleeveLEndFeature.plane.org - self.sleeveLMidFeature.plane.org
                CP2_CP0 = self.sleeveLSeamFeature.plane.org - self.sleeveLMidFeature.plane.org

                # if CP2 normal is not facing the sleeve end invert it
                if CP2_CP1.dot(self.sleeveLMidFeature.plane.normal) < 0:
                    self.sleeveLMidFeature.plane.normal *= -1.0

                # if CP1 normal is facing the sleeve middle invert it
                if CP2_CP1.dot(self.sleeveLEndFeature.plane.normal) < 0:
                    self.sleeveLEndFeature.plane.normal *= -1.0

                # if CP0 normal is not facing sleeve middle invert it
                if CP2_CP0.dot(self.sleeveLSeamFeature.plane.normal) > 0:
                    self.sleeveLSeamFeature.plane.normal *= -1.0

                if self.handleNode is not None:
                    self.handleNode.clearHandles()
                    self.handleNode = None

                if self.gripFeatureL.plane.normal.dot(np.array([0, 0, -1.0])) < 0:
                    # TODO: trash these and reset again
                    print("INVERSION 2: " +str(self.gripFeatureL.plane.normal.dot(np.array([0, 0, -1.0]))))
                    repeat = True
                    continue

        self.controllers[self.currentController].setup()

        self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVerticesL, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVerticesL, clothscene=self.clothScene)*0.03
        self.leftOrientationTarget = self.gripFeatureL.plane.toWorld(self.localLeftOrientationTarget)

        self.curPerDir = 0
        self.perturbationDirections = []
        for k in range(4):
            self.perturbationDirections.append(np.random.uniform(low=-1, high=1, size=3))
            #for i in range(self.reset_number): #should make same seed randomness possible
            self.perturbationDirections[-1] = np.random.uniform(low=-1, high=1, size=3)
            while np.linalg.norm(self.perturbationDirections[-1]) > 1.0:
                self.perturbationDirections[-1] = np.random.uniform(low=-1, high=1, size=3)
            self.perturbationDirections[-1] /= np.linalg.norm(self.perturbationDirections[-1])

        #print(self.clothScene.getFriction())
        a=0

    def extraRenderFunction(self):
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        renderUtils.drawLineStrip(points=[bottomNeck, bottomHead, topHead])

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])


        #seed display
        textHeight = 15
        textLines = 2
        self.clothScene.drawText(x=15., y=textLines * textHeight, text="Seed = " + str(self.setSeed),
                                 color=(0., 0, 0))
        textLines += 1

        if self.perturbationActive:
            #print("active perturbation")
            nodePoint = self.robot_skeleton.bodynodes[self.perturbationNode].to_world(self.perturbationOffset)
            renderUtils.drawArrow(p0=nodePoint, p1=nodePoint+self.perturbationDirections[self.curPerDir]*self.perturbationMagnitude*0.01)

        if self.renderDetails:
            renderUtils.setColor(color=[0.0, 0.0, 0])
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(0, 0, 0)
            GL.glVertex3d(-1, 0, 0)
            GL.glEnd()

            if not self.simpleUI:
                self.collarFeature.drawProjectionPoly(renderNormal=False, renderBasis=False)
                self.gripFeatureL.drawProjectionPoly(renderNormal=False, renderBasis=False)
                self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0,1,0])
                self.sleeveRSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                self.sleeveRMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                self.sleeveREndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
                self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)


            #restPose rendering
            if self.renderRestPose:
                links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
                renderUtils.drawLines(lines=links)

            if self.renderGeodesic:
                for v in range(self.clothScene.getNumVertices()):
                    side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
                    side0geo = self.separatedMesh.nodes[v].geodesic

                    pos = self.clothScene.getVertexPos(vid=v)
                    norm = self.clothScene.getVertNormal(vid=v)
                    renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side0geo))
                    renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01, slices=3)
                    renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side1geo))
                    renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01, slices=3)

            if self.renderOracle and not self.simpleUI:
                ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
                renderUtils.drawArrow(p0=ef, p1=ef + self.prevOracle)


        #render targets
        #fingertip = np.array([0,-0.065,0])

            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            if self.renderRightTarget and not self.simpleUI:
                renderUtils.setColor(color=[1.0,0,0])
                renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
                renderUtils.drawLineStrip(points=[self.rightTarget, efR])

            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            if self.renderLeftTarget and not self.simpleUI:
                renderUtils.setColor(color=[0, 1.0, 0])
                renderUtils.drawSphere(pos=self.leftTarget,rad=0.02)
                renderUtils.drawLineStrip(points=[self.leftTarget, efL])

            if self.renderContainmentTriangle and not self.simpleUI:
                renderUtils.setColor([1.0, 1.0, 0])
                renderUtils.drawTriangle(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2])
                U = self.previousContainmentTriangle[1] - self.previousContainmentTriangle[0]
                V = self.previousContainmentTriangle[2] - self.previousContainmentTriangle[0]

                tnorm = np.cross(U, V)
                tnorm /= np.linalg.norm(tnorm)
                centroid = (self.previousContainmentTriangle[0] + self.previousContainmentTriangle[1] + self.previousContainmentTriangle[2])/3.0
                renderUtils.drawLines(lines=[[centroid, centroid+tnorm]])

            textHeight = 15
            textLines = 2

            if self.renderUI:
                renderUtils.setColor(color=[0.,0,0])
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
                textLines += 1
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
                textLines += 1
                self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
                textLines += 1
                if self.numSteps > 0:
                    renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=self.renderRestPose)

                if self.stepsSinceControlSwitch < 50 and len(self.controllers) > 0:
                    label = self.controllers[self.currentController].name
                    self.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60,
                                             text="Active Controller = " + str(label), color=(0., 0, 0))

    def saveSuccessData(self):
        f = open(self.successTrackingFile, 'w')

        successfulCount = 0
        failCounter = np.zeros(7)
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

    def savePerturbationDir(self, filename):
        print("saving perturbation state")

        if filename is None:
            filename = "perturbationState"
        print("filename " + str(filename))
        f = open(filename, 'w')
        if self.perturbationActive:
            org = self.robot_skeleton.bodynodes[self.perturbationNode].to_world(self.perturbationOffset)
            for i in range(3):
                f.write(str(org[i]))
                f.write(" ")
            for i in range(3):
                if i > 0:
                    f.write(" ")
                f.write(str(self.perturbationDirections[self.curPerDir][i]))
        f.close()