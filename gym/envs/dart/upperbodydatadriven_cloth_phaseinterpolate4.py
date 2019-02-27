# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)
# Phase interpolate2: end of match grip to tuck for 2nd sleeve

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
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

class DartClothUpperBodyDataDrivenClothPhaseInterpolate4Env(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True

        #observation terms
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True #if true, rewarded for 0 torso angle from vertical
        self.stableHeadReward           = True  # if True, rewarded for - head/torso angle
        self.deformationPenalty         = True
        self.restPoseReward             = True
        self.rightTargetReward          = False
        self.leftTargetReward           = False
        self.contactSurfaceReward       = False  # reward for end effector touching inside cloth
        self.contactGeoReward           = False  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if touching feature).
        self.triangleContainmentReward  = False  # active ef rewarded for intersection with triangle from shoulders to passive ef. Also penalized for distance to triangle
        self.triangleAlignmentReward    = False  # dot product reward between triangle normal and character torso vector
        self.tieredTuckReward           = False  # if true, no reward given for containment unless contact is achieved.

        #reward weights
        self.uprightRewardWeight = 1
        self.stableHeadRewardWeight = 2
        self.deformationPenaltyWeight = 10  # was 5...
        self.restPoseRewardWeight = 0.5
        self.contactSurfaceRewardWeight = 6
        self.contactGeoRewardWeight = 2
        self.triangleContainmentRewardWeight = 10
        self.triangleAlignmentRewardWeight = 2


        #other flags
        self.collarTermination = False  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 0 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.loadTargetsFromROMPositions = False
        self.resetPoseFromROMPoints = False
        self.resetTime = 0
        self.resetStateFromDistribution = True
        self.resetDistributionPrefix = "saved_control_states/enter_seq_final"
        self.resetDistributionSize = 20

        #other variables
        self.handleNode = None
        self.updateHandleNodeFrom = 7  # right fingers
        self.prevTau = None
        self.maxDeformation = 30.0
        self.restPose = None
        self.localRightEfShoulder1 = None
        self.localLeftEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.prevErrors = None #stores the errors taken from DART each iteration
        self.previousDeformationReward = 0
        self.previousSelfCollisions = 0
        self.fingertip = np.array([0, -0.075, 0])
        self.previousContainmentTriangle = [np.zeros(3), np.zeros(3), np.zeros(3)]

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 66
        if self.contactIDInObs:
            observation_size += 22
        if self.rightTargetReward:
            observation_size += 9
        if self.leftTargetReward:
            observation_size += 9

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothMeshStateFile = "objFile_regrip.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        # clothing features
        #self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        #self.targetGripVerticesL = [46, 437, 955, 1185, 47, 285, 711, 677, 48, 905, 1041, 49, 741, 889, 45]
        self.targetGripVerticesR = [905, 1041, 49, 435, 50, 570, 992, 1056, 51, 676, 283, 52, 489, 892, 362, 53]
        #self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        #self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        #self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene, b1spanverts=[889,1041], b2spanverts=[47,677])
        #self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene, b1spanverts=[362,889], b2spanverts=[51,992])
        #self.interiorTuckFeature = ClothFeature(verts=[1032], clothScene=self.clothScene)
        #self.interiorTuckFeature = ClothFeature(verts=[403], clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        # load rewards into the RewardsData structure
        if self.uprightReward:
            self.rewardsData.addReward(label="upright",rmin=-2.5,rmax=0,rval=0, rweight=self.uprightRewardWeight)

        if self.stableHeadReward:
            self.rewardsData.addReward(label="stable head",rmin=-1.2,rmax=0,rval=0, rweight=self.stableHeadRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0, rweight=self.deformationPenaltyWeight)

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.contactSurfaceReward:
            self.rewardsData.addReward(label="contact surface", rmin=-1, rmax=1.0, rval=0, rweight=self.contactSurfaceRewardWeight)

        if self.contactGeoReward:
            self.rewardsData.addReward(label="contact geo", rmin=0, rmax=1.0, rval=0, rweight=self.contactGeoRewardWeight)

        if self.triangleContainmentReward:
            self.rewardsData.addReward(label="tri contain", rmin=-1.0, rmax=1.0, rval=0, rweight=self.triangleContainmentRewardWeight)

        if self.triangleAlignmentReward:
            self.rewardsData.addReward(label="tri align", rmin=-1.0, rmax=1.0, rval=0, rweight=self.triangleAlignmentRewardWeight)


        self.state_save_directory = "saved_control_states/"
        self.saveStateOnReset = False

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame
        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVertices, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVertices, clothscene=self.clothScene)*0.03

        '''if self.collarFeature is not None:
            self.collarFeature.fitPlane()
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()'''

        '''if self.triangleContainmentReward:
            self.previousContainmentTriangle = [
                self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)),
                self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)),
                self.gripFeatureR.plane.org
                #self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            ]

        '''
        # update handle nodes
        if self.handleNode is not None:
            self.handleNode.clearHandles()
            #self.handle_node = None

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -5000
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -5000
        '''elif self.collarTermination and self.simulateCloth and self.collarTerminationCD < self.numSteps:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                        self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                #print("collar term")
                return True, -5000'''

        if self.numSteps == 70:
            if self.saveStateOnReset and self.reset_number > 0:
                fname = self.state_save_directory + "ltuck_wide"
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

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        localRightEfShoulder2 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2)  # right fingertip in right shoulder local frame
        localLeftEfShoulder2 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2)  # right fingertip in right shoulder local frame

        #store the previous step's errors
        #if self.reset_number > 0 and self.numSteps > 0:
        #    self.prevErrors = self.dart_world.getAllConstraintViolations()
            #print("after getAllCon..")
            #print("prevErrors: " +str(self.prevErrors))

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

        #startTime = time.time()
        clothDeformation = 0
        if self.simulateCloth:
            maxDef, minDef, avgDef, varDef, ratios = self.clothScene.getAllDeformationStats(0)
            #clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            clothDeformation = maxDef
            self.deformation = clothDeformation
        #print("deformation check took: " + str(time.time()-startTime))

        '''startTime = time.time()
        if self.simulateCloth:
            self.previousSelfCollisions = self.clothScene.getNumSelfCollisions(0, culldistance=0.04)

        print("self collision = " + str(self.previousSelfCollisions) + " | check took: " + str(time.time() - startTime))
        '''

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_record.append(reward_clothdeformation)

        self.previousDeformationReward = reward_clothdeformation

        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = max(-51, -dist)
            reward_record.append(reward_restPose)

        reward_contact_surface = 0
        if self.contactSurfaceReward:
            CID = self.clothScene.getHapticSensorContactIDs()[21]
            # if CID > 0:
            #    reward_contact_surface = 1.0
            # print(CID)
            reward_contact_surface = CID  # -1.0 for full outside, 1.0 for full inside
            # print(reward_contact_surface)
            reward_record.append(reward_contact_surface)

        avgContactGeodesic = None
        '''
        if self.numSteps > 0 and self.simulateCloth:
            contactInfo = pyutils.getContactIXGeoSide(sensorix=21, clothscene=self.clothScene,
                                                      meshgraph=self.separatedMesh)
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)
        self.previousContactGeo = avgContactGeodesic
        '''

        reward_contactGeo = 0
        if self.contactGeoReward:
            if self.simulateCloth:
                if avgContactGeodesic is not None:
                    reward_contactGeo = 1.0 - (avgContactGeodesic / self.separatedMesh.maxGeo)
                    # reward_contactGeo = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo
            #reward_record.append(reward_contactGeo)
            reward_record.append(reward_contactGeo)

        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        reward_rightTarget = 0
        if self.rightTargetReward:
            rDist = np.linalg.norm(self.rightTarget-wRFingertip2)
            #reward_rightTarget = -rDist - rDist**2
            reward_rightTarget = -rDist
            '''if rDist < 0.02:
                reward_rightTarget += 0.25'''

        reward_leftTarget = 0
        if self.leftTargetReward:
            lDist = np.linalg.norm(self.leftTarget - wLFingertip2)
            #reward_leftTarget = -lDist - lDist**2
            reward_leftTarget = -lDist
            '''if lDist < 0.02:
                reward_leftTarget += 0.25'''

        reward_triangleContainment = 0
        if self.triangleContainmentReward:
            if not self.tieredTuckReward or not self.contactSurfaceReward or reward_contact_surface > 0:
                # check intersection
                lines = [
                    [self.robot_skeleton.bodynodes[12].to_world(self.fingertip),
                     self.robot_skeleton.bodynodes[11].to_world(np.zeros(3))],
                    [self.robot_skeleton.bodynodes[11].to_world(np.zeros(3)),
                     self.robot_skeleton.bodynodes[10].to_world(np.zeros(3))]
                ]
                intersect = False
                aligned = False
                for l in lines:
                    hit, dist, pos, aligned = pyutils.triangleLineSegmentIntersection(self.previousContainmentTriangle[0],
                                                                                      self.previousContainmentTriangle[1],
                                                                                      self.previousContainmentTriangle[2],
                                                                                      l0=l[0], l1=l[1])
                    if hit:
                        # print("hit: " + str(dist) + " | " + str(pos) + " | " + str(aligned))
                        intersect = True
                        aligned = aligned
                        break
                if intersect:
                    if aligned:
                        reward_triangleContainment = 1
                    else:
                        reward_triangleContainment = -1
                else:
                    # if no intersection, get distance
                    triangleCentroid = (self.previousContainmentTriangle[0] + self.previousContainmentTriangle[1] +
                                        self.previousContainmentTriangle[2]) / 3.0
                    dist = np.linalg.norm(lines[0][0] - triangleCentroid)
                    # dist, pos = pyutils.distToTriangle(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2],p=lines[0][0])
                    # print(dist)
                    reward_triangleContainment = -dist
            else:
                reward_triangleContainment = -1
            reward_record.append(reward_triangleContainment)

        reward_triangleAlignment = 0
        if self.triangleAlignmentReward:
            U = self.previousContainmentTriangle[1] - self.previousContainmentTriangle[0]
            V = self.previousContainmentTriangle[2] - self.previousContainmentTriangle[0]

            tnorm = np.cross(U, V)
            tnorm /= np.linalg.norm(tnorm)

            torsoV = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3)) - self.robot_skeleton.bodynodes[
                1].to_world(np.zeros(3))
            torsoV /= np.linalg.norm(torsoV)

            reward_triangleAlignment = -tnorm.dot(torsoV)
            reward_record.append(reward_triangleAlignment)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        #print("reward_restPose: " + str(reward_restPose))
        #print("reward_leftTarget: " + str(reward_leftTarget))
        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight \
                      + reward_stableHead * self.stableHeadRewardWeight \
                      + reward_clothdeformation * self.deformationPenaltyWeight \
                      + reward_restPose * self.restPoseRewardWeight\
                      + reward_rightTarget*100 \
                      + reward_leftTarget*100 \
                      + reward_contact_surface * self.contactSurfaceRewardWeight \
                      + reward_triangleContainment * self.triangleContainmentRewardWeight \
                      + reward_triangleAlignment * self.triangleAlignmentRewardWeight \
                      + reward_contactGeo * self.contactGeoRewardWeight

        return self.reward

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
        '''if self.resetTime > 0:
            print("reset " + str(self.reset_number) + " after " + str(time.time()-self.resetTime))
        '''
        self.resetTime = time.time()

        '''if self.reset_number == 0 and self.simulateCloth:
            self.separatedMesh.initSeparatedMeshGraph()
            self.separatedMesh.updateWeights()
            # TODO: compute geodesic depends on cloth state! Deterministic initial condition required!
            # option: maybe compute this before setting the cloth state at all in initialization?
            self.separatedMesh.computeGeodesic(feature=self.interiorTuckFeature, oneSided=False, side=0)'''


        #do any additional resetting here
        '''if self.simulateCloth:
            up = np.array([0,1.0,0])
            varianceR = pyutils.rotateY(((random.random()-0.5)*2.0)*0.3)
            adjustR = pyutils.rotateY(0.2)
            R = self.clothScene.rotateTo(v1=np.array([0,0,1.0]), v2=up)
            self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
            self.clothScene.rotateCloth(cid=0, R=R)
            self.clothScene.rotateCloth(cid=0, R=adjustR)
            self.clothScene.rotateCloth(cid=0, R=varianceR)'''
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)

        '''if self.resetPoseFromROMPoints and len(self.ROMPoints) > 0:
            poseFound = False
            while not poseFound:
                ix = random.randint(0,len(self.ROMPoints)-1)
                qpos = self.ROMPoints[ix]
                efR = self.ROMPositions[ix][:3]
                efL = self.ROMPositions[ix][-3:]
                if efR[2] < 0 and efL[2] < 0: #half-plane constraint on end effectors
                    poseFound = True
        '''
        #Check the constrained population
        '''positive = 0
        for targets in self.ROMPositions:
            efR = targets[:3]
            efL = targets[-3:]
            if efR[2] < 0 and efL[2] < 0:
                positive += 1
        print("Valid Poses: " + str(positive) + " | ratio: " + str(positive/len(self.ROMPositions)))'''


        '''if self.loadTargetsFromROMPositions and len(self.ROMPositions) > 0:
            targetFound = False
            while not targetFound:
                ix = random.randint(0, len(self.ROMPositions) - 1)
                self.rightTarget = self.ROMPositions[ix][:3] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                self.leftTarget = self.ROMPositions[ix][-3:] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                if self.rightTarget[2] < 0 and self.leftTarget[2] < 0: #half-plane constraint on end effectors
                    targetFound = True
        self.set_state(qpos, qvel)'''

        #self.loadCharacterState(filename="characterState_startarm2")

        #find end effector targets and set restPose from solution
        self.leftTarget = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.rightTarget = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        #print("right target: " + str(self.rightTarget))
        #print("left target: " + str(self.leftTarget))
        self.restPose = np.array(self.robot_skeleton.q)

        if self.resetStateFromDistribution:
            if self.reset_number == 0: #load the distribution
                count = 0
                objfname_ix = self.resetDistributionPrefix + "%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    #print(objfname_ix)
                    self.clothScene.addResetStateFrom(filename=objfname_ix+".obj")
                    objfname_ix = self.resetDistributionPrefix + "%05d" % count

            resetStateNumber = random.randint(0,self.resetDistributionSize-1)
            #resetStateNumber = self.reset_number%self.resetDistributionSize
            #resetStateNumber = 0
            #print("resetState: " + str(resetStateNumber))
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            self.loadCharacterState(filename=charfname_ix)
            #self.updateClothCollisionStructures(hapticSensors=True)
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.2, high=0.2, size=self.robot_skeleton.ndofs)
            self.robot_skeleton.set_velocities(qvel)
        else:
            self.loadCharacterState(filename="characterState_regrip")

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            #self.handle_node = None
            self.handleNode.addVertices(verts=self.targetGripVerticesR)
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()

        '''if self.simulateCloth:
            self.collarFeature.fitPlane()
            self.gripFeatureL.fitPlane()
            self.gripFeatureR.fitPlane()'''

        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        renderUtils.drawLineStrip(points=[bottomNeck, bottomHead, topHead])
        '''if self.collarFeature is not None:
            self.collarFeature.drawProjectionPoly(renderNormal=False, renderBasis=False)
        self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0,1,0])'''

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #render targets
        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            renderUtils.setColor(color=[1.0,0,0])
            renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.rightTarget, efR])
        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            renderUtils.drawSphere(pos=self.leftTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.leftTarget, efL])

        if self.restPoseReward:
            links = pyutils.getRobotLinks(self.robot_skeleton,pose=self.restPose)
            renderUtils.drawLines(lines=links)

        if self.triangleContainmentReward:
            renderUtils.setColor([1.0, 1.0, 0])
            renderUtils.drawTriangle(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2])
            U = self.previousContainmentTriangle[1] - self.previousContainmentTriangle[0]
            V = self.previousContainmentTriangle[2] - self.previousContainmentTriangle[0]

            tnorm = np.cross(U, V)
            tnorm /= np.linalg.norm(tnorm)
            centroid = (self.previousContainmentTriangle[0] + self.previousContainmentTriangle[1] + self.previousContainmentTriangle[2])/3.0
            renderUtils.drawLines(lines=[[centroid, centroid+tnorm]])

        '''if self.contactGeoReward:
            renderUtils.drawSphere(pos=self.clothScene.getVertexPos(cid=0, vid=self.interiorTuckFeature.verts[0])'''

        # render geodesic
        if False:
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
                side0geo = self.separatedMesh.nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo,
                                                                    value=self.separatedMesh.maxGeo - side0geo))
                renderUtils.drawSphere(pos=pos - norm * 0.01, rad=0.01)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo,
                                                                    value=self.separatedMesh.maxGeo - side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)

        m_viewport = self.viewer.viewport
        self.rewardsData.render(topLeft=[m_viewport[2]-410,m_viewport[3]-15], dimensions=[400, -m_viewport[3]+30])

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
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Self Collisions = " + str(self.previousSelfCollisions), color=(0., 0, 0))
            textLines += 1

            #display deformation stats
            maxDef, minDef, avgDef, varDef, ratios = self.clothScene.getAllDeformationStats(0)
            stdDevDef = math.sqrt(varDef)
            percentWithin1StdDevDef = 0
            avgDefAboveStdDev = 0
            numDefAboveStdDev = 0
            for r in ratios:
                if abs(avgDef-r) <= avgDef+stdDevDef:
                    percentWithin1StdDevDef += 1
                elif r-avgDef > avgDef+stdDevDef:
                    avgDefAboveStdDev += r
                    numDefAboveStdDev += 1
            if numDefAboveStdDev > 0:
                avgDefAboveStdDev /= numDefAboveStdDev
            percentWithin1StdDevDef /= len(ratios)
            percentWithin1StdDevDef *= 100

            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Deformation [min,max] = [" + str(minDef) + "," + str(maxDef) + "]", color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Deformation [avg,var] = [" + str(avgDef) + "," + str(varDef) + "]", color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Deformation percent within 1 stdDev = " + str(percentWithin1StdDevDef), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Deformation avg above 1 stdDev = " + str(avgDefAboveStdDev), color=(0., 0, 0))
            textLines += 1



            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 60], h=16, w=60, progress=maxDef/25.0, color=[0.0, 1.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 90], h=16, w=60, progress=avgDefAboveStdDev/25.0, color=[0.0, 1.0, 0])
