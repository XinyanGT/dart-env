# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
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

class DartClothUpperBodyDataDrivenTshirtEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.prefix = os.path.dirname(__file__)

        #rendering variables
        self.useOpenGL = True
        self.screenSize = (1080, 720)
        self.renderDARTWorld = True
        self.renderUI = True
        self.renderDisplacerAccuracy = False
        self.renderContactInfo = True
        self.compoundAccuracy = True
        self.recordHistory = False
        self.recordROMPoints = True
        self.ROMPoints = []
        self.ROMPointMinDistance = 1.0

        #sim variables
        self.gravity = False
        self.resetRandomPose = False
        self.resetFile = None#self.prefix + "/assets/ROMPoints_upperbodycapsules_datadriven"
        self.dataDrivenJointLimts = True
        simulateCloth = False
        renderCloth = False

        self.arm = 0 # 0->both, 1->right, 2->left
        self.actuatedDofs = np.arange(22) # full upper body
        self.lockedDofs = []
        self.limbNodesR = [3, 4, 5, 6, 7]
        self.limbNodesL = [8, 9, 10, 11, 12]
        self.efOffset = np.array([0,-0.06,0])

        if self.arm == 1:
            self.actuatedDofs = np.arange(3, 11) # right arm
            self.lockedDofs = np.concatenate([np.arange(3), np.arange(11, 22)])

        elif self.arm == 2:
            self.actuatedDofs = np.arange(11, 19) # left arm
            self.lockedDofs = np.concatenate([np.arange(11), np.arange(19, 22)])

        #task modes
        self.upright_active = True
        self.restPose_active = False
        self.restPose = None
        self.rightDisplacer_active = False
        self.leftDisplacer_active = False
        self.displacerMod1 = False #temporary switch to modified reward scheme
        self.upReacher_active = False
        self.rightTarget_active = False
        self.leftTarget_active = False
        self.prevTauObs = False #if True, T(t-1) is included in the obs
        #dressing terms
        self.elbowFlairReward = True
        self.elbowFlairNode = 10
        self.limbProgressReward = True #if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementReward = True #if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward = True #if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.collarTermination = True #if true and self.collarFeature is defined, head/neck not contained in this feature results in termination
        self.deformationTermination = False
        self.deformationPenalty = True
        self.maxDeformation = 30.0
        self.featureInObs = True #if true, feature centroid location and dispalcement from ef are observed
        self.oracleInObs = True #if true, oracle vector is in obs
        self.contactIDInObs = True #if true, contact ids are in obs
        self.hapticsInObs = True #if true, haptics are in observation
        self.hapticsAware = True  # if false, 0's for haptic input


        self.interArmCollisions = True
        self.shoulderCollisions = True

        self.limbProgress = 0
        self.prevOracle = np.zeros(3)
        self.minContactGeo = 0

        self.rightDisplacement = np.zeros(3) #should be unit vector or 0
        self.leftDisplacement = np.zeros(3) #should be unit vector or 0
        self.displacementParameters = [
            0.3,    #probability of 0 vector on reset
            0.01,   #probability of displacement vector reset (each step)
            0.25,   #probability of displacement vector shift (add noise to vector and re-normalize)
            0.05,   #minimum noise magnitude
            0.1     #maximum noise magnitude
        ]
        self.cumulativeAccurateMotionR = 0
        self.cumulativeMotionR = 0
        self.cumulativeFixedMotionR = 0
        self.cumulativeFixedTimeR = 0
        self.cumulativeAccurateMotionL = 0
        self.cumulativeMotionL = 0
        self.cumulativeFixedMotionL = 0
        self.cumulativeFixedTimeL = 0
        self.displacer0TargetR = np.zeros(3) #location of the ef when (0,0,0) task is activated

        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)

        self.handleNode = None
        self.updateHandleNodeFrom = 12 #left fingers

        #22 dof upper body
        self.action_scale = np.ones(len(self.actuatedDofs))*12
        if 0 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(0)] = 50
        if 1 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(1)] = 50

        self.control_bounds = np.array([np.ones(len(self.actuatedDofs)), np.ones(len(self.actuatedDofs))*-1])
        self.prevTau = np.zeros(len(self.actuatedDofs))
        self.prevPose = np.zeros(len(self.actuatedDofs))
        self.rewardHistory = []
        self.qHistory = []
        self.dqHistory = []
        self.tHistory = []
        self.dispR = np.zeros(3)

        self.reset_number = 0 #debugging
        self.numSteps = 0

        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 66
        if self.rightDisplacer_active:
            observation_size += 3
        if self.leftDisplacer_active:
            observation_size += 3
        if self.rightTarget_active:
            observation_size += 6
        if self.leftTarget_active:
            observation_size += 6
        if self.featureInObs:
            observation_size += 6
        if self.oracleInObs:
            observation_size += 3
        if self.contactIDInObs:
            observation_size += 22


        #model_path = 'UpperBodyCapsules_v3.skel'
        model_path = 'UpperBodyCapsules_datadriven.skel'

        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01,
                                        mesh_path=self.prefix + "/assets/tshirt_m.obj",
                                        state_path=self.prefix + "/../../../../tshirt_regrip5.obj",
                                        scale=1.4)

        clothScene.togglePinned(0, 0)  # turn off auto-pin

        self.separatedMesh = None
        if simulateCloth:
            self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)

        if not renderCloth:
            clothScene.renderClothFill = False
            clothScene.renderClothBoundary = False
            clothScene.renderClothWires = False

        #TODO: add other sleeve and check/rename this
        #self.sleeveLVerts = [7, 140, 2255, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2241, 2326, 2334, 2288, 2289, 2373, 2264, 2419, 2444, 2345, 2408, 2375, 2234, 2399]
        self.sleeveRVerts = [2580, 2495, 2508, 2586, 2518, 2560, 2621, 2529, 2559, 2593, 272, 2561, 2658, 2582, 2666, 2575, 2584, 2625, 2616, 2453, 2500, 2598, 2466]
        #self.sleeveREndVerts = [255, 253, 250, 247, 266, 264, 262, 260, 258]
        # self.splineCP0Verts = [7, 2339, 2398, 2343, 2384, 2405, 2421, 2275, 2250, 134, 136, 138]
        #self.CP1Verts = [232, 230, 227, 225, 222, 218, 241, 239, 237, 235, 233]
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]


        self.CP0Feature = ClothFeature(verts=self.sleeveRVerts, clothScene=clothScene)
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=clothScene)

        #self.handle_node = None#HandleNode(clothScene, org=np.array([0.05, 0.034, -0.975]))
        if simulateCloth:
            self.handleNode = HandleNode(clothScene, org=np.array([0.05, 0.034, -0.975]))

        self.displacerTargets = [[], []]
        self.displacerActual = [[], []]

        self.reward = 0
        self.cumulativeReward = 0
        self.deformation = 0

        self.prevTime = time.time()
        self.totalTime = 0

        self.dotimings = False
        self.timings = []
        self.timingslabels = []

        #intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds, screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds , disableViewer = True, visualize = False)

        #setup data-driven joint limits
        if self.dataDrivenJointLimts:
            leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_left'), self.robot_skeleton.joint('elbowjL'), True)
            rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_right'), self.robot_skeleton.joint('elbowjR'), False)
            leftarmConstraint.add_to_world(self.dart_world)
            rightarmConstraint.add_to_world(self.dart_world)

        utils.EzPickle.__init__(self)

        if not self.gravity:
            self.dart_world.set_gravity(np.zeros(3))
        else:
            self.dart_world.set_gravity(np.array([0., -9.8, 0]))
        
        #self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.5)

        self.collisionCapsuleInfo = None #set in updateClothCollisionStructures(capsules=True)
        self.collisionSphereInfo = None #set in updateClothCollisionStructures()
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = simulateCloth

        #enable DART collision testing
        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        #setup collision filtering
        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                           self.robot_skeleton.bodynodes[12])  # left forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[5],
                                           self.robot_skeleton.bodynodes[7])  # right forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[13])  # torso to neck
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[14])  # torso to head
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[3])  # torso to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[8])  # torso to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                           self.robot_skeleton.bodynodes[3])  # neck to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                           self.robot_skeleton.bodynodes[8])  # neck to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                           self.robot_skeleton.bodynodes[3])  # head to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                           self.robot_skeleton.bodynodes[8])  # head to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[3],
                                           self.robot_skeleton.bodynodes[8])  # right shoulder to left shoulder

        if not self.interArmCollisions: #turn off collisions between the arms
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                               self.robot_skeleton.bodynodes[5])  # left forearm to right forearm
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                               self.robot_skeleton.bodynodes[4])  # left forearm to right bicep
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[5],
                                               self.robot_skeleton.bodynodes[9])  # right forearm to left bicep

        if not self.shoulderCollisions: #turn off collisions involving the shoulder nodes
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[5],
                                               self.robot_skeleton.bodynodes[3])  # right forearm to shoulder
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[6],
                                               self.robot_skeleton.bodynodes[3])  # right hand to shoulder
            collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[6],
                                               self.robot_skeleton.bodynodes[3])  # right fingers to shoulder



        #TODO: make this more generic
        self.torqueGraph = None#pyutils.LineGrapher(title="Torques")

        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])
            
        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

        #enable joint limits
        for i in range(len(self.robot_skeleton.joints)):
            print(self.robot_skeleton.joints[i])

        #DART does not automatically limit joints with any unlimited dofs
        self.robot_skeleton.joints[4].set_position_limit_enforced(True)
        self.robot_skeleton.joints[9].set_position_limit_enforced(True)

    def _getFile(self):
        return __file__

    def addTiming(self, label=None):
        self.timings.append(time.time())
        if label is None:
            self.timingslabels.append(str(len(self.timings)))
        else:
            self.timingslabels.append(label)

    def printTiming(self):
        print("Timings:")
        print(self.timings)
        print(self.timingslabels)
        if len(self.timings) > 1:
            totalTime = self.timings[-1] - self.timings[0]
            percentages = []
            for ix,t in enumerate(self.timings):
                if ix > 0:
                    percentages.append(((t-self.timings[ix-1])/totalTime)*100.0)
            print("Percentages: " + str(percentages))

    def clearTimings(self):
        self.timings = []
        self.timingslabels = []

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)

    def _step(self, a):
        #print("step " + str(self.numSteps) + " a: " + str(a))
        try:
            '''if random.random() > 0.9999:
                print("killing thread")
                sys.exit(123)'''

            qpos = np.zeros(len(self.robot_skeleton.q))
            for ix,dof in enumerate(self.robot_skeleton.q):
                if dof > 6.28:
                    qpos[ix] = dof-6.28
                    print("dof " +str(ix) +" is wrapping")
                elif dof < -6.28:
                    qpos[ix] = dof + 6.28
                    print("dof " + str(ix) + " is wrapping")
                else:
                    qpos[ix] = dof

            self.set_state(qpos, self.robot_skeleton.dq)

                #time.sleep(60)
            if self.dotimings:
                self.addTiming(label="Start")
            if self.numSteps > 1:
                self.totalTime += (time.time() - self.prevTime)
                #print(self.totalTime)
            self.prevTime = time.time()
            clamped_control = np.array(a)
            for i in range(len(clamped_control)):
                if clamped_control[i] > self.control_bounds[0][i]:
                    clamped_control[i] = self.control_bounds[0][i]
                if clamped_control[i] < self.control_bounds[1][i]:
                    clamped_control[i] = self.control_bounds[1][i]
            self.prevTau = np.array(clamped_control)
            self.prevPose = np.array(self.robot_skeleton.q)
            if self.recordHistory:
                self.qHistory.append(np.array(self.robot_skeleton.q))
                self.dqHistory.append(np.array(self.robot_skeleton.dq))
                self.tHistory.append(np.array(self.prevTau))
                self.rewardHistory.append(self.reward)
            if self.recordROMPoints:
                minDist = None
                for p in self.ROMPoints:
                    dist = np.linalg.norm(self.robot_skeleton.q-p)
                    if minDist is None:
                        minDist = dist
                    if dist < minDist:
                        minDist = dist
                        if minDist < self.ROMPointMinDistance:
                            break
                if minDist is not None:
                    if minDist > self.ROMPointMinDistance:
                        self.ROMPoints.append(np.array(self.robot_skeleton.q))
                        print("Saved poses = " + str(len(self.ROMPoints)))
                else: #auto-add when list is empty
                    self.ROMPoints.append(np.array(self.robot_skeleton.q))
            tau = np.multiply(clamped_control, self.action_scale)

            if self.reset_number > 0 and self.torqueGraph is not None:
                self.torqueGraph.yData[0][self.numSteps - 1] = tau[0]
                self.torqueGraph.yData[1][self.numSteps - 1] = tau[1]
                self.torqueGraph.update()

            fingertip = np.array([0.0, -0.06, 0.0])
            wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1) #right fingertip in right shoulder local frame
            #vecR1 = self.target-wRFingertip1
            #vecL1 = self.target2-wLFingertip1

            if self.dotimings:
                self.addTiming(label="Features/Handles")

            if self.CP0Feature is not None:
                self.CP0Feature.fitPlane()
            if self.collarFeature is not None:
                self.collarFeature.fitPlane()


            if self.handleNode is not None:
                if self.updateHandleNodeFrom >= 0:
                    self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
                self.handleNode.step()

            #apply action and simulate
            if len(tau) < len(self.robot_skeleton.q):
                newtau = np.array(tau)
                tau = np.zeros(len(self.robot_skeleton.q))
                for ix,dof in enumerate(self.actuatedDofs):
                    tau[dof] = newtau[ix]
            if self.dotimings:
                self.addTiming(label="Simulate")
            self.do_simulation(tau, self.frame_skip)
            if self.dotimings:
                self.addTiming(label="Reward")
            #set position and 0 velocity of locked dofs
            qpos = self.robot_skeleton.q
            qvel = self.robot_skeleton.dq
            for dof in self.lockedDofs:
                qpos[dof] = 0
                qvel[dof] = 0
            self.set_state(qpos, qvel)

            wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            localRightEfShoulder2 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2)  # right fingertip in right shoulder local frame
            self.dispR = wRFingertip2 - wRFingertip1
            #vecR2 = self.target-wRFingertip2
            #vecL2 = self.target2-wLFingertip2

            reward_elbow_flair = 0
            if self.elbowFlairReward:
                root = self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
                spine = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
                elbow = self.robot_skeleton.bodynodes[self.elbowFlairNode].to_world(np.zeros(3))
                dist = pyutils.distToLine(p=elbow, l0=root, l1=spine)
                z=0.5
                s=16
                l=0.2
                reward_elbow_flair = -(1 - (z * math.tanh(s*(dist-l)) + z))
                #print("reward_elbow_flair: " + str(reward_elbow_flair))

            reward_limbprogress = 0
            if self.limbProgressReward and self.simulateCloth:
                self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR,offset=np.array([0,-0.06,0])), feature=self.CP0Feature)
                reward_limbprogress = self.limbProgress
                if reward_limbprogress < 0: #remove euclidean distance penalty before containment
                    reward_limbprogress = 0

            '''minContactGeodesic = None
            if self.numSteps > 0 and self.simulateCloth:
                minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=12, clothscene=self.clothScene, meshgraph=self.separatedMesh)
                self.minContactGeo = minContactGeodesic'''

            avgContactGeodesic = None
            if self.numSteps > 0 and self.simulateCloth:
                contactInfo = pyutils.getContactIXGeoSide(sensorix=12, clothscene=self.clothScene, meshgraph=self.separatedMesh)
                if len(contactInfo) > 0:
                    avgContactGeodesic = 0
                    for c in contactInfo:
                        avgContactGeodesic += c[1]
                    avgContactGeodesic /= len(contactInfo)

            reward_contactGeo = 0
            if self.contactGeoReward and self.simulateCloth:
                if self.limbProgress > 0:
                    reward_contactGeo = 1.0
                elif avgContactGeodesic is not None:
                    reward_contactGeo = 1.0 - (avgContactGeodesic / self.separatedMesh.maxGeo)
                    #reward_contactGeo = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo

            #check cloth deformation for termination
            clothDeformation = 0
            if self.simulateCloth is True:
                clothDeformation = self.clothScene.getMaxDeformationRatio(0)
                self.deformation = clothDeformation

            reward_clothdeformation = 0
            if clothDeformation > 15 and self.deformationPenalty is True:
                reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant

            #force magnitude penalty
            reward_ctrl = -np.square(tau).sum()

            #reward for maintaining posture
            reward_upright = 0
            if self.upright_active:
                reward_upright = -abs(self.robot_skeleton.q[0])-abs(self.robot_skeleton.q[1])

            #reward for reaching up with both arms.
            reward_upreach = 0
            if self.upReacher_active:
                reward_upreach = wRFingertip2[1] + wLFingertip2[1]

            #reward for following displacement goals
            reward_displacement = 0
            if self.rightDisplacer_active:
                actual_displacement = wRFingertip2 - wRFingertip1
                if np.linalg.norm(self.rightDisplacement) == 0:
                    if self.displacerMod1:
                        reward_displacement += -np.linalg.norm(wRFingertip2-self.displacer0TargetR)
                    else:
                        reward_displacement += -np.linalg.norm(actual_displacement)
                else:
                    reward_displacement += actual_displacement.dot(self.rightDisplacement)
            if self.leftDisplacer_active:
                actual_displacement = wLFingertip2 - wLFingertip1
                if np.linalg.norm(self.leftDisplacement) == 0:
                    reward_displacement += -np.linalg.norm(actual_displacement)
                else:
                    reward_displacement += actual_displacement.dot(self.leftDisplacement)

            reward_oracleDisplacement = 0
            if self.oracleDisplacementReward and np.linalg.norm(self.prevOracle) > 0:
                #world_ef_displacement = wRFingertip2 - wRFingertip1
                relative_displacement = localRightEfShoulder2-localRightEfShoulder1
                oracle0 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2+self.prevOracle) - localRightEfShoulder2
                #oracle0 = oracle0/np.linalg.norm(oracle0)
                reward_oracleDisplacement += relative_displacement.dot(oracle0)

            reward_target = 0
            if self.rightTarget_active:
                targetDistR = np.linalg.norm(self.rightTarget-wRFingertip2)
                reward_target -= targetDistR
                if targetDistR < 0.01:
                    reward_target += 0.5
            if self.leftTarget_active:
                targetDistL = np.linalg.norm(self.leftTarget-wLFingertip2)
                reward_target -= targetDistL
                if targetDistL < 0.01:
                    reward_target += 0.5

            reward_restPose = 0
            if self.restPose_active and self.restPose is not None:
                z = 0.5 #half the max magnitude (e.g. 0.5 -> [0,1])
                s = 1.0 #steepness (higher is steeper)
                l = 4.2 #translation
                dist = np.linalg.norm(self.robot_skeleton.q-self.restPose)
                reward_restPose = -(z * math.tanh(s * (dist - l)) + z)
                #print("distance: " + str(dist) + " -> " + str(reward_restPose))


            #total reward
            self.reward = reward_ctrl*0\
                          + reward_upright\
                          + reward_upreach\
                          + reward_displacement\
                          + reward_target*10\
                          + reward_limbprogress*3\
                          + reward_contactGeo*2\
                          + reward_clothdeformation*3\
                          + reward_oracleDisplacement*50\
                          + reward_elbow_flair\
                          + reward_restPose
            self.cumulativeReward += self.reward
            if self.dotimings:
                self.addTiming(label="Observation")
            #record accuracy
            if self.renderDisplacerAccuracy and self.useOpenGL and self.numSteps>0:
                vecR = wRFingertip2 - wRFingertip1
                dispMagR = np.linalg.norm(vecR)
                if self.compoundAccuracy and len(self.displacerTargets[0]) > 0:
                    self.displacerTargets[0].append(self.displacerTargets[0][-1] + self.rightDisplacement * dispMagR)
                else:
                    self.displacerTargets[0].append(wRFingertip1+self.rightDisplacement*dispMagR)
                self.displacerActual[0].append(wRFingertip2)
                if np.linalg.norm(self.rightDisplacement) > 0:
                    self.cumulativeMotionR += dispMagR
                    self.cumulativeAccurateMotionR += vecR.dot(self.rightDisplacement)
                else:
                    self.cumulativeFixedMotionR += dispMagR
                    self.cumulativeFixedTimeR += 1
                vecL = wLFingertip2 - wLFingertip1
                dispMagL = np.linalg.norm(vecL)
                if self.compoundAccuracy and len(self.displacerTargets[1]) > 0:
                    self.displacerTargets[1].append(self.displacerTargets[1][-1] + self.leftDisplacement * dispMagL)
                else:
                    self.displacerTargets[1].append(wLFingertip1 + self.leftDisplacement * dispMagL)
                self.displacerActual[1].append(wLFingertip2)
                if np.linalg.norm(self.leftDisplacement) > 0:
                    self.cumulativeMotionL += dispMagL
                    self.cumulativeAccurateMotionL += vecL.dot(self.leftDisplacement)
                else:
                    self.cumulativeFixedMotionL += dispMagR
                    self.cumulativeFixedTimeL += 1

            #compute changes in displacements before the next observation phase
            if self.rightDisplacer_active:
                if random.random() < self.displacementParameters[1]: #reset vector
                    if random.random() < self.displacementParameters[0]:
                        self.rightDisplacement = np.zeros(3)
                        self.displacer0TargetR = np.array(wRFingertip2)
                    else:
                        self.rightDisplacement = pyutils.sampleDirections(num=1)[0]
                elif random.random() < self.displacementParameters[2] and np.linalg.norm(self.rightDisplacement) > 0: #add noise to vector
                    noise = pyutils.sampleDirections(num=1)[0]
                    noise *= LERP(self.displacementParameters[3], self.displacementParameters[4], random.random())
                    self.rightDisplacement += noise
                    self.rightDisplacement /= np.linalg.norm(self.rightDisplacement)
            if self.leftDisplacer_active:
                if random.random() < self.displacementParameters[1]: #reset vector
                    if random.random() < self.displacementParameters[0]:
                        self.leftDisplacement = np.zeros(3)
                    else:
                        self.leftDisplacement = pyutils.sampleDirections(num=1)[0]
                elif random.random() < self.displacementParameters[2] and np.linalg.norm(self.leftDisplacement) > 0: #add noise to vector
                    noise = pyutils.sampleDirections(num=1)[0]
                    noise *= LERP(self.displacementParameters[3], self.displacementParameters[4], random.random())
                    self.leftDisplacement += noise
                    self.leftDisplacement /= np.linalg.norm(self.leftDisplacement)

            ob = self._get_obs()
            s = self.state_vector()
            if self.dotimings:
                self.addTiming(label="Termination")

            #update physx capsules
            self.updateClothCollisionStructures(hapticSensors=True)

            #check termination conditions
            topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
            bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
            bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))
            done = False
            if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
                print("Detecting potential instability")
                print(s)
                done = True
                self.reward -= 500
            elif not np.isfinite(s).all():
                print("Infinite value detected..." + str(s))
                print("reward: " + str(self.reward))
                print("prevT: " + str(self.prevTau))
                print("prevQ: " + str(self.prevPose))
                print("obs: " + str(ob))
                pyutils.saveList(self.qHistory,filename="qhistory", listoflists=True)
                pyutils.saveList(self.dqHistory, filename="dqhistory", listoflists=True)
                pyutils.saveList(self.tHistory, filename="thistory", listoflists=True)
                pyutils.saveList(self.rewardHistory, filename="rewardhistory", listoflists=False)
                done = True
                self.reward = -500
                ob = np.zeros(len(ob))
            elif self.deformationTermination and clothDeformation > self.maxDeformation:
                done = True
                self.reward -= 500
            elif self.collarTermination and self.simulateCloth:
                if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                    done = True
                    self.reward -= 500
            #increment the step counter
            self.numSteps += 1

            if self.dotimings:
                self.addTiming(label="End")
                self.printTiming()
                self.clearTimings()

            return ob, self.reward, done, {}
        except:
            print("step " + str(self.numSteps) + " failed")
            #self.step(action=np.zeros(len(a)))

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix,dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        fingertip = np.array([0.0, -0.06, 0.0])
        #vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
        #vec2 = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target2

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

        if self.rightDisplacer_active:
            obs = np.concatenate([obs, self.rightDisplacement]).ravel()
        if self.leftDisplacer_active:
            obs = np.concatenate([obs, self.leftDisplacement]).ravel()
        if self.rightTarget_active:
            shoulderR = self.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
            efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            obs = np.concatenate([obs, self.rightTarget-shoulderR, self.rightTarget-efR]).ravel()
        if self.leftTarget_active:
            shoulderL = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
            efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            obs = np.concatenate([obs, shoulderL-self.leftTarget, efL-self.leftTarget]).ravel()
        if self.featureInObs:
            centroid = self.CP0Feature.plane.org
            efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            disp = centroid-efR
            obs = np.concatenate([obs, centroid, disp]).ravel()
        if self.oracleInObs:
            oracle = np.zeros(3)
            if self.reset_number == 0:
                a=0 #nothing
            elif self.limbProgress > 0:
                oracle = self.CP0Feature.plane.normal
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=12,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                     returnOnlyGeo=False)
                if minGeoVix is None:
                    #oracle points to the garment when ef not in contact
                    efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
                    #closeVert = self.clothScene.getCloseVertex(p=efR)
                    #target = self.clothScene.getVertexPos(vid=closeVert)
                    centroid = self.CP0Feature.plane.org
                    target = centroid
                    vec = target - efR
                    oracle = vec/np.linalg.norm(vec)
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
        return obs

    def reset_model(self):
        try:
            #print("reset")
            self.cumulativeReward = 0
            self.clearTimings()
            self.qHistory = []
            self.dqHistory = []
            self.tHistory = []
            self.rewardHistory = []
            self.totalTime = 0
            self.cumulativeReward = 0
            self.dart_world.reset()
            self.clothScene.reset()
            self.clothScene.translateCloth(0, np.array([0.05, 0.025, 0]))
            #self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
            qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
            if self.resetRandomPose and self.resetFile is None:
                qpos = pyutils.getRandomPose(self.robot_skeleton)
            #qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-1.21, high=1.21, size=self.robot_skeleton.ndofs)
            #print(self.robot_skeleton.dq)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)

            #qpos = np.array([0.00984645029962, -0.00599959056884, -0.00671256780872, 0.00964348234246, 0.00328979646153, -0.00837484539897, -0.00392902285366, 0.00853631720415, 0.00648306871506, 0.00987675799087, 0.00065973832776, 0.0010888166833, -0.00496454405722, 0.400025413727, -1.35451513127, 0.521185386519, 1.74393040052, 0.341179954072, 0.29062079937, 0.00471862920899, -0.00625057092677, 0.00977505517246])
            #qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.193940077276, -0.0995533658543, -0.680941213889, -0.524547635822, 0.929809563123, 1.56740631418, 0.272488901324, 0.335129983119, 0.0, 0.0, 0.0])

            '''qpos = np.array([0.00210387423755, -0.00263432512464, -0.00608727794507, 0.00048188759912, -0.000790891203943,
                             0.00566944033496, -0.0062818525027, -0.139898427718, 2.89881383212, 0.542917778508,
                             0.593770039475, -0.196204829136, -0.0973580394792, -0.685992369829, -0.560684467346,
                             0.867649241514, 1.83056855945, 0.442013405854, 0.395205993027, -5.23263424256e-05,
                             0.000398533604261, -0.000139073022976])'''

            '''count = 0
            while self.dart_world.collision_result.num_contacts() > 0 or (count == 0 and self.simulateCloth):
                qpos = np.array([-0.0246826682648, 0.018944398895, -0.0134400014064, -0.0542085581649, -0.0124679311973, 0.136099614883, -0.149784150499, -0.238294428353, 2.73616801182, 0.000688467033079, 0.00286269170768, -0.196131932122, -0.100183323563, -0.667021683751, -0.551852317711, 0.876048055372, 1.82933888348, 0.442029727913, 0.395167537265, -0.00267152791664, -0.00169419615312, -8.29783655585e-05])
                #qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.195644348936, -0.098536397969, -0.681261731063, -0.553258926467, 0.869507748284, 1.82987701823, 0.442015199582, 0.395189219215, 0.0, 0.0, 0.0])
                #qpos = np.array([0.00210387423755, -0.00263432512464, -0.00608727794507, 0.00048188759912, -0.000790891203943, 0.00566944033496, -0.0062818525027, -0.139898427718, 2.89881383212, 0.542917778508, 0.593770039475, -0.196204829136, -0.0973580394792, -0.685992369829, -0.560684467346, 0.867649241514, 1.83056855945, 0.442013405854, 0.395205993027, -5.23263424256e-05, 0.000398533604261, -0.000139073022976])

                #qpos[7:9] += self.np_random.uniform(low=-1.0, high=1.0, size=2)
                #qpos[9:11] += self.np_random.uniform(low=-0.6, high=0.6, size=2)
                self.set_state(qpos, qvel)
                self.dart_world.check_collision()
                print(self.dart_world.collision_result.num_contacts())
                for b in self.dart_world.collision_result.contacted_bodies:
                    print(b)
                if count > 1:
                    print("Looping more than once: " + str(count))
                count += 1'''

            #if self.simulateCloth:
            qpos = np.array([-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004, -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334, -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637, 0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])


            #qpos = np.array([0.00789179104753, -0.00778707237558, -0.00575381599684, 0.0160314569834, -0.0215529394607, 0.00116310449511, 0.0141075139802, 0.0210178621179, -0.00535740130501, 0.0971533372584, -0.119663743335, -0.0831924264075, -0.107272709514, 0.343058669125, -0.929541335875, 0.572642382432, 1.96034075215, 0.0425568556954, 0.109450420964, -0.0104435230604, -0.0187997920078, -0.00127660576633])
            #qpos = np.array([0.0133847298842, 0.00623134347354, 0.00901967139099, 0.0197357484081, -0.0147453364021, -0.00214595430132, 0.0267835260132, -0.0122496485753, -0.0012161563398, 0.0614366033843, -0.166508814892, -0.109839479566, -0.0906491452046, 0.386626055839, -1.34713152342, 0.556287307477, 1.68271950222, 0.0305543921666, 0.6, -0.0163504623659, -0.0306551917448, 0.0135542729712])

            #qpos =np.array([0.2731471618,-0.3451429935,0.7688092455,-0.1597715944,0.1860187999,0.2504905961,0.042594533,1.0984740419,1.5495887065,-0.4358303746,-0.0589387622,0.2504341555,-0.2507510455,0.6146392381,-1.4901314838,-1.4741124414,1.1877315006,-0.6148575721,-0.2584124213,-0.0088981053,0.5894612576,0.7433916044])
            #qvel =np.array([0.0567240367,0.1688628579,1.2036850076,-1.7722794241,-1.7820411113,4.524206467,0.2259652913,3.8046292181,0.5771628785,5.825386422,2.767861514,3.28269589289E-009,-4.10147826813E-009,56.4937752114,-8.1748603334,-64.0955705123,-2.6189647867,-6.8044512247E-009,12.1735770456,-3.5029552058,-0.9008247279,0.6619836226])
            self.set_state(qpos, qvel)
            self.restPose = qpos

            if self.resetRandomPose:
                if self.resetFile is not None:
                    if len(self.ROMPoints) < 1:
                        self.ROMPoints = pyutils.loadListOfVecs(filename=self.resetFile)
                    ix = random.randint(0,len(self.ROMPoints)-1)
                    qpos = self.ROMPoints[ix]
                    self.set_state(qpos, qvel)
                    self.dart_world.check_collision()
                    while self.dart_world.collision_result.num_contacts() > 0:
                        ix = random.randint(0, len(self.ROMPoints) - 1)
                        qpos = self.ROMPoints[ix]
                        self.set_state(qpos, qvel)
                        self.dart_world.check_collision()
                    #self.dart_world.check_collision()
                else:
                    self.dart_world.check_collision()
                    while self.dart_world.collision_result.num_contacts() > 0:
                        qpos = pyutils.getRandomPose(self.robot_skeleton)
                        self.set_state(qpos, qvel)
                        self.dart_world.check_collision()
            #print(self.dart_world.collision_result.num_contacts())

            if random.random() < self.displacementParameters[0]:
                self.rightDisplacement = np.zeros(3)
                wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(np.array([0,-0.06,0]))
                self.displacer0TargetR = np.array(wRFingertip2)
            else:
                self.rightDisplacement = pyutils.sampleDirections(num=1)[0]
            if random.random() < self.displacementParameters[0]:
                self.leftDisplacement = np.zeros(3)
            else:
                self.leftDisplacement = pyutils.sampleDirections(num=1)[0]

            armLength = 0.75
            if self.rightTarget_active:
                self.rightTarget = self.robot_skeleton.bodynodes[4].to_world(np.zeros(3))+pyutils.sampleDirections(1)[0]*random.random()*armLength
            if self.leftTarget_active:
                self.leftTarget = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))+pyutils.sampleDirections(1)[0]*random.random()*armLength

            self.clothScene.setSelfCollisionDistance(0.025)

            if self.handleNode is not None:
                self.handleNode.clearHandles()
                self.handleNode.addVertices(verts=[570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55])
                self.handleNode.setOrgToCentroid()
                if self.updateHandleNodeFrom >= 0:
                    self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
                self.handleNode.recomputeOffsets()

            if self.simulateCloth:
                self.CP0Feature.fitPlane()
                self.collarFeature.fitPlane()
                if self.reset_number == 0:
                    self.separatedMesh.initSeparatedMeshGraph()
                    self.separatedMesh.updateWeights()
                    self.separatedMesh.computeGeodesic(feature=self.CP0Feature, oneSided=True, side=0, normalSide=0)

            #update physx capsules
            self.updateClothCollisionStructures(hapticSensors=True)
            self.clothScene.clearInterpolation()

            #debugging
            self.reset_number += 1
            self.numSteps = 0

            if self.limbProgressReward:
                self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR,offset=np.array([0,-0.06,0])), feature=self.CP0Feature)

            if self.torqueGraph is not None:
                xdata = np.arange(400)
                self.torqueGraph.xdata = xdata
                initialYData0 = np.zeros(400)
                initialYData1 = np.zeros(400)
                self.torqueGraph.plotData(ydata=initialYData0)
                self.torqueGraph.plotData(ydata=initialYData1)

            self.displacerTargets = [[], []]
            self.displacerActual = [[], []]
            #self.cumulativeMotionR = 0.00001
            #self.cumulativeAccurateMotionR = 0
            #self.cumulativeMotionL = 0.00001
            #self.cumulativeAccurateMotionL = 0

            if self.recordROMPoints:
                if len(self.ROMPoints) > 1:
                    pyutils.saveList(self.ROMPoints, filename="ROMPoints", listoflists=True)

            #print("done reset")
            return self._get_obs()
        except:
            print("Failed on reset " + str(self.reset_number))

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a=0
        #collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[14].to_world(np.array([0,0.175,0]))
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
        csVars0 = np.array([0.15, -1, -1, 0,0,0])
        csVars1 = np.array([0.07, -1, -1, 0,0,0])
        csVars2 = np.array([0.1, -1, -1, 0,0,0])
        csVars3 = np.array([0.1, -1, -1, 0,0,0])
        csVars4 = np.array([0.065, -1, -1, 0,0,0])
        csVars5 = np.array([0.05, -1, -1, 0,0,0])
        csVars6 = np.array([0.0365, -1, -1, 0,0,0])
        csVars7 = np.array([0.04, -1, -1, 0,0,0])
        csVars8 = np.array([0.046, -1, -1, 0,0,0])
        csVars9 = np.array([0.065, -1, -1, 0,0,0])
        csVars10 = np.array([0.05, -1, -1, 0,0,0])
        csVars11 = np.array([0.0365, -1, -1, 0,0,0])
        csVars12 = np.array([0.04, -1, -1, 0,0,0])
        csVars13 = np.array([0.046, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7, csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13, csVars13]).ravel()
        self.collisionSphereInfo = np.array(collisionSpheresInfo)
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        if np.isnan(np.sum(collisionSpheresInfo)): #this will keep nans from propagating into PhysX resulting in segfault on reset()
            return
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((14,14))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[1,4] = 1
            collisionCapsuleInfo[1,9] = 1
            collisionCapsuleInfo[2,3] = 1
            collisionCapsuleInfo[4,5] = 1
            collisionCapsuleInfo[5,6] = 1
            collisionCapsuleInfo[6,7] = 1
            collisionCapsuleInfo[7,8] = 1
            collisionCapsuleInfo[9,10] = 1
            collisionCapsuleInfo[10,11] = 1
            collisionCapsuleInfo[11,12] = 1
            collisionCapsuleInfo[12,13] = 1
            collisionCapsuleBodynodes = -1 * np.ones((14,14))
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
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            hapticSensorRadii = np.array([csVars0[0], csVars1[0], csVars2[0], csVars3[0], csVars4[0], LERP(csVars4[0], csVars5[0], 0.33), LERP(csVars4[0], csVars5[0], 0.66), csVars5[0], LERP(csVars5[0], csVars6[0], 0.33), LERP(csVars5[0], csVars6[0], 0.66), csVars6[0], csVars7[0], csVars8[0], csVars9[0], LERP(csVars9[0], csVars10[0], 0.33), LERP(csVars9[0], csVars10[0], 0.66), csVars10[0], LERP(csVars10[0], csVars11[0], 0.33), LERP(csVars10[0], csVars11[0], 0.66), csVars11[0], csVars12[0], csVars13[0]])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            self.clothScene.setHapticSensorRadii(hapticSensorRadii)

    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None, resetFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc, self.reset_model)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        #testing elbow flair reward
        '''root = self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
        spine = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
        elbow = self.robot_skeleton.bodynodes[self.elbowFlairNode].to_world(np.zeros(3))
        dist = pyutils.distToLine(p=elbow, l0=root, l1=spine)
        z = 0.5
        s = 16
        l = 0.2
        reward_elbow_flair = (1 - (z * math.tanh(s * (dist - l)) + z))
        #print("reward_elbow_flair: " + str(reward_elbow_flair))
        renderUtils.drawLineStrip(points=[root, elbow, spine])'''

        reward_oracleDisplacement = 0
        if self.oracleDisplacementReward and np.linalg.norm(self.prevOracle) > 0:
            efR = self.robot_skeleton.bodynodes[7].to_world(np.array([0,-0.06,0]))
            actual_displacement = self.dispR
            reward_oracleDisplacement += actual_displacement.dot(self.prevOracle)
            #print(reward_oracleDisplacement)
            renderUtils.setColor([1.0, 0, 0])
            if reward_oracleDisplacement > 0:
                renderUtils.setColor([0,1.0,0])
            renderUtils.drawArrow(p0=efR, p1=efR + actual_displacement/np.linalg.norm(actual_displacement)*0.15, hwRatio=0.15)


        #HSL = self.clothScene.getHapticSensorLocations()
        #renderUtils.drawSphere(pos=HSL[12*3:13*3],rad=0.1)

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[5].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[5].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[14].to_world(np.array([0.0, 0, -0.105])), self.robot_skeleton.bodynodes[14].to_world(np.array([0.0, 0.2, -0.105]))])

        #render sample range
        #renderUtils.drawSphere(pos=self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)), rad=0.75, solid=False)

        if self.CP0Feature is not None:
            self.CP0Feature.drawProjectionPoly()
        if self.collarFeature is not None:
            self.collarFeature.drawProjectionPoly()

        #render geodesic
        '''
        for v in range(self.clothScene.getNumVertices()):
            side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
            side0geo = self.separatedMesh.nodes[v].geodesic

            pos = self.clothScene.getVertexPos(vid=v)
            norm = self.clothScene.getVertNormal(vid=v)
            renderUtils.setColor(color=[0.0, 1.0 - (side0geo/self.separatedMesh.maxGeo), 0.0])
            renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
            renderUtils.setColor(color=[0.0, 1.0 - (side1geo / self.separatedMesh.maxGeo), 0.0])
            renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)
        '''

        if self.renderContactInfo and self.simulateCloth:
            contactInfo = pyutils.getContactIXGeoSide(sensorix=12, clothscene=self.clothScene,meshgraph=self.separatedMesh)
            renderUtils.drawContactInfo(sensorix=12, contactInfo=contactInfo, clothScene=self.clothScene, maxGeo=self.separatedMesh.maxGeo, viewer=self.viewer)
            renderUtils.drawHeatMapBar(topLeft=[self.viewer.viewport[2]-300,320], h=20, w=300)
            points2D = []
            for c in contactInfo:
                x = LERP(p0=self.viewer.viewport[2], p1=self.viewer.viewport[2]-300, t=c[1]/self.separatedMesh.maxGeo)
                points2D.append(np.array([x, 318]))
                points2D.append(np.array([x, 302]))
            renderUtils.drawLines2D(points=points2D, color=np.array([0.3,0.3,0.3]))
            renderUtils.drawText(x=self.viewer.viewport[2]-300, y=335,text=str(self.separatedMesh.maxGeo) + " <= geodesic <= 0")

        textHeight = 15
        textLines = 2

        #render targets
        if self.rightTarget_active:
            renderUtils.setColor(color=[1.0,1.0,0])
            renderUtils.drawSphere(pos=self.rightTarget, rad=0.05)
            renderUtils.drawLineStrip(points=[self.rightTarget, self.robot_skeleton.bodynodes[7].to_world(np.array([0,-0.06,0]))])
            renderUtils.drawLineStrip(points=[self.rightTarget, self.robot_skeleton.bodynodes[4].to_world(np.zeros(3))])
        if self.leftTarget_active:
            renderUtils.setColor(color=[0.0, 1.0, 1.0])
            renderUtils.drawSphere(pos=self.leftTarget, rad=0.05)
            renderUtils.drawLineStrip(points=[self.leftTarget, self.robot_skeleton.bodynodes[12].to_world(np.array([0, -0.06, 0]))])

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            if self.totalTime > 0:
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps) + " framerate = " + str(self.numSteps/self.totalTime), color=(0., 0, 0))
                textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            if self.simulateCloth:
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Deformation = " + str(self.deformation), color=(0., 0, 0))
                textLines += 1
            if self.contactGeoReward:
                self.clothScene.drawText(x=15., y=textLines * textHeight, text="Min Contact Geo = " + str(self.minContactGeo), color=(0., 0, 0))
                textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])

            self.dart_world.check_collision()
            self.clothScene.drawText(x=15., y=textLines * textHeight,text="Num Contacts = " + str(self.dart_world.collision_result.num_contacts()), color=(0., 0, 0))
            textLines += 1

        if self.rightDisplacer_active:
            renderUtils.setColor(color=[0.0,0.0,1.0])
            ef = self.robot_skeleton.bodynodes[7].to_world(np.array([0.0, -0.06, 0.0]))
            if np.linalg.norm(self.rightDisplacement) == 0:
                renderUtils.drawBox(cen=ef,dim=[0.2,0.2,0.2], fill=False)
            else:
                renderUtils.drawArrow(p0=ef, p1=ef+self.rightDisplacement*0.15, hwRatio=0.15)
            if self.renderDisplacerAccuracy:
                renderUtils.setColor(color=[1.0,0,0])
                renderUtils.drawLineStrip(self.displacerTargets[0])
                renderUtils.setColor(color=[0.0, 0, 1.0])
                renderUtils.drawLineStrip(self.displacerActual[0])
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Disp. Acc. R = " + str(self.cumulativeAccurateMotionR/self.cumulativeMotionR), color=(0., 0, 0))
                textLines += 1
                if self.cumulativeFixedTimeR > 0:
                    self.clothScene.drawText(x=15., y=textLines*textHeight, text="Fixed Motion R = " + str(self.cumulativeFixedMotionR/self.cumulativeFixedTimeR), color=(0., 0, 0))
                    textLines += 1
        if self.leftDisplacer_active:
            renderUtils.setColor(color=[0.0, 0.0, 1.0])
            ef = self.robot_skeleton.bodynodes[12].to_world(np.array([0.0, -0.06, 0.0]))
            if np.linalg.norm(self.leftDisplacement) == 0:
                renderUtils.drawBox(cen=ef, dim=[0.2, 0.2, 0.2], fill=False)
            else:
                renderUtils.drawArrow(p0=ef, p1=ef+self.leftDisplacement*0.15, hwRatio=0.15)
            if self.renderDisplacerAccuracy:
                renderUtils.setColor(color=[1.0,1.0,0])
                renderUtils.drawLineStrip(self.displacerTargets[1])
                renderUtils.setColor(color=[0.0, 1.0, 1.0])
                renderUtils.drawLineStrip(self.displacerActual[1])
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Disp. Acc. L = " + str(self.cumulativeAccurateMotionL / self.cumulativeMotionL), color=(0., 0, 0))
                textLines += 1
                if self.cumulativeFixedTimeL > 0:
                    self.clothScene.drawText(x=15., y=textLines*textHeight, text="Fixed Motion L = " + str(self.cumulativeFixedMotionL / self.cumulativeFixedTimeL), color=(0., 0, 0))
                    textLines += 1

        if self.oracleInObs:
            renderUtils.setColor(color=[1.0, 1.0, 0])
            ef = self.robot_skeleton.bodynodes[7].to_world(np.array([0.0, -0.06, 0.0]))
            renderUtils.drawArrow(p0=ef, p1=ef + self.prevOracle * 0.15, hwRatio=0.15)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
        self.track_skeleton_id = 0
        if not self.renderDARTWorld:
            self.viewer.renderWorld = False
        self.clothScene.renderCollisionCaps = True
        self.clothScene.renderCollisionSpheres = True




