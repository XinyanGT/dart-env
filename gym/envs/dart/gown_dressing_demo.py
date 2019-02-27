# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
import quaternion
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.clothHandles import *
from pyPhysX.clothfeature import *
import pyPhysX.meshgraph as meshgraph

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

''' This env is setup for upper body interaction with gown garment gripped and moved on target path'''

class DartClothGownDemoEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.prefix = os.path.dirname(__file__)
        self.useOpenGL = True
        #self.screenSize = (540, 720)
        self.screenSize = (1080,720)
        self.renderDARTWorld = False
        self.renderUI = True
        self.drawDebuggingBoxes = False
        self.target = np.array([0.8, -0.6, 0.6])
        self.targetInObs = True
        self.geoVecInObs = True
        self.contactIDInObs = True
        self.hapticsInObs = True
        self.hapticsAware = True #if false, 0's for haptic input
        self.pureProprioceptionObs = False
        self.arm = 2

        self.renderObs = False

        self.deformationTerm = False
        self.deformationTermLimit = 40
        self.deformationPenalty = True
        self.maxDeformation = 0.0

        self.gripperCover = True
        self.renderCover = True
        self.renderGripPath = False

        self.reward = 0
        self.prevAction = None

        self.graphDataOutputPrefix = "/home/alexander/Documents/graphoutput/"
        self.graphArmProgress = False
        self.graphDeformation = False
        self.graphArmHaptics = False
        self.armHapticsGraphs = []
        self.armHapticsIndices = [13,14,15,16,17,18,19,20,21] #only haptic sensors ID'd here will be graphed
        self.armProgressGraph = None
        self.deformationGraph = None
        self.regraphPeriod = 399
        self.colorFromDistribution = True #if true, draw graph color based on distribution point
        self.renderZoneColorFromDistribution = False #if true, sample the linear target range and color spheres as map
        self.domainTesting = False #if true, sample gown locations from a grid instead of random
        self.jitteredSamples = True #True if using jittered sampling inside the grid instead of grid corners
        self.twoDistributions = True #True if domain testing on initial and end distributions
        #self.domainTestingDim = np.array([4, 4, 1])
        #self.domainTestingDim = np.array([2, 2, 1])
        self.domainTestingDim = np.array([1, 2, 2]) #side linear range

        self.numRollouts = 100 #terminate after this many resets and save the graphs (if -1, don't terminate)
        #if self.domainTesting:
        #    self.numRollouts = self.domainTestingDim[0]*self.domainTestingDim[1]*self.domainTestingDim[2]
        self.numRollouts = -1

        self.renderDomainTestingResults = False  # if true, sample the linear target range and color spheres as map

        #self.progressHistogram = pyutils.Histogram2D()


        #22 dof upper body
        self.action_scale = np.ones(22)*10
        self.control_bounds = np.array([np.ones(22), np.ones(22)*-1])

        if self.arm > 0:
            self.action_scale = np.ones(11) * 10
            #self.action_scale[0] *= 2.0 #increase torso strength only
            #self.action_scale[1] *= 2.0
            self.control_bounds = np.array([np.ones(11), np.ones(11) * -1])

            '''self.action_scale[0] = 25  # torso
            self.action_scale[1] = 25
            self.action_scale[2] = 10  # spine
            self.action_scale[3] = 10  # clav
            self.action_scale[4] = 10
            self.action_scale[5] = 10  # shoulder
            self.action_scale[6] = 10
            self.action_scale[7] = 8
            self.action_scale[8] = 8  # elbow
            self.action_scale[9] = 6  # wrist
            self.action_scale[10] = 6'''

        '''self.action_scale[0] = 150  # torso
        self.action_scale[1] = 150
        self.action_scale[2] = 100  # spine
        self.action_scale[3] = 50  # clav
        self.action_scale[4] = 50
        self.action_scale[5] = 30  # shoulder
        self.action_scale[6] = 30
        self.action_scale[7] = 20
        self.action_scale[8] = 20  # elbow
        self.action_scale[9] = 8  # wrist
        self.action_scale[10] = 8'''

        self.numSteps = 0 #increments every step, 0 on reset

        self.arm_progress = 0.  # set in step when first queried
        self.armLength = -1.0  # set when arm progress is queried

        #set these in reset
        self.q_target = None
        self.q_weight = None
        self.q_target_reward = 0
        self.q_target_reward_active = True #set this here

        # handle node setup
        self.handleNode = None
        self.gripper = None

        #interactive handle mode
        self.interactiveHandleNode = False

        #randomized spline target mode
        self.randomHandleTargetSpline = False
        self.handleTargetSplineWindow = 10.0 #time window for the full motion (split into equal intervals b/t CPs)
        self.numHandleTargetSplinePoints = 4
        self.handleTargetSplineGlobalBounds = [np.array([0.75,0.3,1.0]), np.array([-0.0,-0.5,0.])] #total drift allowed from origin for target orgs
        self.handleTargetSplineLocalBounds = [np.array([0.25,0.25,0.35]), np.array([-0.25,-0.25,-0.05])] #cartesian drift allowed b/t neighboring CPs
        #TODO: add rotation
        #self.handleTargetSplineGlobalRotationBounds

        #linear spline target mode
        self.handleTargetLinearMode = 7 # 1 is linear, 2 is small range, 3 is larger range, 4 is new static, 5 is new small linear, 6 is beside, 7 large static range with increase min y
        self.randomHandleTargetLinear = True
        self.linearTargetFixed = True #if true, end point is start point
        self.orientationFromSpline = False #if true, the gripper orientation is changed to match the spline direction (y rotation only)
        self.handleTargetLinearWindow = 10.0
        self.handleTargetLinearInitialRange = None
        self.handleTargetLinearEndRange = None
        self.targetSeeds = []
        #self.targetSeeds = [0, 15, 20, 21, 22, 23, 24] #3
        #self.targetSeeds = [0, 2, 3, 4, 9] #2
        #self.targetSeeds = [0, 5, 6, 8, 12, 17, 19] #1
        #self.targetSeeds = np.arange(30,60)

        self.debuggingBoxes = []

        if self.handleTargetLinearMode == 1: #original fast linear
            # old linear track
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.7,0.5,0.15]),
                                                                   c1=np.array([-0.3, -0.5, -0.15]),
                                                                   org=np.array([-0.17205264,  0.12056234, -1.07377446]))
            self.handleTargetLinearEndRange = pyutils.BoxFrame(c0=np.array([0.5, 0.3, 0.2]),
                                                               c1=np.array([0.1, -0.1, -0.1]),
                                                               org=np.array([0.,0.,0.]))
        elif self.handleTargetLinearMode == 2: #original small range
            # small distribution
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            self.handleTargetLinearEndRange = self.handleTargetLinearInitialRange

        elif self.handleTargetLinearMode == 3: #original larger range
            #slightly larger distribution
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.25, 0.2, 0.15]),
                                                                   c1=np.array([-0.35, -0.25, -0.15]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            smallhandleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            self.debuggingBoxes.append(smallhandleTargetLinearInitialRange)
            self.handleTargetLinearEndRange = self.handleTargetLinearInitialRange

        elif self.handleTargetLinearMode == 4: #new larger static range
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.25, 0.2, 0.05]),
                                                                   c1=np.array([-0.35, -0.25, -0.05]),
                                                                   org=np.array([0.27205264, 0.052056234, -0.37377446]))
            smallhandleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            self.debuggingBoxes.append(smallhandleTargetLinearInitialRange)
            self.handleTargetLinearEndRange = self.handleTargetLinearInitialRange
        elif self.handleTargetLinearMode == 5: #close linear range
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.2, 0.25, 0.05]),
                                                                   c1=np.array([-0.2, -0.2, -0.05]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.57377446]))
            self.handleTargetLinearEndRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, 0.0]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            smallhandleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            #self.debuggingBoxes.append(smallhandleTargetLinearInitialRange)
        elif self.handleTargetLinearMode == 6: #linear range beside the character
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.05, 0.25, 0.25]),
                                                                   c1=np.array([-0.05, -0.2, -0.25]),
                                                                   org=np.array([0.77205264, 0.052056234, -0.27377446]))
            self.handleTargetLinearEndRange = pyutils.BoxFrame(c0=np.array([0.05, 0.15, 0.15]),
                                                                   c1=np.array([-0.05, -0.15, -0.15]),
                                                                   org=np.array([0.36205264, 0.052056234, -0.17377446]))
            smallhandleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            self.debuggingBoxes.append(smallhandleTargetLinearInitialRange)
        elif self.handleTargetLinearMode == 7: #large static range with increase min y
            self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.25, 0.2, 0.05]),
                                                                   c1=np.array([-0.35, -0.15, -0.05]),
                                                                   org=np.array([0.27205264, 0.052056234, -0.37377446]))
            smallhandleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.15, 0.15, 0.1]),
                                                                   c1=np.array([-0.15, -0.15, -0.1]),
                                                                   org=np.array([0.17205264, 0.052056234, -0.37377446]))
            #self.debuggingBoxes.append(smallhandleTargetLinearInitialRange)
            self.handleTargetLinearEndRange = self.handleTargetLinearInitialRange

        #debugging boxes for visualizing distributions
        self.debuggingBoxes.append(self.handleTargetLinearInitialRange)
        self.debuggingBoxes.append(self.handleTargetLinearEndRange)
        self.debuggingColors = [[0., 1, 0], [0, 0, 1.], [1., 0, 0], [1., 1., 0], [1., 0., 1.], [0, 1., 1.]]

        self.reset_number = 0  # increments on env.reset()

        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01,
                                        mesh_path = self.prefix + "/assets/fullgown1.obj",
                                        #mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/fullgown1.obj",
                                        #mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/tshirt_m.obj",
                                        #state_path="/home/alexander/Documents/dev/1stSleeveState.obj",
                                        state_path=self.prefix + "/../../../../hanginggown.obj",
                                        scale=1.3)

        clothScene.togglePinned(0,0) #turn off auto-pin
        #clothScene.togglePinned(0, 144)
        #clothScene.togglePinned(0, 190)

        self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)

        #abridged feature
        #self.CP0Feature = ClothFeature(verts=[475, 860, 1620, 1839, 994, 469, 153, 531, 1932, 140],
        #                               clothScene=clothScene)
        #full feature
        self.CP0Feature = ClothFeature(
            verts=[413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531],
            clothScene=clothScene)
        self.armLength = -1.0  # set when arm progress is queried

        observation_size = 66  # pose(sin,cos), pose vel, haptics
        if self.pureProprioceptionObs:
            observation_size = 44

        if self.hapticsInObs:
            observation_size += 66
        if self.targetInObs:
            observation_size += 6  # target reaching
        if self.contactIDInObs:
            observation_size += 22
        if self.geoVecInObs:
            observation_size += 3

        self.hapticSensorNodes = []
        self.hapticSensorOffsets = []

        self.prevTime = time.time()
        self.totalTime = 0

        #intialize the parent env
        model_path = 'UpperBodyCapsules_collisiontest.skel'
        if self.gripperCover:
            model_path = 'UpperBodyCapsules_gripper.skel'
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds, screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size,
                                  action_bounds=self.control_bounds, disableViewer=True, visualize=False)
        utils.EzPickle.__init__(self)

        #eanble pydart2 collision
        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        #setup pydart collision filter
        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[4],
                                           self.robot_skeleton.bodynodes[6]) #right forearm to upper-arm
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                           self.robot_skeleton.bodynodes[12]) #left forearm to upper-arm
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[12],
                                           self.robot_skeleton.bodynodes[14])  # left forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[6],
                                           self.robot_skeleton.bodynodes[8])  # right forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[15])  # torso to neck
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[3])  # torso to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[9])  # torso to left shoulder


        #setup HandleNode here
        self.handleNode = HandleNode(self.clothScene, org=np.array([0.05,0.034,-0.975]))
        if self.interactiveHandleNode:
            self.viewer.interactors[2].frame.org = self.handleNode.org
            self.viewer.interactors[2].frame.orienation = self.handleNode.orientation
            self.viewer.interactors[2].frame.updateQuaternion()

        #self.handle_node.addVertex(0)
        #self.handle_node.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])

        #self.gripper = pyutils.BoxFrame(c0=np.array([0.06, -0.075, 0.06]), c1=np.array([-0.06, -0.125, -0.06]))
        #self.gripper = pyutils.EllipsoidFrame(c0=np.array([0,-0.1,0]), dim=np.array([0.05,0.025,0.05]))
        #self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)
        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.4)
        
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = True
        
        self.renderDofs = True #if true, show dofs text 
        self.renderForceText = False


        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])

        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

        if self.graphArmProgress:
            self.armProgressGraph = pyutils.LineGrapher(title="Arm Progress")
        if self.graphDeformation:
            self.deformationGraph = pyutils.LineGrapher(title="Deformation")
        if self.graphArmHaptics:
            for ix in self.armHapticsIndices:
                self.armHapticsGraphs.append(pyutils.LineGrapher(title="Haptic Sensor " + str(ix)))


        print("done init")
        #print("random = " + str(random.randint(1, 100)))
        #print(self.getFile())

    def _getFile(self):
        return __file__

    def limits(self, dof_ix):
        return np.array([self.robot_skeleton.dof(dof_ix).position_lower_limit(), self.robot_skeleton.dof(dof_ix).position_upper_limit()])

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)
        
    def loadObjState(self):
        self.clothScene.loadObjState("objState", 0)

    def _step(self, a):
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
        self.prevAction = np.array(clamped_control)
        tau = np.multiply(clamped_control, self.action_scale)


        if self.handleNode is not None:
            #self.handle_node.setTranslation(T=self.viewer.interactors[2].frame.org)
            if self.interactiveHandleNode:
                self.handleNode.org = self.viewer.interactors[2].frame.org
                self.handleNode.setOrientation(R=self.viewer.interactors[2].frame.orientation)
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.step()

        if self.gripperCover and self.handleNode is not None:
            #self.dart_world.skeletons[1].q = [0, 0, 0, self.CP0Feature.plane.org[0], self.CP0Feature.plane.org[1], self.CP0Feature.plane.org[2]]
            self.dart_world.skeletons[1].q = [0, 0, 0, self.handleNode.org[0], self.handleNode.org[1], self.handleNode.org[2]]

        #if self.gripper is not None:
        #    self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        #increment self collision distance test
        #currentDistance = self.clothScene.getSelfCollisionDistance()
        #print("current self-collision distance = " + str(currentDistance))
        #self.clothScene.setSelfCollisionDistance(currentDistance + 0.0001)

        #apply action and simulate
        # apply action and simulate
        if self.arm == 1:
            tau = np.concatenate([tau, np.zeros(11)])
        elif self.arm == 2:
            tau = np.concatenate([tau[:3], np.zeros(8), tau[3:], np.zeros(3)])
        self.do_simulation(tau, self.frame_skip)

        self.target = pyutils.getVertCentroid(self.CP0Feature.verts, self.clothScene)
        if self.renderObs is False:
            self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]
        #self.dart_world.skeletons[0].q = [0, 0, 0, 0, 0, 0]

        self.CP0Feature.fitPlane()


        reward = 0
        self.arm_progress = self.armSleeveProgress() / self.armLength

        minContactGeodesic = None
        if self.numSteps > 0:
            minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene,
                                                           meshgraph=self.separatedMesh)
        contactGeoReward = 0
        if self.arm_progress > 0:
            contactGeoReward = 1.0
        elif minContactGeodesic is not None:
            contactGeoReward = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo

        ob = self._get_obs()
        s = self.state_vector()

        self.q_target_reward = 0
        if self.q_target_reward_active and self.numSteps > 0:
            self.q_target_reward = np.linalg.norm((self.robot_skeleton.q-self.q_target)*self.q_weight)

        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        if clothDeformation > self.maxDeformation:
            self.maxDeformation = clothDeformation

        clothDeformationReward = 0
        if clothDeformation > 15 and self.deformationPenalty is True:
            #clothDeformationReward = 15.0-clothDeformation
            clothDeformationReward = (math.tanh(9.24-0.5*clothDeformation)-1)/2.0 #near 0 at 15, ramps up to -1.0 at ~22 and remains constant

        #torque penalty
        torquePenalty = 0
        torquePenalty = -np.linalg.norm(tau)

        reward += self.arm_progress*5 + contactGeoReward*2 - self.q_target_reward + clothDeformationReward*6# + torquePenalty*0.1
        self.reward = reward
        #print("reward = " + str(reward))
        
        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        
        #check termination conditions
        done = False


        if not np.isfinite(s).all():
            #print("Infinite value detected..." + str(s))
            done = True
            reward -= 500
        elif (clothDeformation > self.deformationTermLimit and self.deformationTerm is True):
            #print("Deformation Termination")
            done = True
            reward -= 5000
        #elif self.armLength > 0 and self.arm_progress >= 0.95:
        #    done=True
        #    reward = 1000
        #    print("Dressing completed!")

        self.numSteps += 1

        if self.graphArmProgress and self.armProgressGraph is not None:
            #self.armProgressGraph.addToLinePlot(data=[[self.arm_progress]])
            self.armProgressGraph.yData[self.reset_number-1][self.numSteps-1] = self.arm_progress
            if self.numSteps % self.regraphPeriod == 0:
                self.armProgressGraph.update()
        if self.graphDeformation and self.deformationGraph is not None:
            #if clothDeformation > 50:
            #    clothDeformation = 50
            self.deformationGraph.yData[self.reset_number - 1][self.numSteps - 1] = clothDeformation
            if self.numSteps % self.regraphPeriod == 0:
                self.deformationGraph.update()
            #self.deformationGraph.addToLinePlot(data=[[clothDeformation]])
        if self.graphArmHaptics and len(self.armHapticsGraphs) > 0:
            f = self.clothScene.getHapticSensorObs()
            for i,ix in enumerate(self.armHapticsIndices):
                self.armHapticsGraphs[i].yData[self.reset_number - 1][self.numSteps - 1] = np.linalg.norm(f[ix*3:ix*3+3])
                if self.numSteps % self.regraphPeriod == 0:
                    self.armHapticsGraphs[i].update()

        #if self.numSteps >= 400 or done is True:
        #    print("arm_progress = " + str(self.arm_progress) + " | maxDeformation = " + str(self.maxDeformation))

        return ob, reward, done, {}

    def _get_obs(self):
        '''get_obs'''
        f_size = 66
        theta = self.robot_skeleton.q
        efnodeix = 8
        if self.arm == 2:
            efnodeix = 14

        if self.simulateCloth is True and self.hapticsAware is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        else:
            f = np.zeros(f_size)

        obs = np.array([])
        if self.pureProprioceptionObs:
            obs = np.concatenate([theta, self.robot_skeleton.dq]).ravel()
        else:
            obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq]).ravel()

        if self.targetInObs:
            fingertip = np.array([0.0, -0.06, 0.0])
            vec = self.robot_skeleton.bodynodes[efnodeix].to_world(fingertip) - self.target
            obs = np.concatenate([obs, vec, self.target]).ravel()

        if self.geoVecInObs:
            #print("Getting geoVec Obs...")
            if self.reset_number == 0:
                obs = np.concatenate([obs, np.zeros(3)]).ravel()
            elif self.arm_progress > 0:
                #print("plane mode")
                obs = np.concatenate([obs, self.CP0Feature.plane.normal]).ravel()
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=21,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                     returnOnlyGeo=False)
                if minGeoVix is None:
                    # obs = np.concatenate([obs, np.zeros(3)]).ravel()
                    #no contact points: ef toward the sleeve target (added 09/08/2017)
                    fingertip = np.array([0.0, -0.06, 0.0])
                    vec = self.target - self.robot_skeleton.bodynodes[efnodeix].to_world(fingertip)
                    vec = vec / np.linalg.norm(vec)
                    obs = np.concatenate([obs, vec]).ravel()
                else:
                    vixSide = 0
                    if _side:
                        vixSide=1
                    if minGeoVix >= 0:
                        geoVec = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
                        obs = np.concatenate([obs, geoVec]).ravel()
                        #print("geoVec: " + str(geoVec))
                    else:
                        print("error to be here")

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        if self.hapticsInObs:
            obs = np.concatenate([obs, f * 3.]).ravel()
        #obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq, vec, self.target, f]).ravel()
        #obs = np.concatenate([theta, self.robot_skeleton.dq, f]).ravel()
        return obs

    def reset_model(self):
        '''reset_model'''
        self.totalTime = 0
        self.numSteps = 0
        self.maxDeformation = 0.0
        self.dart_world.reset()
        self.clothScene.reset()
        self.clothScene.setSelfCollisionDistance(0.03)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)

        #terminate after predesignated # of rollouts
        if self.numRollouts >= 0:
            if self.numRollouts <= self.reset_number:
                if self.graphArmProgress:
                    self.armProgressGraph.save(filename=self.graphDataOutputPrefix+"armprogress.png", datafile=self.graphDataOutputPrefix+"armprogress.txt")
                if self.graphDeformation:
                    self.deformationGraph.save(filename=self.graphDataOutputPrefix+"deformation.png", datafile=self.graphDataOutputPrefix+"deformation.txt")
                if self.graphArmHaptics:
                    for i,ix in enumerate(self.armHapticsIndices):
                        self.armHapticsGraphs[i].save(filename=self.graphDataOutputPrefix+"haptics"+str(ix)+".png", datafile=self.graphDataOutputPrefix+"haptics"+str(ix)+".txt")
                '''dataTest = pyutils.loadData2D(filename="/home/alexander/armprogress.txt")
                print("dataTest = " + str(dataTest))
                print("||dataTest|| = " + str(len(dataTest)))
                for ix,i in enumerate(dataTest):
                    print(" ||"+str(ix)+"||: " + str(len(i)))
                sortedIXs = pyutils.getListsIXByLastElement(dataTest)
                print("sortedIXs: " + str(sortedIXs))
                for ix in sortedIXs:
                    print(" ix " + str(ix) + ": " + str(dataTest[ix][-1]))'''
                exit()

        qpos = [-0.0678033793307, 0.0460394372646, -0.0228567701463, 0.0164748821823, -0.0111482825353, -0.2088004225982,
         -0.00116452660407, -0.00536063771987, 0.0106001520861, 0.00893180602343, 0.000975322470016, 0.00297590969194,
         -0.0135842975992, -0.0107381796688, -0.805728339233, 1.44280155211, 2.65716610139, -0.00193051041281,
         -0.00455716796044, -0.0229811572279, -0.00388094984923, -0.0132110585751] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.025, high=.025, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.q_target = np.array(qpos)

        self.q_weight = np.zeros(len(self.q_target)) #no penalty for pose
        self.q_weight[0] = 1.0 #torso only
        self.q_weight[1] = 1.0

        #Usefull to position the hanging gown if no state_path is given...
        '''self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 0, 1.])))
        self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 1., 0.])))
        self.clothScene.translateCloth(0, np.array([0.75, -0.75, -0.75]))  # shirt in front of person'''

        self.clothScene.translateCloth(0, np.array([0.25, 0., 0.]))

        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))
        
        #load cloth state from ~/Documents/dev/objFile.obj
        #self.clothScene.loadObjState()

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)

        self.handleNode.clearHandles()
        #self.handle_node.addVertex(vid=0)
        #self.clothScene.setPinned(cid=0, vid=0)

        #self.clothScene.refreshMotionConstraints()
        #self.clothScene.refreshCloth()
        #self.clothScene.clearInterpolation()

        self.handleNode.clearHandles()
        self.handleNode.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])
        self.handleNode.setOrgToCentroid()
        if self.gripperCover and self.handleNode is not None:
            self.dart_world.skeletons[1].q = [0, 0, 0, self.handleNode.org[0], self.handleNode.org[1], self.handleNode.org[2]]
        #print("org = " + str(self.handle_node.org))
        if self.interactiveHandleNode:
            self.handleNode.usingTargets = False
            self.viewer.interactors[2].frame.org = self.handleNode.org
        elif self.randomHandleTargetSpline:
            self.handleNode.usingTargets = True
            self.handleNode.clearTargetSpline()
            dt = self.handleTargetSplineWindow / self.numHandleTargetSplinePoints
            #debugging
            self.debuggingBoxes.clear()
            self.debuggingBoxes.append(pyutils.BoxFrame(c0=self.handleTargetSplineGlobalBounds[0], c1=self.handleTargetSplineGlobalBounds[1], org=self.handleNode.org))
            #end debugging
            #print("org = " + str(self.handle_node.org))
            for i in range(self.numHandleTargetSplinePoints):
                t = dt + dt*i
                pos = self.handleNode.targetSpline.pos(t=t)
                localDriftRange = np.array(self.handleTargetSplineLocalBounds)
                localDriftRange[0] = np.minimum(localDriftRange[0], self.handleTargetSplineGlobalBounds[0]+self.handleNode.org-pos)
                localDriftRange[1] = np.maximum(localDriftRange[1],
                                                self.handleTargetSplineGlobalBounds[1] + self.handleNode.org - pos)
                self.debuggingBoxes.append(pyutils.BoxFrame(c0=localDriftRange[0],
                                                            c1=localDriftRange[1],
                                                            org=pos))
                delta = np.array([random.uniform(localDriftRange[1][0], localDriftRange[0][0]),
                                  random.uniform(localDriftRange[1][1], localDriftRange[0][1]),
                                  random.uniform(localDriftRange[1][2], localDriftRange[0][2])])
                newpos = pos + delta
                self.handleNode.addTarget(t=t, pos=newpos)
        elif self.randomHandleTargetLinear:
            self.handleNode.usingTargets = True
            #draw initial pos
            oldOrg = np.array(self.handleNode.org)
            if len(self.targetSeeds) > self.reset_number:
                random.seed(self.targetSeeds[self.reset_number])
            self.handleNode.org = self.handleTargetLinearInitialRange.sample(1)[0]
            print("target = " + str(self.handleNode.org))
            if self.domainTesting:
                dim = self.domainTestingDim
                lpos = np.array([float(self.reset_number%dim[0]), math.floor(self.reset_number/dim[0])%dim[1],  math.floor(self.reset_number/(dim[0]*dim[1]))%dim[2]])
                #print("ix="+str(self.reset_number) + " | lpos="+str(lpos))
                if self.jitteredSamples:
                    for i in range(3):
                        lpos[i] = lpos[i] / float(max(dim[i], 1))
                    gridSize = np.array([1.0/dim[0], 1.0/dim[1], 1.0/dim[2]])
                    rand = np.random.rand()
                    self.handleNode.org = self.handleTargetLinearInitialRange.localSample(lpos=lpos+(rand*gridSize))
                else:
                    for i in range(3):
                        lpos[i] = lpos[i] / float(max(dim[i] - 1, 1))
                    self.handleNode.org = self.handleTargetLinearInitialRange.localSample(lpos=lpos)
            disp = self.handleNode.org-oldOrg
            self.handleNode.clearTargetSpline()
            if self.linearTargetFixed:
                self.handleNode.addTarget(t=self.handleTargetLinearWindow,
                                          pos=self.handleNode.org)
                self.clothScene.translateCloth(0, disp)
            else:
                if self.twoDistributions and self.domainTesting:
                    endIX = math.floor(self.reset_number/(dim[0]*dim[1]*dim[2]))
                    lpos = np.array([float(endIX % dim[0]), math.floor(endIX / dim[0]) % dim[1], math.floor(endIX / (dim[0] * dim[1]))])

                    if self.jitteredSamples:
                        for i in range(3):
                            lpos[i] = lpos[i] / float(max(dim[i], 1))
                        gridSize = np.array([1.0 / dim[0], 1.0 / dim[1], 1.0 / dim[2]])
                        rand = np.random.rand()
                        self.handleNode.addTarget(t=self.handleTargetLinearWindow, pos=self.handleTargetLinearEndRange.localSample(lpos=lpos + (rand * gridSize)))
                    else:
                        for i in range(3):
                            lpos[i] = lpos[i] / float(max(dim[i] - 1, 1))
                        self.handleNode.addTarget(t=self.handleTargetLinearWindow, pos=self.handleTargetLinearEndRange.localSample(lpos=lpos))

                else:
                    self.handleNode.addTarget(t=self.handleTargetLinearWindow, pos=self.handleTargetLinearEndRange.sample(1)[0])
                if self.orientationFromSpline:
                    self.clothScene.translateCloth(0, -oldOrg)
                    splineDirection = self.handleNode.targetSpline.pos(1)-self.handleNode.targetSpline.pos(0)
                    splineDirection[1] = 0 #project onto xz plane
                    splineDirection = splineDirection / np.linalg.norm(splineDirection)
                    angle = math.acos(splineDirection.dot(np.array([0,0,1.])))
                    cross = np.cross(splineDirection, np.array([0,0,1.]))
                    if cross.dot(np.array([0,1.0,0])) > 0:
                        angle = -angle
                    self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=angle, axis=np.array([0, 1.0, 0.0])))
                    self.clothScene.translateCloth(0, self.handleNode.org)
                    self.handleNode.recomputeOffsets()
                else:
                    self.clothScene.translateCloth(0, disp)

        else:
            oldOrg = np.array(self.handleNode.org)
            self.handleNode.usingTargets = False
            self.handleNode.org = np.array([0.2, 0.15, -0.3])
            disp = self.handleNode.org - oldOrg
            self.clothScene.translateCloth(0, disp)

        self.target = pyutils.getVertCentroid(verts=self.CP0Feature.verts, clothscene=self.clothScene)
        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        self.CP0Feature.fitPlane()
        if self.reset_number == 0:
            self.testMeshGraph()

        #self.handle_node.reset()
        if self.handleNode is not None:
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.recomputeOffsets()

        if self.gripper is not None:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        #print("reset " + str(self.reset_number))

        color = None
        if self.colorFromDistribution and self.randomHandleTargetLinear:
            xmin = min(self.handleTargetLinearInitialRange.c0[0], self.handleTargetLinearInitialRange.c1[0]) + self.handleTargetLinearInitialRange.org[0]
            xmax = max(self.handleTargetLinearInitialRange.c0[0], self.handleTargetLinearInitialRange.c1[0]) + self.handleTargetLinearInitialRange.org[0]
            ymin = min(self.handleTargetLinearInitialRange.c0[1], self.handleTargetLinearInitialRange.c1[1]) + self.handleTargetLinearInitialRange.org[1]
            ymax = max(self.handleTargetLinearInitialRange.c0[1], self.handleTargetLinearInitialRange.c1[1]) + self.handleTargetLinearInitialRange.org[1]
            #print("x ["+str(xmin)+","+str(xmax)+"]")
            #print("y [" + str(ymin) + "," + str(ymax) + "]")
            #print("point: " +str(self.handle_node.org[0])+","+str(self.handle_node.org[1]))
            #print("color [" + str((self.handle_node.org[0]-xmin)/(xmin-xmax)) + "," + str((self.handle_node.org[1]-ymin)/(ymin-ymax)) + "]")
            color = np.array([(self.handleNode.org[0]-xmin)/(xmax-xmin), 0.0, (self.handleNode.org[1]-ymin)/(ymax-ymin)])
        if self.graphArmProgress:
            if self.reset_number == 0:
                xdata = np.arange(400)
                self.armProgressGraph.xdata = xdata
            initialYData = np.zeros(400)
            self.armProgressGraph.plotData(ydata=initialYData, color=color)

        if self.graphDeformation:
            if self.reset_number == 0:
                xdata = np.arange(400)
                self.deformationGraph.xdata = xdata
            initialYData = np.zeros(400)
            self.deformationGraph.plotData(ydata=initialYData, color=color)

        if self.graphArmHaptics:
            for i, ix in enumerate(self.armHapticsIndices):
                if self.reset_number == 0:
                    xdata = np.arange(400)
                    self.armHapticsGraphs[i].xdata = xdata
                initialYData = np.zeros(400)
                self.armHapticsGraphs[i].plotData(ydata=initialYData, color=color)

        self.reset_number += 1

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        #collision spheres creation
        a=0
        
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[16].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[16].to_world(np.array([0,0.175,0]))
        cs4 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs5 = self.robot_skeleton.bodynodes[6].to_world(z)
        cs6 = self.robot_skeleton.bodynodes[7].to_world(z)
        cs7 = self.robot_skeleton.bodynodes[8].to_world(z)
        cs8 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        cs9 = self.robot_skeleton.bodynodes[10].to_world(z)
        cs10 = self.robot_skeleton.bodynodes[12].to_world(z)
        cs11 = self.robot_skeleton.bodynodes[13].to_world(z)
        cs12 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs13 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
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
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            if len(self.hapticSensorNodes) == 0: #if this is the first call, setup dual coupling data-structures
                self.hapticSensorNodes = [1,2,16,16,4,4,4,6,6,6,7,8,10,10,10,12,12,12,13,14,14]
                self.hapticSensorOffsets = [z,z,z,np.array([0,0.175,0]),z,
                                            self.robot_skeleton.bodynodes[4].to_local(LERP(cs4, cs5, 0.33)),
                                            self.robot_skeleton.bodynodes[4].to_local(LERP(cs4, cs5, 0.66)),
                                            z,
                                            self.robot_skeleton.bodynodes[6].to_local(LERP(cs5, cs6, 0.33)),
                                            self.robot_skeleton.bodynodes[6].to_local(LERP(cs5, cs6, 0.66)),
                                            z,z,fingertip,z,
                                            self.robot_skeleton.bodynodes[10].to_local(LERP(cs9, cs10, 0.33)),
                                            self.robot_skeleton.bodynodes[10].to_local(LERP(cs9, cs10, 0.66)),
                                            z,
                                            self.robot_skeleton.bodynodes[12].to_local(LERP(cs10, cs11, 0.33)),
                                            self.robot_skeleton.bodynodes[12].to_local(LERP(cs10, cs11, 0.66)),
                                            z,z,fingertip
                                            ]
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)

            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)

    def renderObservation(self):
        self._get_viewer().renderWorld = False
        self.clothScene.renderCollisionCaps = False
        self.clothScene.renderCollisionSpheres = False
        ef = None
        fingertip = np.array([0.0, -0.06, 0.0])
        if self.arm == 1:
            ef = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        elif self.arm == 2:
            ef = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vec = ef-self.target
        renderUtils.drawArrow(p1=ef, p0=self.target)
        renderUtils.setColor(color=[1.0,0,0])
        renderUtils.drawSphere(self.target,0.01)
        HSL = self.clothScene.getHapticSensorLocations()
        HSF = self.clothScene.getHapticSensorObs()
        HSIDs = self.clothScene.getHapticSensorContactIDs()
        for i in range(int(len(HSL)/3)):
            p = HSL[i * 3: i * 3 + 3]
            f = HSF[i * 3: i * 3 + 3]

            GL.glColor3d(0.0, 0.0, 1.0)
            if HSIDs[i] < 0:
                GL.glColor3d(1.0, 0.0, 0.0)
            elif HSIDs[i] > 0:
                GL.glColor3d(0.0, 1.0, 0.0)
            renderUtils.drawSphere(pos=p, rad=0.015)

            renderUtils.drawArrow(p0=p, p1=p+f)
            #GL.glBegin(GL.GL_LINES)
            #GL.glVertex(p[0] + f[0], p[1] + f[1], p[2] + f[2])
            #GL.glVertex(p[0], p[1], p[2])
            #GL.glEnd()
        minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene, meshgraph=self.separatedMesh, returnOnlyGeo=False)
        if minGeoVix is not None and self.arm_progress < 0:
            vixSide = 0
            if _side:
                vixSide = 1
            geoVec = None
            if minGeoVix >= 0:
                geoVec = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
                renderUtils.setColor(color=(0.,1.0,1.0))
                renderUtils.drawArrow(p0=ef, p1=ef+geoVec*0.25)
        elif self.arm_progress >= 0:
            geoVec = self.CP0Feature.plane.normal
            renderUtils.setColor(color=(0., 1.0, 1.0))
            renderUtils.drawArrow(p0=ef, p1=ef + geoVec * 0.25)
        else:
            geoVec = self.target - ef
            geoVec = geoVec/np.linalg.norm(geoVec)
            renderUtils.setColor(color=(0., 1.0, 1.0))
            renderUtils.drawArrow(p0=ef, p1=ef + geoVec * 0.25)
        a=0
        
    def extraRenderFunction(self):
        #self._get_viewer().renderWorld = True
        self.clothScene.renderCollisionCaps = True
        self.clothScene.renderCollisionSpheres = True
        #print("extra render function")

        #draw a gripper box
        if self.handleNode is not None and self.renderCover:
            renderUtils.setColor(color=[0.6, 0.6, 0.6])
            renderUtils.drawBox(cen=self.handleNode.org, dim=np.array([0.125,0.025,0.1]))

        if self.renderUI:
            if self.totalTime > 0:
                self.clothScene.drawText(x=15., y=30., text="Steps = " + str(self.numSteps) + " framerate = " + str(self.numSteps / self.totalTime), color=(0., 0, 0))
            #self.clothScene.drawText(x=15., y=30., text="Steps = " + str(self.numSteps), color=(0., 0, 0))

        if self.renderObs:
            self.renderObservation()

        if self.renderZoneColorFromDistribution and self.randomHandleTargetLinear:
            xmin = min(self.handleTargetLinearInitialRange.c0[0], self.handleTargetLinearInitialRange.c1[0]) + \
                   self.handleTargetLinearInitialRange.org[0]
            xmax = max(self.handleTargetLinearInitialRange.c0[0], self.handleTargetLinearInitialRange.c1[0]) + \
                   self.handleTargetLinearInitialRange.org[0]
            ymin = min(self.handleTargetLinearInitialRange.c0[1], self.handleTargetLinearInitialRange.c1[1]) + \
                   self.handleTargetLinearInitialRange.org[1]
            ymax = max(self.handleTargetLinearInitialRange.c0[1], self.handleTargetLinearInitialRange.c1[1]) + \
                   self.handleTargetLinearInitialRange.org[1]
            samples = 20
            for x in range(samples):
                for y in range(samples):
                    pos = np.array([(x/samples*(xmax-xmin))+xmin, (y/samples*(ymax-ymin))+ymin, self.handleTargetLinearInitialRange.org[2]])
                    #pos += self.handleTargetLinearInitialRange.org
                    color = color = np.array([(pos[0]-xmin)/(xmax-xmin), 0.0, (pos[1]-ymin)/(ymax-ymin)])
                    color = color*1.5
                    renderUtils.setColor(color=color)
                    renderUtils.drawSphere(pos=pos,rad=min(xmax-xmin, ymax-ymin)/samples)

        #draw contact points
        contactIndices = self.clothScene.getHapticSensorContactVertexIndices(21)
        for c in contactIndices:
            f = self.clothScene.getVertForce(vid=c)
            n = self.clothScene.getVertNormal(vid=c)
            side = f.dot(n) < 0
            '''print("v["+str(c)+"]"
                  #+" f="+str(f)
                  #+" n="+str(n)
                  +" side="+str(side))'''

            geo = 0
            if side:
                geo = self.separatedMesh.nodes[c + self.separatedMesh.numv].geodesic
            else:
                geo = self.separatedMesh.nodes[c].geodesic

            #print("geo = " + str(geo) + " (" + str(self.separatedMesh.nodes[c].geodesic) +"/"+ str(self.separatedMesh.nodes[c+self.separatedMesh.numv].geodesic)+")")

            if self.separatedMesh.maxGeo > 0:
                col = pyutils.heatmapColor(minimum=0.0, maximum=1.0, value=(geo / self.separatedMesh.maxGeo))
                renderUtils.setColor(color=col)
            vpos = self.clothScene.getVertexPos(vid=c)
            rad = 0.005
            geoSide = 0
            if side:
                geoSide = 1
            geoVec = self.separatedMesh.geoVectorAt(vix=c, side=geoSide)
            spherePos = None
            if side:
                #renderUtils.drawSphere(pos=vpos + (n * rad), rad=rad)
                spherePos = vpos + (n * rad)
            else:
                #renderUtils.drawSphere(pos=vpos - (n * rad), rad=rad)
                spherePos = vpos - (n * rad)
            #renderUtils.drawArrow(p0=spherePos, p1=spherePos+geoVec*0.05)

        #self.dart_world.skeletons[0].q = [0, 0, 0, 0, 0, 0]
        
        #GL.glBegin(GL.GL_LINES)
        #GL.glVertex3d(0,0,0)
        #GL.glVertex3d(-1,0,0)
        #GL.glEnd()

        #draw control torque bars
        topLeft = np.array([1100, self.viewer.viewport[3]-15])
        if self.renderUI:
            for ix,a in enumerate(self.prevAction):
                c = np.array([0.0,1.0,0.0])
                if a < 0:
                    c = np.array([1.0, 0.0, 0.0])
                self.clothScene.drawText(x=topLeft[0]-125-50, y=topLeft[1]-15,
                                         text=self.robot_skeleton.dof(ix).name,
                                         color=(0., 0, 0))
                self.clothScene.drawText(x=topLeft[0]-50, y=topLeft[1]-15, text='%.2f' % self.action_scale[ix],
                                         color=(0., 0, 0))
                renderUtils.drawProgressBar(topLeft=topLeft, h=16, w=100, progress=(a+1)/2.0, color=c, origin=0.5)
                topLeft[1] -= 20

        contactIndices = self.clothScene.getHapticSensorContactVertexIndices(21)
        HSL = self.clothScene.getHapticSensorLocations()
        pos = HSL[21 * 3:21 * 3 + 3]
        if self.renderObs is False:
            for c in contactIndices:
                side = ((pos - self.clothScene.getVertexPos(vid=c)).dot(self.clothScene.getVertNormal(vid=c))) < 0
                if side is True:
                    renderUtils.setColor(color=[1.0, 1.0, 0.])
                else:
                    renderUtils.setColor(color=[1.0, 0.0, 1.0])
                #renderUtils.drawSphere(self.clothScene.getVertexPos(vid=c), rad=0.005)

        # draw meshGraph stuff
        '''for n in self.separatedMesh.nodes:
            vpos = self.clothScene.getVertexPos(vid=n.vix)
            norm = self.clothScene.getVertNormal(cid=0, vid=n.vix)
            #geoVec = self.separatedMesh.geoVectorAt(n.vix,n.side)
            offset = 0.005
            #c = n.ix / len(self.separatedMesh.nodes)
            c=np.array([0.,0.,0.])
            if self.separatedMesh.maxGeo > 0:
                c = pyutils.heatmapColor(minimum=0.0, maximum=1.0, value=(n.geodesic / self.separatedMesh.maxGeo))
                #c = n.geodesic / self.separatedMesh.maxGeo
            renderUtils.setColor(color=c)
            spherePos = None
            rad = 0.005
            if n.side == 0:
                spherePos = vpos - (norm * rad)
            else:
                spherePos = vpos + (norm * rad)
            if n.geodesic < 0.1:
                geoVec = self.separatedMesh.geoVectorAt(n.vix, n.side)
                renderUtils.drawSphere(pos=spherePos, rad=rad)
                renderUtils.drawArrow(p0=spherePos, p1=spherePos + geoVec * 0.05)'''

        minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene, meshgraph=self.separatedMesh)
        if minContactGeodesic is not None and self.renderUI:
            self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 50,
                                     text="Contact Geo reward = " + str(1.0-minContactGeodesic/self.separatedMesh.maxGeo),
                                     color=(0., 0, 0))
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 38], h=16, w=60,
                                        progress=1.0-minContactGeodesic/self.separatedMesh.maxGeo, color=[0.0, 3.0, 3.0])


        if self.renderUI:
            self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 65,
                                    text="Reward = " + str(self.reward),
                                    color=(0., 0, 0))

            self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 25,
                                    text="Arm progress = " + str(self.arm_progress),
                                    color=(0., 0, 0))
            #renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.arm_progress, color=[0.0, 3.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.armSleeveProgress(), color=[0.0, 3.0, 0])

        #self.CP0Feature.drawProjectionPoly(renderPoints=True, renderNormal=False, fillColor=[0.75, 0.2, 0.2])

        #render debugging boxes
        if self.drawDebuggingBoxes:
            for ix,b in enumerate(self.debuggingBoxes):
                c = self.debuggingColors[ix]
                GL.glColor3d(c[0],c[1],c[2])
                b.draw()
                #for s in b.sample(50):
                #    self.viewer.drawSphere(p=s, r=0.01)

        #render the vertex handle_node(s)/Handle(s)
        if self.handleNode is not None and self.renderGripPath:
            self.handleNode.draw()

        if self.gripper is not None and False:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.gripper.draw()
            if self.clothScene is not None and False:
                vix = self.clothScene.getVerticesInShapeFrame(self.gripper)
                GL.glColor3d(0,0,1.)
                for v in vix:
                    p = self.clothScene.getVertexPos(vid=v)
                    GL.glPushMatrix()
                    GL.glTranslated(p[0], p[1], p[2])
                    GLUT.glutSolidSphere(0.005, 10, 10)
                    GL.glPopMatrix()
            
        m_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        
        textX = 15.
        if self.renderUI:
            if self.renderForceText:
                HSF = self.clothScene.getHapticSensorObs()
                for i in range(self.clothScene.getNumHapticSensors()):
                    self.clothScene.drawText(x=textX, y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
                textX += 160

            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)

            self.clothScene.drawText(x=15 , y=600., text='Friction: %.2f' % self.clothScene.getFriction(), color=(0., 0, 0))
            #f = self.clothScene.getHapticSensorObs()
            f = np.zeros(66)
            maxf_mag = 0

            for i in range(int(len(f)/3)):
                fi = f[i*3:i*3+3]
                #print(fi)
                mag = np.linalg.norm(fi)
                #print(mag)
                if mag > maxf_mag:
                    maxf_mag = mag
            #exit()
            self.clothScene.drawText(x=15, y=620., text='Max force (1 dim): %.2f' % np.amax(f), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=640., text='Max force (3 dim): %.2f' % maxf_mag, color=(0., 0, 0))
        #print(self.viewer.renderWorld)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            if self.gripperCover:
                self.viewer.interactors[4].skelix = 2
            #self._get_viewer().scene.tb.trans[2] = -3.5
            #self._get_viewer().scene.tb._set_theta(180)
            #self._get_viewer().scene.tb._set_phi(180)
            #self._get_viewer().scene.tb._set_orientation(180-16.7, 180+5.4)
            #self._get_viewer().scene.tb.trans[2] = -1.9

            #camera settings for forward fixed gown
            self._get_viewer().scene.tb.trans[:3] = [-0.33, 0.11, -1.9]
            self._get_viewer().scene.tb._rotation = [-0.051643844511207959, 0.80910393196194152, 0.080374382346619963, 0.57984799669415554]

            #camera settings for side linear gown
            #self._get_viewer().scene.tb.trans[:3] = [0.45, 0.07, -1.9]
            #self._get_viewer().scene.tb._rotation = [-0.024850523841696202, 0.98784543610802078, 0.0013834212176602832, -0.15343380322172276]

            #camera settings for front linear gown
            #self._get_viewer().scene.tb.trans[:3] = [-0.42000000000000015, 0.029999999999999999, -1.5000000000000002]
            #self._get_viewer().scene.tb._rotation = [-0.061093026927481663, 0.4956926655636571, -0.031464958589811055, 0.86577501681913671]

            #camera settigs for frontal character close-up
            #self._get_viewer().scene.tb.trans[:3] = [3.4694469519536142e-18, 0.19000000000000003, -1.5999999999999983]
            #self._get_viewer().scene.tb._rotation = [6.123233995736766e-17, 1.0, 6.123233995736766e-17, 3.749399456654644e-33]


            self._get_viewer().scene.tb.drag_to(0.01, 0.01, 0.01, 0.01)
            #theta,phi = self._get_viewer().scene.tb._get_orientation()
            #self._get_viewer().scene.tb._set_orientation(theta, phi)
            #m = self._get_viewer().scene.tb._q_rotmatrix(self._get_viewer().scene.tb._rotation)
            #self._get_viewer().scene.tb._matrix =
            self.track_skeleton_id = 0
        if not self.renderDARTWorld:
            self.viewer.renderWorld = False


    def armSleeveProgress(self):
        # return the progress of the arm through the 1st sleeve seam
        limblines = []
        fingertip = np.array([0.0, -0.07, 0.0])
        end_effector = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        if self.arm == 2:
            end_effector = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        armProgress = 0

        if self.CP0Feature.plane is not None:
            armProgress = -np.linalg.norm(end_effector - self.CP0Feature.plane.org)

        if self.arm == 1:
            limblines.append([self.robot_skeleton.bodynodes[8].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[8].to_world(fingertip)])
            limblines.append([self.robot_skeleton.bodynodes[7].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))])
            limblines.append([self.robot_skeleton.bodynodes[6].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[7].to_world(np.zeros(3))])
            limblines.append([self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[6].to_world(np.zeros(3))])
        elif self.arm == 2:
            limblines.append([self.robot_skeleton.bodynodes[14].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[14].to_world(fingertip)])
            limblines.append([self.robot_skeleton.bodynodes[13].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))])
            limblines.append([self.robot_skeleton.bodynodes[12].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))])
            limblines.append([self.robot_skeleton.bodynodes[10].to_world(np.zeros(3)),
                              self.robot_skeleton.bodynodes[12].to_world(np.zeros(3))])

        #print("---")
        if self.armLength < 0:
            self.armLength = 0.
            for line in limblines:
                #print("limb length = " + str(np.linalg.norm(line[1] - line[0])))
                self.armLength += np.linalg.norm(line[1] - line[0])
        #print("arm length = " +str(self.armLength))
        contains = False
        intersection_ix = -1
        intersection_depth = -1.0
        for ix, line in enumerate(limblines):
            line_contains, intersection_dist, intersection_point = self.CP0Feature.contains(l0=line[0], l1=line[1])
            if line_contains is True:
                intersection_ix = ix
                intersection_depth = intersection_dist
                contains = True

        if contains is True:
            armProgress = -intersection_depth
            for i in range(intersection_ix + 1):
                armProgress += np.linalg.norm(limblines[i][1] - limblines[i][0])

        limb = []
        for ix in range(len(limblines)):
            limb.append(limblines[-(ix+1)][1])
        armProgress2 = pyutils.limbFeatureProgress(limb, self.CP0Feature)

        print("New metric: " + str(armProgress2) + " | Old Metric " + str(armProgress))

        return armProgress

    def testMeshGraph(self):
        print("Testing MeshGraph")

        self.separatedMesh.initSeparatedMeshGraph()
        self.separatedMesh.updateWeights()
        self.separatedMesh.computeGeodesic(feature=self.CP0Feature, oneSided=True, side=0, normalSide=1)

        print("done")

    def applyClothForcesToSkel(self):
        #apply all cumulative force sensor values to the DART skeleton
        if len(self.hapticSensorNodes) == 0:
            return
        HSF = self.clothScene.getHapticSensorObs()
        for i in range(int(len(HSF)/3)):
            f = HSF[i * 3: i * 3 + 3]
            node = self.robot_skeleton.bodynodes[self.hapticSensorNodes[i]]
            offset = self.hapticSensorOffsets[i]


def LERP(p0, p1, t):
    return p0 + (p1-p0)*t
