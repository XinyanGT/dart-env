# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
import quaternion
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.clothHandles import *
from pyPhysX.clothfeature import *
import pyPhysX.meshgraph as meshgraph

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

''' This env is setup for upper body single arm reduced action space learning with draped shirt'''



class DartClothGrippedTshirtSpline2ndArmEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.prefix = os.path.dirname(__file__)
        self.useOpenGL = True
        self.target = np.array([0.8, -0.6, 0.6])
        self.targetInObs = True
        self.phaseInObs = False
        self.arm = 2  # 0 both, 1 right, 2 left

        self.arm_progress = 0.  # set in step when first queried
        self.armLength = -1.0  # set when arm progress is queried

        self.interiorContactReward = 0.0

        self.reward = 0

        #22 dof upper body
        self.action_scale = np.ones(22)*10
        self.control_bounds = np.array([np.ones(22), np.ones(22)*-1])

        if self.arm > 0:
            self.action_scale = np.ones(11) * 10
            self.control_bounds = np.array([np.ones(11), np.ones(11) * -1])

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

        # target spline
        self.targetSpline = pyutils.Spline()
        self.splineCP0Verts = [7, 140, 2255, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2241, 2326, 2334, 2288, 2289, 2373, 2264, 2419, 2444, 2345, 2408, 2375, 2234, 2399]
        #self.splineCP0Verts = [7, 2339, 2398, 2343, 2384, 2405, 2421, 2275, 2250, 134, 136, 138]
        self.splineCP1Verts = [232, 230, 227, 225, 222, 218, 241, 239, 237, 235, 233]
        self.targetSplineTime = 0  # set to 0 in reset
        self.incrementSplineTime = True
        self.extendedSplineTarget = True

        # handle node setup
        self.handleNode = None
        self.gripper = None

        #SPD control
        self.useSPD = False
        self.q_target = None #set in reset()
        self.Kp_scale = 800.0  # default params
        self.timeStep = 0.04  # default params
        self.Kd_scale = self.Kp_scale * self.timeStep * 0.5  # default params
        self.Kd = None
        self.Kp = None

        self.enforceTauLimits = False
        self.tau_limits = [np.ones(22) * -10, np.ones(22) * 10]
        if self.arm > 0:
            self.tau_limits = [np.ones(11) * -10, np.ones(11) * 10]

        self.graphTau = False
        if self.graphTau:
            self.linegraph = pyutils.LineGrapher(title="||Tau||")

        #interactive handle mode
        self.interactiveHandleNode = False

        self.updateHandleNodeFrom = 8 #if -1, this feature is disabled
        #self.interactiveIkTarget = True

        #saved IK info
        self.ikPose = None
        self.ikRestPose = None
        self.interactiveIK = False
        self.ikTarget = np.array([0.2, -0.1, -0.5])
        self.ikLines = []

        #randomized spline target mode
        self.randomHandleTargetSpline = False
        self.handleTargetSplineWindow = 10.0 #time window for the full motion (split into equal intervals b/t CPs)
        self.numHandleTargetSplinePoints = 4
        self.handleTargetSplineGlobalBounds = [np.array([0.75,0.3,1.0]), np.array([-0.0,-0.5,0.])] #total drift allowed from origin for target orgs
        self.handleTargetSplineLocalBounds = [np.array([0.25,0.25,0.35]), np.array([-0.25,-0.25,-0.05])] #cartesian drift allowed b/t neighboring CPs
        #TODO: add rotation
        #self.handleTargetSplineGlobalRotationBounds

        #linear spline target mode
        self.randomHandleTargetLinear = False
        self.handleTargetLinearWindow = 10.0
        self.handleTargetLinearInitialRange = pyutils.BoxFrame(c0=np.array([0.7,0.5,0.15]),
                                                               c1=np.array([-0.3, -0.5, -0.15]),
                                                               org=np.array([-0.17205264,  0.12056234, -1.07377446]))
        self.handleTargetLinearEndRange = pyutils.BoxFrame(c0=np.array([0.5, 0.3, 0.2]),
                                                           c1=np.array([0.1, -0.1, -0.1]),
                                                           org=np.array([0.,0.,0.]))

        #debugging boxes for visualizing distributions
        self.drawDebuggingBoxes = True
        #self.debuggingBoxes = [self.handleTargetLinearInitialRange, self.handleTargetLinearEndRange]
        self.debuggingBoxes = []
        self.debuggingColors = [[0., 1, 0], [0, 0, 1.], [1., 0, 0], [1., 1., 0], [1., 0., 1.], [0, 1., 1.]]

        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01,
                                        #mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/fullgown1.obj",
                                        mesh_path=self.prefix + "/assets/tshirt_m.obj",
                                        #state_path="/home/alexander/Documents/dev/tshirt_regrip1.obj",
                                        #state_path="/home/alexander/Documents/dev/tshirt_regrip2.obj",
                                        #state_path="/home/alexander/Documents/dev/tshirt_regrip3.obj",
                                        #state_path="/home/alexander/Documents/dev/1st_to_2nd_regrip.obj",
                                        #state_path="/home/alexander/Documents/dev/start2ndSleeve_gripped.obj",
                                        state_path=self.prefix + "/../../../../start2ndSleeve_gripped_easy.obj",
                                        #state_path="/home/alexander/Documents/dev/1stSleeveState.obj",
                                        scale=1.4)

        clothScene.togglePinned(0,0) #turn off auto-pin
        #clothScene.togglePinned(0, 144)
        #clothScene.togglePinned(0, 190)

        self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)

        self.reset_number = 0  # increments on env.reset()

        observation_size = 66 + 66  # pose(sin,cos), pose vel, haptics
        if self.targetInObs:
            observation_size += 6 #target reaching
        if self.phaseInObs:
            observation_size += 1

        #intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules_handplane_L.skel', frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds)#, disableViewer=True, visualize=False)
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules_handplane_L.skel',
                                  frame_skip=4,
                                  observation_size=observation_size,
                                  action_bounds=self.control_bounds, disableViewer=True, visualize=False)

        utils.EzPickle.__init__(self)

        self.CP0Feature = ClothFeature(verts=self.splineCP0Verts, clothScene=self.clothScene)
        self.CP1Feature = ClothFeature(verts=self.splineCP1Verts, clothScene=self.clothScene)
        self.collarFeature = ClothFeature(verts=[117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120],
                                          clothScene=self.clothScene)

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

        print("done init")

    def _getFile(self):
        return __file__

    def limits(self, dof_ix):
        return np.array([self.robot_skeleton.dof(dof_ix).position_lower_limit(), self.robot_skeleton.dof(dof_ix).position_upper_limit()])

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)
        
    def loadObjState(self):
        self.clothScene.loadObjState("objState", 0)

    def hemisphereSample(self, maxradius=1., minradius = 0., norm=np.array([0,0,1.]), frustrum = 0.7):
        p = norm
        while True:
            p = self.np_random.uniform(low=-maxradius, high=maxradius, size=3)
            p_n = np.linalg.norm(p)
            if p_n <= maxradius and p_n >= minradius:
                if(np.dot(p/p_n, norm) > frustrum):
                    return p

    def _step(self, a):
        if self.reset_number < 1:
            return self._get_obs(), 0, False, {}

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)


        if self.handleNode is not None:
            #self.handle_node.setTranslation(T=self.viewer.interactors[2].frame.org)
            if self.interactiveHandleNode:
                self.handleNode.org = self.viewer.interactors[2].frame.org
                self.handleNode.setOrientation(R=self.viewer.interactors[2].frame.orientation)
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()

        #if self.gripper is not None:
        #    self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        #increment self collision distance test
        #currentDistance = self.clothScene.getSelfCollisionDistance()
        #print("current self-collision distance = " + str(currentDistance))
        #self.clothScene.setSelfCollisionDistance(currentDistance + 0.0001)

        #test SPD
        if self.useSPD is True:
            #if self.numSteps == 0:
            #    self.viewer.interactors[2].frame.org = self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))
            #eftarget = self.viewer.interactors[2].frame.org
            #eftarget = self.robot_skeleton.bodynodes[8].to_world(np.zeros(3)) + np.array([0.1,0,0])
            #self.q_target = self.robot_skeleton.q + self.IK(target=eftarget)
            #self.q_target = self.iterativeIK(target=self.ikTarget, iterations=50)

            iknodes = [8]
            offsets = [np.zeros(3)]
            targets = [self.ikTarget]

            if self.interactiveIK:
                iknodes = self.viewer.interactors[3].ikNodes#[8]#, 13]
                offsets = self.viewer.interactors[3].ikOffsets#[np.zeros(3)]#, np.zeros(3)]
                targets = self.viewer.interactors[3].ikTargets#[self.ikTarget]#, self.ikTarget]

            #self.q_target = self.iterativeMultiTargetIK(robot=self.robot_skeleton, nodeixs=iknodes, offsets=offsets, targets=targets, iterations=10, restPoseWeight=0.1, restPose=self.ikRestPose)
            tau = self.SPD(qt=self.q_target, Kd=self.Kd, Kp=self.Kp, dt=self.timeStep)
            #tau = np.zeros(len(self.robot_skeleton.q))

        if self.enforceTauLimits:
            tau = np.maximum(np.minimum(self.tau_limits[1], tau), self.tau_limits[0])

        #update targetspline
        fingertip = np.array([0.0, -0.06, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        if self.arm == 2:
            wRFingertip1 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vecR1 = self.target - wRFingertip1
        self.setTargetSplineFromVerts()

        #now compute splineTime
        if self.incrementSplineTime is True:
            while self.targetSplineTime < 1.0 and np.linalg.norm(vecR1) < 0.1:
                self.targetSplineTime += 0.01
                self.target = self.targetSpline.pos(self.targetSplineTime)
                vecR1 = self.target - wRFingertip1
        self.target = self.targetSpline.pos(self.targetSplineTime)
        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #apply action and simulate
        if self.arm == 1:
            tau = np.concatenate([tau, np.zeros(11)])
        elif self.arm == 2:
            tau = np.concatenate([tau[:3], np.zeros(8), tau[3:], np.zeros(3)])
        self.do_simulation(tau, self.frame_skip)

        normhint = None
        if self.numSteps == 0:
            normhint = np.array([0.,1.0,0])
        #update the features
        self.collarFeature.fitPlane()
        self.CP0Feature.fitPlane()
        self.CP1Feature.fitPlane(normhint=normhint)

        reward = 0
        self.arm_progress = self.armSleeveProgress() / self.armLength
        ob = self._get_obs()
        s = self.state_vector()
        
        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        
        #check termination conditions
        done = False

        #get contact IDs
        #HSIDs = self.clothScene.getHapticSensorContactIDs()
        '''self.interiorContactReward = 0.0
        if self.arm_progress <= 0:
            for i in range(20,22):
                if HSIDs[i] <= 0.0:
                    self.interiorContactReward += -1.5
                elif HSIDs[i] > 0.0:
                    self.interiorContactReward += 0.5
        else:
            #automatic bonus for being in/beyond the sleeve
            self.interiorContactReward = 1.0'''

        minContactGeodesic = None
        if self.numSteps > 0:
            minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene,
                                                               meshgraph=self.separatedMesh)
        contactGeoReward = 0
        if self.arm_progress > 0:
            contactGeoReward = 1.0
        elif minContactGeodesic is not None:
            contactGeoReward = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo

        wRFingertip2 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        if self.arm == 2:
            wRFingertip2 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vecR2 = self.target - wRFingertip2

        reward_progress = self.targetSplineTime
        reward_distR = -np.linalg.norm(vecR2)*10
        reward_ctrl = - np.square(tau).sum() * 0.00005
        alive_bonus = -0.001
        #reward = reward_ctrl + alive_bonus + reward_distR + reward_progress + arm_progress
        reward = alive_bonus + reward_ctrl + self.arm_progress*10.0 + contactGeoReward*2.0 + self.targetSplineTime#+ self.interiorContactReward

        self.reward = reward

        # check cloth deformation for termination
        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        if not np.isfinite(s).all():
            #print("Infinite value detected..." + str(s))
            done = True
            reward -= 500
        elif (clothDeformation > 20):
            #print("Deformation Termination")
            done = True
            reward -= 5000
        elif self.targetSplineTime >= 1.0:
            #print("Spline Termination")
            done = True
            reward += 1000
        elif self.headCollarContainment() is None:
            #print("Collar Termiantion")
            done = True
            reward = -4000
        elif self.armLength > 0 and self.arm_progress >= 0.95:
            done=True
            reward = 1000
            print("Dressing completed!")


        #graphing force
        if self.graphTau and self.reset_number > 0:
            self.linegraph.addToLinePlot(data=[[np.linalg.norm(tau)]])

        self.numSteps += 1

        #self.IK()

        return ob, reward, done, {}

    def _get_obs(self):
        '''get_obs'''
        f_size = 66
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.06, 0.0])

        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        else:
            f = np.zeros(f_size)

        obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq]).ravel()
        if self.targetInObs:
            vec = None
            if self.arm == 1:
                vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
            else:
                vec = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target
            obs = np.concatenate([obs, vec, self.target]).ravel()
        if self.phaseInObs:
            obs = np.concatenate([obs, np.array([self.arm_progress])]).ravel()

        obs = np.concatenate([obs, f*3.]).ravel()
        '''print("cos obs: " + str(obs[:22]))
        print("sin obs: " + str(obs[22:44]))
        print("dq obs: " + str(obs[44:66]))
        print("target obs: " + str(obs[66:72]))
        print("phase obs: " + str(obs[72]))
        print("f obs: " + str(obs[72:138]))'''
        #print(obs)
        #obs = np.concatenate([theta, self.robot_skeleton.dq, f]).ravel()
        return obs

    def reset_model(self):
        '''reset_model'''
        #if self.reset_number > 0:
        #    self.enforceTauLimits = True
        self.numSteps = 0
        self.targetSplineTime = 0
        self.dart_world.reset()
        self.clothScene.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
        #first try: couples with tshirt_regrip1.obj
        '''qpos = [  2.18672993e-03,  -4.72069042e-03,  -1.13570380e-02,   1.49440348e-02,
   1.29127624e-02,  -1.14191071e-02,   6.39350903e-03,  -2.15463769e-03,
  -8.75123999e-03,   1.12041144e-02,  -3.17658842e-02,  -1.11111111e-02,
   8.33333333e-02,   1.16666667e-01,   1.12604700e+00,   6.30000000e-01,
   1.38555556e+00,   6.00000000e-01,  -7.23481822e-02,  -7.26463958e-03,
  -8.24054408e-03,  -1.29043413e-03] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''
        #2nd pose: couples with tshirt_regrip2.obj
        '''qpos = [0.0119845464534, -0.0148237339287, 0.00114245295676, 0.116666666667, -0.25, -0.326666666667,
                0.360326294171, 2.1, 2.9, -0.00347161244598, -0.0439589287717, -0.0236843580476, 0.0709439334176,
                0.053095524649, 1.15934253669, 0.626036755784, 1.71154787065, 0.600289882587, -0.0740444449891,
                -0.0163328396778, 0.00592014136357, -0.00843219489643] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''

        #start pose of 1st sleeve dressing
        '''qpos = [0.123146842221, -0.0289355342829, 0.0228754335829, 0.178160667409, -0.232380860117, -0.92466391357,
         0.799702454732, 2.1, 2.8925789487, -0.00872324714354, -0.0501696842966, 0.00447277300298, 0.0888644449976,
         0.0648296316731, 1.17563923517, 0.641600883822, 1.69865162075, 0.612304711936, -0.0668473720041,
         -0.0071334969361, -0.00759981159759, -0.0168213446051] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''

        #end pose of 1st sleeve dressing
        '''qpos = [-0.41544270972, 0.242822559203, 1.00014704416, 0.203617333923, 0.0384987751629, 0.618666743985,
                2.49978726812, 2.06799741286, 0.0577308155211, -0.605127798666, 0.186585275629, -0.225383463656,
                0.105293054792, -0.332072203848, 0.970443711246, 0.920802253398, 1.66793432967, 0.611389466222,
                -0.077386511834, 0.190709600282, -0.0302107834603, 0.00960955816463] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''

        #regrip pose (hands together):
        '''qpos = [-0.0665044783311, 0.0320441055726, 0.132492367645, -0.162985559062, 0.24800151523, -0.673821597214,
         0.447179812699, 1.23709603772, 1.96643981164, 0.517966631991, -0.0315090321144, -0.00558062164692,
         0.0466310070329, -0.385470651234, 1.04880411208, 0.86507793576, 1.58016635743, 0.615275768064,
         -0.0824549688069, 0.185599327056, -0.0233487609124, -0.00244875091393] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''

        #start pose of 2nd sleeve dressing (gripped) (corresponds to start2ndsleeve_gripped.obj)
        '''qpos = [-0.157687732756, 0.0162382715017, 0.0967432360388, -0.160683852563, 0.242810336962, -0.673174385754,
         0.486173577025, 1.24943535367, 1.98050908209, 0.527001190001, -0.0222554188732, 0.00663342222776,
         0.0836726418349, -0.63553814711, 0.866285606682, 1.14543917537, 2.08035747595, 0.6, 0.0572976400435,
         0.167755129145, -0.0126048093718, 0.0106245477183] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''

        #easier start pose of 2nd sleeve dressing (gripped) (corresponds to start2ndsleeve_gripped_easy.obj)
        qpos = [0.00362010610933, 0.0703250612824, 0.0166962543044, -0.184366642354, 0.20626496349, -0.707279042691,
         0.633210692541, 1.21458866989, 2.11223003587, 0.472799630871, -0.186799890496, 0.0571894134444, 0.123649461588,
         -0.602553150012, 0.797597241975, 1.50669631424, 2.26882959353, 0.579261786627, 0.0720725378308, 0.146064284387,
         -0.0228403491972, 0.012006764166] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.025, high=.025, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        '''self.q_target = [-0.0665044783311, 0.0320441055726, 0.132492367645, -0.162985559062, 0.24800151523, -0.673821597214,
         0.447179812699, 1.23709603772, 1.86643981164, 0.517966631991, -0.0315090321144, -0.00558062164692,
         0.0466310070329, -0.385470651234, 1.04880411208, 0.86507793576, 1.68016635743, 0.615275768064,
         -0.0824549688069, 0.185599327056, -0.0233487609124, -0.00244875091393]'''
        #self.q_target = np.array(self.robot_skeleton.q)
        #self.q_target[8] = 1.0

        '''if self.reset_number == 0:
            self.q_target = pyutils.getRandomPose(robot=self.robot_skeleton)
            self.q_target[0] = 0.39
            self.q_target = [ 0.39294164, -0.28923569,  0.44674558,  0.21799112,  0.22719583, -0.34199703,
 -0.8429895,   1.50691939,  1.28366487,  0.54321876, -0.36126751,  0.11050672,
 -0.12107801, -1.6634833,   1.03492742,  1.512525,    0.22033149,  0.48043698,
  0.28899992,  0.15556353, -0.02484711,  0.19311922]
        self.ikLines.append(pyutils.getRobotLinks(robot=self.robot_skeleton, pose=self.q_target))'''

        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 0, 1.])))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 1., 0.])))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=-1., axis=np.array([1., 0., 0.])))
        #self.clothScene.translateCloth(0, np.array([0.75, -0.5, -0.5]))  # shirt in front of person
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))

        #self.clothScene.translateCloth(0, np.array([0.0, 1.0, 0.0]))

        #self.clothScene.translateCloth(0, np.array([5.75, -0.5, -0.5]))  # get the cloth out of the way
        self.clothScene.setSelfCollisionDistance(0.03)

        #reset reacher target from hemisphere
        reacher_range = 1.1
        hemisphereDir = np.array([-1.0,0,0])
        if self.arm == 2:
            hemisphereDir = np.array([1.0, 0, 0])
        self.target = self.hemisphereSample(maxradius=reacher_range, minradius=0.9, norm=hemisphereDir)
        while self.target[1] < 0:
            self.target = self.hemisphereSample(maxradius=reacher_range, minradius=0.9, norm=hemisphereDir)
        if self.targetInObs:
            self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        self.targetSpline.points = []
        self.targetSpline.insert(t=0, p=np.zeros(3))
        self.targetSpline.insert(t=0.5, p=np.zeros(3))
        self.targetSpline.insert(t=1.0, p=self.target)

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)

        self.handleNode.clearHandles()
        #self.handle_node.addVertex(vid=0)
        #self.clothScene.setPinned(cid=0, vid=0)

        #self.clothScene.refreshMotionConstraints()
        #self.clothScene.refreshCloth()
        #self.clothScene.clearInterpolation()
        #self.dart_world.skeletons[0].q = [0, 0, 0, self.ikTarget[0], self.ikTarget[1], self.ikTarget[2]]
        self.ikRestPose = np.array(self.robot_skeleton.q)
        self.ikRestPose[8] += 2.5 #elbow bend
        self.ikRestPose[6] += 1.0

        self.handleNode.clearHandles()
        self.handleNode.addVertices(verts=[489, 787, 327, 362, 676, 887, 54, 55])
        #self.handle_node.addVertices(verts=[468, 1129, 975, 354, 594, 843, 654, 682, 415, 378, 933, 547, 937, 946, 763, 923, 2395, 2280, 2601, 2454])
        #self.handle_node.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])
        #self.handle_node.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])
        self.handleNode.setOrgToCentroid()
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
            self.handleNode.org = self.handleTargetLinearInitialRange.sample(1)[0]
            disp = self.handleNode.org-oldOrg
            self.clothScene.translateCloth(0, disp)
            self.handleNode.clearTargetSpline()
            self.handleNode.addTarget(t=self.handleTargetLinearWindow, pos=self.handleTargetLinearEndRange.sample(1)[0])
        elif self.updateHandleNodeFrom >= 0:
            self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            #self.handle_node.org = self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].com()


        #set target spline
        self.targetSpline.points[2].p = self.target
        self.setTargetSplineFromVerts()
        self.target = self.targetSpline.pos(self.targetSplineTime)
        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        self.CP0Feature.fitPlane()
        if self.reset_number == 0:
            self.testMeshGraph()

        self.reset_number += 1

        #self.handle_node.reset()
        if self.handleNode is not None:
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.recomputeOffsets()

        if self.gripper is not None:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        self.arm_progress = self.armSleeveProgress() / self.armLength

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
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)

            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)

    def headCollarContainment(self):
        #returns None or distance to top of head of the collar triangle
        tp0 = self.clothScene.getVertexPos(cid=0, vid=657)
        tp1 = self.clothScene.getVertexPos(cid=0, vid=109)
        tp2 = self.clothScene.getVertexPos(cid=0, vid=8)
        lp0 = self.robot_skeleton.bodynodes[16].to_world(np.array([0., -0.05, 0.]))  # neck root
        lp1 = self.robot_skeleton.bodynodes[16].to_world(np.array([0., 0.28, 0.]))  # head root
        l_len = np.linalg.norm(lp1 - lp0)
        intsersects, intersection_distance, p = self.collarFeature.contains(l0=lp0, l1=lp1)
        #intersection_distance = pyutils.triangleLineSegIntersect(tp0, tp1, tp2, lp0, lp1, False)
        #intersection_point = pyutils.triangleLineSegIntersect(tp0, tp1, tp2, lp0, lp1, True)

        if intersection_distance is not None:
            #return the remaining distance to the top of the head
            return l_len-intersection_distance
        #print("intersection_distance = " + str(intersection_distance))
        return intersection_distance

    def armSleeveProgress(self):
        #return the progress of the arm through the 1st sleeve seam
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

        if self.armLength < 0:
            self.armLength = 0.
            for line in limblines:
                self.armLength += np.linalg.norm(line[1] - line[0])
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

        return armProgress

    def getVertCentroid(self, verts):
        centroid = np.zeros(3)
        if len(verts) > 0:
            for v in verts:
                centroid += self.clothScene.getVertexPos(cid=0, vid=v)
            centroid /= len(verts)
        return centroid

    def setTargetSplineFromVerts(self):
        self.targetSpline.points[0].p = self.getVertCentroid(self.splineCP0Verts)
        self.targetSpline.points[1].p = self.getVertCentroid(self.splineCP1Verts)
        if self.extendedSplineTarget is True and self.CP1Feature.plane is not None:
            #get the target by extending the sleeve plane
            self.targetSpline.points[2].p = self.targetSpline.points[1].p - self.CP1Feature.plane.normal

    def extraRenderFunction(self):
        #print("extra render function")

        self.clothScene.drawText(x=15., y=30., text="Steps = " + str(self.numSteps), color=(0., 0, 0))

        self.targetSpline.draw()
        
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        # draw meshGraph stuff
        '''for n in self.separatedMesh.nodes:
            vpos = self.clothScene.getVertexPos(vid=n.vix)
            norm = self.clothScene.getVertNormal(cid=0, vid=n.vix)
            offset = 0.005
            #c = n.ix / len(self.separatedMesh.nodes)
            c=np.array([0.,0.,0.])
            if self.separatedMesh.maxGeo > 0:
                c = pyutils.heatmapColor(minimum=0.0, maximum=1.0, value=(n.geodesic / self.separatedMesh.maxGeo))
                #c = n.geodesic / self.separatedMesh.maxGeo
            renderUtils.setColor(color=c)
            if n.side == 0:
                renderUtils.drawSphere(pos=vpos + norm * offset, rad=0.005)
            else:
                renderUtils.drawSphere(pos=vpos - norm * offset, rad=0.005)'''

        minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene,
                                                           meshgraph=self.separatedMesh)
        if minContactGeodesic is not None:
            self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 50,
                                     text="Contact Geo reward = " + str(
                                         1.0 - minContactGeodesic / self.separatedMesh.maxGeo),
                                     color=(0., 0, 0))
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 38], h=16, w=60,
                                        progress=1.0 - minContactGeodesic / self.separatedMesh.maxGeo,
                                        color=[0.0, 3.0, 3.0])

        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 65,
                                 text="Reward = " + str(self.reward),
                                 color=(0., 0, 0))

        #test best fit plane
        limblines = []
        fingertip = np.array([0.0, -0.07, 0.0])
        end_effector = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        armProgress = 0
        armProgress = self.armSleeveProgress()
        '''if self.CP0Feature.plane is not None:
            armProgress = -np.linalg.norm(end_effector-self.CP0Feature.plane.org)
        limblines.append([self.robot_skeleton.bodynodes[8].to_world(np.zeros(3)),
                          self.robot_skeleton.bodynodes[8].to_world(fingertip)])
        limblines.append([self.robot_skeleton.bodynodes[7].to_world(np.zeros(3)),
                          self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))])
        limblines.append([self.robot_skeleton.bodynodes[6].to_world(np.zeros(3)),
                          self.robot_skeleton.bodynodes[7].to_world(np.zeros(3))])
        limblines.append([self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)),
                          self.robot_skeleton.bodynodes[6].to_world(np.zeros(3))])
        contains = False
        intersection_ix = -1
        intersection_depth = -1.0
        for ix,line in enumerate(limblines):
            line_contains, intersection_dist, intersection_point = self.CP0Feature.contains(l0=line[0], l1=line[1])
            if line_contains is True:
                intersection_ix = ix
                intersection_depth = intersection_dist
                contains = True

        fillColor = [1.0, 0., 0.]
        if contains is True:
            fillColor = [0., 1., 0.]
            armProgress = -intersection_depth
            for i in range(intersection_ix+1):
                armProgress += np.linalg.norm(limblines[i][1]-limblines[i][0])'''
        self.CP0Feature.drawProjectionPoly(fillColor=[0.,1.0,0.0])

        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 55,
                                 text="Interior Contact Reward = " + str(self.interiorContactReward),
                                 color=(0., 0, 0))
        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 40,
                                 text="Arm progress Reward = " + str((self.arm_progress/self.armLength)*10.0),
                                 color=(0., 0, 0))
        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 25,
                                 text="Arm progress = " + str(armProgress),
                                 color=(0., 0, 0))
        renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=armProgress/self.armLength, color=[0.0,3.0,0])
        '''self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 40,
                                 text="Intersection_ix = " + str(intersection_ix),
                                 color=(0., 0, 0))
        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 55,
                                 text="intersection_depth = " + str(intersection_depth),
                                 color=(0., 0, 0))
        if intersection_ix >= 0:
            self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 70,
                                     text="ix limb length = " + str(np.linalg.norm(limblines[intersection_ix][1]-limblines[intersection_ix][0])),
                                     color=(0., 0, 0))'''

        '''self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 85,
                                 text="progress function = " + str(self.armSleeveProgress()),
                                 color=(0., 0, 0))'''

        #plane = self.CP0Feature.fitPlane()
        #plane.draw()
        #for v in self.splineCP0Verts:
         #   renderUtils.drawSphere(pos=self.clothScene.getVertexPos(vid=v))

        #test triangle/line segment intersection for neck/head
        '''tp0 = self.clothScene.getVertexPos(cid=0, vid=657)
        tp1 = self.clothScene.getVertexPos(cid=0, vid=109)
        tp2 = self.clothScene.getVertexPos(cid=0, vid=8)
        lp0 = self.robot_skeleton.bodynodes[16].to_world(np.array([0.,-0.05,0.])) #neck root
        lp1 = self.robot_skeleton.bodynodes[16].to_world(np.array([0.,0.28,0.])) #head root
        l_len = np.linalg.norm(lp1-lp0)
        intersection_distance = pyutils.triangleLineSegIntersect(tp0, tp1, tp2, lp0, lp1, False)
        #intersection_point = pyutils.triangleLineSegIntersect(tp0, tp1, tp2, lp0, lp1, True)
        distance_text = str(intersection_distance)
        if intersection_distance is not None:
            distance_text = str(l_len-intersection_distance)

        self.clothScene.drawText(x=360, y=self.viewer.viewport[3]-10,
                                 text="Head/neck intersection distance = " + distance_text,
                                 color=(0., 0, 0))
        #if intersection_distance is not None:
        #    renderUtils.drawSphere(intersection_point)
        renderUtils.drawTriangle(tp0,tp1,tp2)
        renderUtils.drawLines(lines=[[lp0,lp1]])'''
        self.clothScene.drawText(x=360, y=self.viewer.viewport[3] - 10,
                                 text="Head/neck intersection distance = " + str(self.headCollarContainment()),
                                 color=(0., 0, 0))

        #render force observations
        '''obs = self._get_obs()
        for i in range(22):
            f = obs[-66 + i*3:-66 + i*3 + 3]
            self.clothScene.drawText(x=360, y=60. + 15 * i,
                                     text="||f[" + str(i) + "]|| = " + str(f),
                                     color=(0., 0, 0))'''

        #target sample render
        '''reacher_range = 1.1
        hemisphereDir = np.array([-1.0, 0, 0])
        for i in range(100):
            target = self.hemisphereSample(maxradius=reacher_range, minradius=0.9, norm=hemisphereDir)
            while target[1] < 0:
                target = self.hemisphereSample(maxradius=reacher_range, minradius=0.9, norm=hemisphereDir)
            renderUtils.drawSphere(target)'''

        '''GL.glBegin(GL.GL_POLYGON)
        GL.glVertex3d(-0.53636903, 0.0341332, -0.0731871)
        GL.glVertex3d(-0.54120499, 0.038279, -0.0522403)
        GL.glVertex3d(-0.53161001, 0.0271241, -0.0933676)
        GL.glVertex3d(-0.50181198, -0.138647, -0.0973828)
        GL.glVertex3d(-0.49843001, -0.167151, -0.0937369)
        # GL.glVertex3d(-0.47647399, -0.174687, -0.0643502)
        GL.glVertex3d(-0.50627899, -0.175074, -0.0748381)
        GL.glVertex3d(-0.51507902, -0.14102399, -0.0483567)
        GL.glVertex3d(-0.52044398, -0.0982356, -0.0367543)
        GL.glVertex3d(-0.52189898, -0.0590227, -0.0141631)
        GL.glVertex3d(-0.527426, -0.0366101, -0.00950058)
        GL.glVertex3d(-0.53686303, 0.00799755, -0.0190606)
        GL.glVertex3d(-0.54120499, 0.038279, -0.0522403)
        GL.glEnd()'''

        #links = pyutils.getRobotLinks(self.robot_skeleton)
        GL.glBegin(GL.GL_LINES)
        for iter in self.ikLines:
            for l in iter:
                GL.glVertex3d(l[0][0], l[0][1], l[0][2])
                GL.glVertex3d(l[1][0], l[1][1], l[1][2])
        GL.glEnd()

        #draw spheres at body node 0's
        '''for ix, b in enumerate(self.robot_skeleton.bodynodes):
            GL.glColor3d(0,0,0)
            if ix == int(self.numSteps / 10):
                print("b("+str(ix)+"): "+str(self.robot_skeleton.bodynodes[ix]))
                GL.glColor3d(1, 0, 0)
            renderUtils.drawSphere(b.to_world(np.zeros(3)))
        '''
        #render debugging boxes
        if self.drawDebuggingBoxes:
            for ix,b in enumerate(self.debuggingBoxes):
                c = self.debuggingColors[ix]
                GL.glColor3d(c[0],c[1],c[2])
                b.draw()
                #for s in b.sample(50):
                #    self.viewer.drawSphere(p=s, r=0.01)

        #render the vertex handle_node(s)/Handle(s)
        if self.handleNode is not None:
            self.handleNode.draw()

        if self.gripper is not None:
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
        if self.renderForceText:
            HSF = self.clothScene.getHapticSensorObs()
            for i in range(self.clothScene.getNumHapticSensors()):
                self.clothScene.drawText(x=textX, y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
            textX += 160
        
        renderUtils.renderDofs(self.robot_skeleton, restPose=None, renderRestPose=False)
        #renderUtils.renderDofs(self.robot_skeleton, restPose=self.q_target, renderRestPose=True)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self.viewer.interactors[4].skelix = 2
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
            self.track_skeleton_id = 0

    def SPD(self, qt=None, Kp=None, Kd=None, dt=None):
        #compute the control torques to track qt using SPD

        #defaults
        if qt is None:
            qt = np.array(self.robot_skeleton.q)
        if Kp is None:
            Kp = np.identity(len(qt))*self.Kp_scale
        if Kd is None:
            Kd = np.identity(len(qt))*self.Kd_scale
        if dt is None:
            dt = self.timeStep

        invM = np.linalg.inv(self.robot_skeleton.mass_matrix() + Kd * dt)
        p = -Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * dt - qt)
        d = -Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.coriolis_and_gravity_forces() + p + d)
        T = p + d - Kd.dot(qddot * dt)

        return T

    def IK(self, nodeix=8, offset=np.zeros(3), target=None):
        if target is None:
            target = self.robot_skeleton.bodynodes[nodeix].to_world(offset)
        J = self.robot_skeleton.bodynodes[8].linear_jacobian(offset=offset, full=True)
        Jt = np.transpose(J)
        #pseudo inverse
        Jinv = Jt.dot(np.linalg.inv(J.dot(Jt)))
        #get global desired displacement
        disp = target-self.robot_skeleton.bodynodes[nodeix].to_world(offset)
        dQ = Jinv.dot(np.transpose(disp))
        return dQ

    def iterativeIK(self, nodeix=8, offset=np.zeros(3), target=None, iterations=1, restPoseWeight=0.5, restPose=None):
        if target is None:
            target = self.robot_skeleton.bodynodes[nodeix].to_world(offset)
        if restPose is None:
            #restPose = self.ikRestPose
            restPose = np.array(self.robot_skeleton.q)
        orgPose = np.array(self.robot_skeleton.q)
        for i in range(iterations):
            J = self.robot_skeleton.bodynodes[8].linear_jacobian(offset=offset, full=True)
            Jt = np.transpose(J)
            # pseudo inverse
            Jinv = Jt.dot(np.linalg.inv(J.dot(Jt)))
            # get global desired displacement
            disp = target - self.robot_skeleton.bodynodes[nodeix].to_world(offset)
            dQ = Jinv.dot(np.transpose(disp))
            dQrest = (restPose-self.robot_skeleton.q)*restPoseWeight
            newPose = self.robot_skeleton.q+dQ+dQrest
            newPose = np.maximum(self.robot_skeleton.position_lower_limits(), np.minimum(newPose, self.robot_skeleton.position_upper_limits()))
            self.robot_skeleton.set_positions(newPose)
        self.ikLines.append(pyutils.getRobotLinks(self.robot_skeleton))
        ikPose = np.array(self.robot_skeleton.q)
        self.robot_skeleton.set_positions(orgPose)
        return ikPose

    def iterativeMultiTargetIK(self, robot, nodeixs=[], offsets=[], targets=[], iterations=1, restPose=None, restPoseWeight=0.1):
        #iterative Jacobian pseudo inverse IK with rest pose objective
        orgPose = np.array(robot.q)
        if restPose is None:
            restPose = np.array(robot.q)

        for i in range(iterations):
            dQ = np.zeros(len(robot.q))
            for ix, n in enumerate(nodeixs):
                J = robot.bodynodes[n].linear_jacobian(offset=offsets[ix], full=True)
                Jt = np.transpose(J)
                # pseudo inverse
                Jinv = Jt.dot(np.linalg.inv(J.dot(Jt)))
                # get global desired displacement
                disp = targets[ix] - robot.bodynodes[n].to_world(offsets[ix])
                dQ += Jinv.dot(np.transpose(disp))
            dQrest = (restPose - robot.q) * restPoseWeight
            newPose = robot.q + dQ + dQrest
            newPose = np.maximum(robot.position_lower_limits(), np.minimum(newPose, robot.position_upper_limits()))
            robot.set_positions(newPose)
        ikPose = np.array(robot.q)
        self.ikLines = [pyutils.getRobotLinks(robot)]
        robot.set_positions(orgPose)
        return ikPose

    def testMeshGraph(self):
        print("Testing MeshGraph")

        self.separatedMesh.initSeparatedMeshGraph()
        self.separatedMesh.updateWeights()
        self.separatedMesh.computeGeodesic(feature=self.CP0Feature, oneSided=True)

        print("done")
        #exit()
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t


