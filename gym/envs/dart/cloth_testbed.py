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



class DartClothTestbedEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.prefix = os.path.dirname(__file__)
        self.useOpenGL = True
        self.totalTime = 0
        self.lastTime = 0

        #self.timeGraph = pyutils.LineGrapher("Step Time")

        #22 dof upper body
        self.action_scale = np.ones(22)*10
        self.control_bounds = np.array([np.ones(22), np.ones(22)*-1])

        self.numSteps = 0 #increments every step, 0 on reset

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
        self.totalTime = 0.

        self.enforceTauLimits = False
        self.tau_limits = [np.ones(22) * -10, np.ones(22) * 10]
        self.tau_limits[0][0] = -250
        self.tau_limits[1][0] = 250
        self.tau_limits[0][1] = -250
        self.tau_limits[1][1] = 250
        self.tau_limits[0][2] = -250
        self.tau_limits[1][2] = 250

        self.tau_limits[0][3] = -150
        self.tau_limits[1][3] = 150
        self.tau_limits[0][4] = -150
        self.tau_limits[1][4] = 150
        self.tau_limits[0][5] = -60
        self.tau_limits[1][5] = 60
        self.tau_limits[0][6] = -60
        self.tau_limits[1][6] = 60
        self.tau_limits[0][7] = -50
        self.tau_limits[1][7] = 50
        self.tau_limits[0][8] = -50
        self.tau_limits[1][8] = 50

        self.tau_limits[0][11] = -150
        self.tau_limits[1][11] = 150
        self.tau_limits[0][12] = -150
        self.tau_limits[1][12] = 150
        self.tau_limits[0][13] = -60
        self.tau_limits[1][13] = 60
        self.tau_limits[0][14] = -60
        self.tau_limits[1][14] = 60
        self.tau_limits[0][15] = -50
        self.tau_limits[1][15] = 50
        self.tau_limits[0][16] = -50
        self.tau_limits[1][16] = 50

        self.action_scale = np.array(self.tau_limits[1])

        self.graphTau = False
        self.tauGraphDof = 4 #if not none, graph only one dof
        if self.graphTau:
            self.tauGraph = pyutils.LineGrapher(title="||Tau||")

        self.graphError = False
        self.errorGraphDof = 4 # if not none, graph only one dof
        if self.graphError:
            self.errorGraph = pyutils.LineGrapher(title="||Error||")

        #self.graphTimerGraph = pyutils.LineGrapher(title="Graph Time")

        #interactive handle mode
        self.interactiveHandleNode = False

        self.updateHandleNodeFrom = -1
        #self.updateHandleNodeFrom = 14 #if -1, this feature is disabled
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
                                        #mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/tshirt_m.obj",
                                        #state_path =self.prefix + "/../../../../tshirt_regrip1.obj",
                                        state_path=self.prefix + "/../../../../1stSleeveState.obj",
                                        #state_path="/home/alexander/Documents/dev/tshirt_regrip1.obj",
                                        scale=1.4)

        clothScene.togglePinned(0,0) #turn off auto-pin
        #clothScene.togglePinned(0, 144)
        #clothScene.togglePinned(0, 190)

        self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)
        #self.CP0Verts = [2621, 2518, 2576, 2549, 2554, 2544, 2616, 2558, 2519, 2561, 2593]
        self.CP0Verts = [2, 2561, 2476, 2519, 2479, 2634, 2620, 2553, 2619, 2464, 2668, 2642, 2654, 2554, 2549, 2576, 2532, 2585, 2624, 2487, 2633, 2559, 2593]
        self.CP0Feature = ClothFeature(verts=self.CP0Verts, clothScene=clothScene)

        self.reset_number = 0  # increments on env.reset()

        #intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4, observation_size=(44+66), action_bounds=self.control_bounds)#, disableViewer=True, visualize=False)
            utils.EzPickle.__init__(self)
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4,
                                  observation_size=(44 + 66),
                                  action_bounds=self.control_bounds, disableViewer=True, visualize=False)
        utils.EzPickle.__init__(self)

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

    def _step(self, a):
        #starttime = time.time()
        #print("     step " +str(self.numSteps))
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

        #apply action and simulate
        #starttime = time.time()
        self.do_simulation(tau, self.frame_skip)
        #endtime = time.time()

        minContactGeodesic = pyutils.getMinContactGeodesic(sensorix=21, clothscene=self.clothScene, meshgraph=self.separatedMesh)
        print("minContactGeodesic = " + str(minContactGeodesic))

        reward = 0
        ob = self._get_obs()
        s = self.state_vector()
        
        #update physx capsules

        self.updateClothCollisionStructures(hapticSensors=True)

        #check termination conditions
        done = False

        #graphing force
        #starttime = time.time()
        if self.graphTau and self.reset_number > 0:
            if self.tauGraphDof is None:
                self.tauGraph.addToLinePlot(data=[np.linalg.norm(tau)])
            else:
                self.tauGraph.addToLinePlot(data=[tau[self.tauGraphDof]])

        if self.graphError and self.reset_number > 0:
            dif = self.robot_skeleton.q - self.q_target
            if self.errorGraphDof is None:
                self.errorGraph.addToLinePlot(data=[np.linalg.norm(dif)])
            else:
                self.errorGraph.addToLinePlot(data=[dif[self.errorGraphDof]])
        self.numSteps += 1
        #endtime = time.time()
        #self.graphTimerGraph.addToLinePlot(data=[endtime-starttime])
        #self.totalTime += ((endtime - starttime) * 1000)
        #self.timeGraph.addToLinePlot(data=[(endtime - starttime) * 1000])
        #self.IK()
        return ob, reward, done, {}

    def _get_obs(self):
        '''get_obs'''
        f_size = 66
        theta = self.robot_skeleton.q

        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        else:
            f = np.zeros(f_size)

        #obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq, vec, self.target, f]).ravel()
        obs = np.concatenate([theta, self.robot_skeleton.dq, f]).ravel()
        return obs

    def reset_model(self):
        #starttime = time.time()
        #print("reset " +str(self.reset_number))
        '''reset_model'''
        #if self.reset_number > 0:
        #    self.enforceTauLimits = True
        #if self.reset_number > 0:
        #    print("average step time = " + str(self.totalTime/self.numSteps))
        self.totalTime = 0
        self.numSteps = 0
        self.dart_world.reset()
        self.clothScene.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
        '''qpos = [  2.18672993e-03,  -4.72069042e-03,  -1.13570380e-02,   1.49440348e-02,
   1.29127624e-02,  -1.14191071e-02,   6.39350903e-03,  -2.15463769e-03,
  -8.75123999e-03,   1.12041144e-02,  -3.17658842e-02,  -1.11111111e-02,
   8.33333333e-02,   1.16666667e-01,   1.12604700e+00,   6.30000000e-01,
   1.38555556e+00,   6.00000000e-01,  -7.23481822e-02,  -7.26463958e-03,
  -8.24054408e-03,  -1.29043413e-03] + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)'''
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.025, high=.025, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        #self.q_target = np.array(self.robot_skeleton.q)
        #self.q_target[8] = 1.0

        if self.reset_number == 0:
            self.q_target = pyutils.getRandomPose(robot=self.robot_skeleton)
            self.q_target[0] = 0.39
            self.q_target = [ 0.39294164, -0.28923569,  0.44674558,  0.21799112,  0.22719583, -0.34199703,
 -0.8429895,   1.50691939,  1.28366487,  0.54321876, -0.36126751,  0.11050672,
 -0.12107801, -1.6634833,   1.03492742,  1.512525,    0.22033149,  0.48043698,
  0.28899992,  0.15556353, -0.02484711,  0.19311922]
        self.ikLines.append(pyutils.getRobotLinks(robot=self.robot_skeleton, pose=self.q_target))

        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 0, 1.])))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=3.14, axis=np.array([0, 1., 0.])))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=-1., axis=np.array([1., 0., 0.])))
        #self.clothScene.translateCloth(0, np.array([0.75, -0.5, -0.5]))  # shirt in front of person
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))

        #self.clothScene.translateCloth(0, np.array([5.75, -0.5, -0.5]))  # get the cloth out of the way
        self.clothScene.setSelfCollisionDistance(0.025)

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
        self.dart_world.skeletons[0].q = [0, 0, 0, self.ikTarget[0], self.ikTarget[1], self.ikTarget[2]]
        self.ikRestPose = np.array(self.robot_skeleton.q)
        self.ikRestPose[8] += 2.5 #elbow bend
        self.ikRestPose[6] += 1.0

        self.handleNode.clearHandles()
        #self.handle_node.addVertices(verts=[570, 1041, 993, 1185, 285, 1056, 283, 958, 905, 711, 435, 992, 50, 935, 489, 787, 327, 362, 676, 842, 873, 887, 54, 55])
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


        self.reset_number += 1

        #self.handle_node.reset()
        if self.handleNode is not None:
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.recomputeOffsets()

        if self.gripper is not None:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        self.CP0Feature.fitPlane()
        self.testMeshGraph()

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
        
    def extraRenderFunction(self):
        #print("extra render function")

        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        #draw meshGraph stuff
        for n in self.separatedMesh.nodes:
            vpos = self.clothScene.getVertexPos(vid=n.vix)
            norm = self.clothScene.getVertNormal(cid=0, vid=n.vix)
            offset = 0.005
            c=n.ix/len(self.separatedMesh.nodes)
            if self.separatedMesh.maxGeo > 0:
                c = n.geodesic/self.separatedMesh.maxGeo
            renderUtils.setColor(color=(c,c,c))
            if n.side == 0:
                renderUtils.drawSphere(pos=vpos+norm*offset, rad=0.005)
            else:
                renderUtils.drawSphere(pos=vpos - norm * offset, rad=0.005)

        #test plane
        org = self.robot_skeleton.bodynodes[8].to_world(np.zeros(3))
        normal = self.robot_skeleton.bodynodes[8].to_world(np.array([0.0,-0.1,0.0])) - org
        normal /= np.linalg.norm(normal)
        basis1 = self.robot_skeleton.bodynodes[8].to_world(np.array([0.0,0.,0.1])) - org
        basis2 = self.robot_skeleton.bodynodes[8].to_world(np.array([0.1, 0., 0.0])) - org
        plane = Plane(org=org, normal=normal, b1=basis1, b2=basis2)
        plane.draw()

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

        if self.numSteps > 0:
            renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.q_target, renderRestPose=True)

        self.clothScene.drawText(x=15., y=30., text="Time = " + str(self.numSteps * 0.04), color=(0., 0, 0))
        self.clothScene.drawText(x=15 , y=45., text='Friction: %.2f' % self.clothScene.getFriction(), color=(0., 0, 0))

        #    self.clothScene.drawText(x=15, y=620., text='Average SPD time = %.2f' % float(self.totalTime/self.reset_number), color=(0., 0, 0))

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
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


