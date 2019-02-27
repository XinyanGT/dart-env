# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.fullbodydatadriven_cloth_base import *
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

class SPDController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.002
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([1600.0] * (ndofs))
        self.Kd = np.diagflat([16.0] * (ndofs))

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
        self.env.frameskip = 1
        self.env.SPDTorqueLimits = True
        #reset the target
        #cur_q = np.array(self.skel.q)
        #self.env.loadCharacterState(filename="characterState_regrip")
        self.target = np.array([ 0., 0., 0., 0., 0., 0., 0., 0., 0.302, 0., 0., 0., 0., 0., 0., 0., 0.302, 0., 0., 0., 0., 0.])
        self.env.restPose = np.array(self.target)
        #self.target = np.array(self.skel.q)
        #self.env.robot_skeleton.set_positions(cur_q)

        #clear the handles
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles();
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
        if pDist < 0.1:
            return True
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

class DartClothUpperBodyDataDrivenClothTshirtMasterEnv(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True
        self.simpleUI = True
        dt = 0.002
        frameskip = 5

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.resetTime = 0
        self.save_state_on_control_switch = False #if true, the cloth and character state is saved when controllers are switched
        self.state_save_directory = "saved_control_states_shorts/"
        self.renderGeodesic             = False
        self.renderOracle               = False
        self.renderRestPose             = False

        #other variables
        self.restPose = None
        self.prevErrors = None #stores the errors taken from DART each iteration
        self.limbProgress = -1
        self.fingertip = np.array([0, -0.08, 0])
        self.stabilityPolygonCentroid = np.zeros(3)
        self.projectedCOM = np.zeros(3)
        self.footBodyNode = 20  # 17 left, 20 right
        self.prevOracle = None
        self.prevWaistContainment = 0
        self.footOffsets = [np.array([0, 0, -0.2]), np.array([0.05, 0, 0.03]), np.array([-0.05, 0, 0.03])] #local positions of the foot to query for foot location reward
        self.footTargets = [] #world foot position targets for the foo location reward
        self.targetCOM = np.zeros(3) #target for the center of the stability region



        self.actuatedDofs = np.arange(34)
        observation_size = 34 * 3 + 6  # q[6:](sin,cos), dq         #(0,108)
        observation_size += 40*3 #haptics                           #(108,228)
        observation_size += 40 #contact IDs                         #(228,268)
        observation_size += 3 #COM                                  #(268,271)
        observation_size += 6 #garment feature                      #(271,277)
        observation_size += 3 #oracle                               #(277,280)
        observation_size += 2 #limb progress                        #(280,282)

        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="shorts_med.obj",
                                                          clothScale=np.array([0.9,0.9,0.9]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          gravity=False,
                                                          dt=dt,
                                                          frameskip=frameskip,
                                                          lockedLFoot=True)

        #define shorts garment features
        self.targetGripVerticesL = [85, 22, 13, 92, 212, 366]
        self.targetGripVerticesR = [5, 146, 327, 215, 112, 275]
        self.legEndVerticesL = [42, 384, 118, 383, 37, 164, 123, 404, 36, 406, 151, 338, 38, 235, 81, 266, 40, 247, 80, 263]
        self.legEndVerticesR = [45, 394, 133, 397, 69, 399, 134, 401, 20, 174, 127, 336, 21, 233, 79, 258, 29, 254, 83, 264]
        self.legMidVerticesR = [91, 400, 130, 396, 128, 162, 165, 141, 298, 417, 19, 219, 270, 427, 440, 153, 320, 71, 167, 30, 424]
        self.legMidVerticesL = [280, 360, 102, 340, 196, 206, 113, 290, 41, 178, 72, 325, 159, 147, 430, 291, 439, 55, 345, 125, 429]
        self.waistVertices = [215, 278, 110, 217, 321, 344, 189, 62, 94, 281, 208, 107, 188, 253, 228, 212, 366, 92, 160, 119, 230, 365, 77, 0, 104, 163, 351, 120, 295, 275, 112]
        #leg entrance L and R
        self.legStartVerticesR = [232, 421, 176, 82, 319, 403, 256, 314, 98, 408, 144, 26, 261, 84, 434, 432, 27, 369, 132, 157, 249, 203, 99, 184, 437]
        self.legStartVerticesL = [209, 197, 257, 68, 109, 248, 238, 357, 195, 108, 222, 114, 205, 86, 273, 35, 239, 137, 297, 183, 105]


        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene)
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene)

        self.legEndFeatureL = ClothFeature(verts=self.legEndVerticesL, clothScene=self.clothScene)
        self.legEndFeatureR = ClothFeature(verts=self.legEndVerticesR, clothScene=self.clothScene)
        self.legMidFeatureL = ClothFeature(verts=self.legMidVerticesL, clothScene=self.clothScene)
        self.legMidFeatureR = ClothFeature(verts=self.legMidVerticesR, clothScene=self.clothScene)
        self.legStartFeatureR = ClothFeature(verts=self.legStartVerticesR, clothScene=self.clothScene)
        self.legStartFeatureL = ClothFeature(verts=self.legStartVerticesL, clothScene=self.clothScene)

        self.waistFeature = ClothFeature(verts=self.waistVertices, clothScene=self.clothScene)

        #variables for returning feature obs
        self.focusFeature = None        #if set, this feature centroid is used to get the "feature" obs
        self.focusFeatureNode = None    #if set, this body node is used to fill feature displacement obs
        self.progressFeature = None     #if set, this feature is used to fill oracle normal and check arm progress
        self.contactSensorIX = None     #if set, used to compute oracle

        self.simulateCloth = clothSimulation

        # handle nodes
        self.handleNodes = []
        self.updateHandleNodesFrom = [7, 12]

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        #controller initialzation
        self.controllers = [
            #TODO
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
        self.limbProgress = -1 #reset this in controller

        self.projectedCOM = np.array(self.robot_skeleton.com())
        self.COMHeight = self.projectedCOM[1]
        self.projectedCOM[1] = -1.3

        #update feature planes
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()
        self.legEndFeatureL.fitPlane()
        self.legEndFeatureR.fitPlane()
        self.legMidFeatureL.fitPlane()
        self.legMidFeatureR.fitPlane()
        self.legStartFeatureL.fitPlane()
        self.legStartFeatureR.fitPlane()
        self.waistFeature.fitPlane()

        if len(self.handleNodes) > 1 and self.reset_number > 0:
            self.handleNodes[0].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[0]].T)
            self.handleNodes[1].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[1]].T)
            self.handleNodes[0].step()
            self.handleNodes[1].step()

        self.additionalAction = np.zeros(22) #reset control
        #update controller specific variables and produce
        if self.currentController is not None:
            self.controllers[self.currentController].update()
            if self.controllers[self.currentController].transition():
                changed = self.currentController
                self.currentController = min(len(self.controllers)-1, self.currentController+1)
                changed = (changed != self.currentController)
                if changed:
                    self.controllers[self.currentController].setup()
                    self.controllers[self.currentController].update()
            obs = self._get_obs()
            self.additionalAction = self.controllers[self.currentController].query(obs)

        self.stepsSinceControlSwitch += 1
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, 0
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, 0

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        #unnecessary for master control env... not meant for training
        return 0

    def _get_obs(self):
        f_size = 120
        orientation = np.array(self.robot_skeleton.q[:3])
        theta = np.array(self.robot_skeleton.q[6:])
        dq = np.array(self.robot_skeleton.dq)
        trans = np.array(self.robot_skeleton.q[3:6])

        obs = np.concatenate([np.cos(theta), np.sin(theta), dq]).ravel()

        #haptics
        f = np.zeros(f_size)
        if self.simulateCloth and self.hapticsAware:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        obs = np.concatenate([obs, f]).ravel()

        #contactIDs
        HSIDs = self.clothScene.getHapticSensorContactIDs()
        obs = np.concatenate([obs, HSIDs]).ravel()

        # COM
        com = np.array(self.robot_skeleton.com()).ravel()
        obs = np.concatenate([obs, com]).ravel()

        #feature
        if self.focusFeature is None or self.focusFeatureNode is None:
            obs = np.concatenate([obs, np.zeros(6)]).ravel()
        else:
            centroid = self.focusFeature.plane.org
            ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.toeOffset)
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
                    # new: oracle points to the waist feature centroid when not in contact with cloth
                    target = self.waistFeature.plane.org
                    ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.toeOffset)
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

        #limb progress
        obs = np.concatenate([obs, [max(-1, self.limbProgress), max(-1, self.prevWaistContainment)]]).ravel()

        return obs

    def additionalResets(self):
        count = 0
        recordForRenderingDirectory = "saved_render_states/shortsseq" + str(count)
        while(os.path.exists(recordForRenderingDirectory)):
            count += 1
            recordForRenderingDirectory = "saved_render_states/shortsseq" + str(count)
        self.recordForRenderingOutputPrefix = recordForRenderingDirectory+"/shortsseq"
        if self.recordForRendering:
            if not os.path.exists(recordForRenderingDirectory):
                os.makedirs(recordForRenderingDirectory)

        self.resetTime = time.time()
        self.currentController = 0

        qpos = np.array(
            [0.0780131557357, -0.142593660368, 0.143019989259, 0.144666815206, -0.035, 0.165147835811, 0.0162260416596,
             0.19115105848, -0.0299336428088, 0.0445035430603, -0.025419636699, -0.878286887463, -0.485843951506,
             0.239911240107, 1.48781704099, 0.00147260210175, -3.84887833923e-05, -0.0116786422327, 0.0287998551014,
             0.424678918993, -1.20629912179, 0.675013212728, 0.936068431591, -0.118766088348, 0.130936683699,
             -0.00550651147978, 0.0111253708206, 0.000890767938847, -0.130121733054, -0.195712660157, 0.413533717103,
             0.588166252597, 0.281757292531, 0.0899107535319, -0.625904521458, -1.56979781802, 0.202940224704,
             2.14854759605, -0.171377608919, 0.163232950118])

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.restPose = qpos

        RX = pyutils.rotateX(-1.56)
        self.clothScene.rotateCloth(cid=0, R=RX)
        self.clothScene.translateCloth(cid=0, T=np.array([0.555, -0.5, -1.45]))

        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()
        self.legEndFeatureL.fitPlane()
        self.legEndFeatureR.fitPlane()
        self.legMidFeatureL.fitPlane()
        self.legMidFeatureR.fitPlane()
        self.legStartFeatureL.fitPlane()
        self.legStartFeatureR.fitPlane()
        self.waistFeature.fitPlane()

        # sort out feature normals
        CPLE_CPLM = self.legEndFeatureL.plane.org - self.legMidFeatureL.plane.org
        CPLS_CPLM = self.legStartFeatureL.plane.org - self.legMidFeatureL.plane.org

        CPRE_CPRM = self.legEndFeatureR.plane.org - self.legMidFeatureR.plane.org
        CPRS_CPRM = self.legStartFeatureR.plane.org - self.legMidFeatureR.plane.org

        estimated_groin = (self.legStartFeatureL.plane.org + self.legStartFeatureR.plane.org) / 2.0
        CPW_EG = self.waistFeature.plane.org - estimated_groin

        if CPW_EG.dot(self.waistFeature.plane.normal) > 0:
            self.waistFeature.plane.normal *= -1.0

        if CPLS_CPLM.dot(self.legStartFeatureL.plane.normal) > 0:
            self.legStartFeatureL.plane.normal *= -1.0

        if CPLE_CPLM.dot(self.legEndFeatureL.plane.normal) < 0:
            self.legEndFeatureL.plane.normal *= -1.0

        if CPLE_CPLM.dot(self.legMidFeatureL.plane.normal) < 0:
            self.legMidFeatureL.plane.normal *= -1.0

        if CPRS_CPRM.dot(self.legStartFeatureR.plane.normal) > 0:
            self.legStartFeatureR.plane.normal *= -1.0

        if CPRE_CPRM.dot(self.legEndFeatureR.plane.normal) < 0:
            self.legEndFeatureR.plane.normal *= -1.0

        if CPRE_CPRM.dot(self.legMidFeatureR.plane.normal) < 0:
            self.legMidFeatureR.plane.normal *= -1.0

        if len(self.handleNodes) > 1:
            self.handleNodes[0].clearHandles()
            self.handleNodes[0].addVertices(verts=self.targetGripVerticesR)
            self.handleNodes[0].setOrgToCentroid()
            self.handleNodes[0].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[0]].T)
            self.handleNodes[0].recomputeOffsets()

            self.handleNodes[1].clearHandles()
            self.handleNodes[1].addVertices(verts=self.targetGripVerticesL)
            self.handleNodes[1].setOrgToCentroid()
            self.handleNodes[1].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[1]].T)
            self.handleNodes[1].recomputeOffsets()

        self.controllers[self.currentController].setup()

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

        if not self.simpleUI:
            self.gripFeatureL.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
            self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
            self.legEndFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
            self.legEndFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
            self.legMidFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
            self.legMidFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
            self.legStartFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
            self.legStartFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
            self.waistFeature.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 0.0, 1.0])

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

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
            ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.toeOffset)
            renderUtils.drawArrow(p0=ef, p1=ef + self.prevOracle)

        textHeight = 15
        textLines = 2

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
            textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=self.renderRestPose)

            if self.stepsSinceControlSwitch < 50 and len(self.controllers) > 0:
                label = self.controllers[self.currentController].name
                self.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60,
                                         text="Active Controller = " + str(label), color=(0., 0, 0))
