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

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

''' This env is setup for upper body single arm reduced action space learning with draped shirt'''

class DartClothShirtReacherEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        
        #22 dof upper body
        self.action_scale = np.ones(11)*10
        self.control_bounds = np.array([np.ones(11), np.ones(11)*-1])
        self.arm = 1  # if 1, left arm (character's perspective), if 2, right
        
        self.doSettle = False
        self.settlePeriod = 100
        self.doInterpolation = False
        self.interpolationPeriod = 200
        self.poseSpline = pyutils.Spline()
        #self.interpolationGoal = np.zeros(22)
        #self.interpolationStart = np.zeros(22)
        self.numSteps = 0 #increments every step, 0 on reset

        #cloth inflator functionality
        self.inflatorActive = False
        self.renderInflator = True
        self.inflatorP1 = np.array([0., 0, -0.15]) #start position
        self.inflatorP2 = np.array([0.1, 0, -0.2]) #end position
        self.inflatorR1 = 0.01 #start radius
        self.inflatorR2 = 0.15 #end radius
        self.inflaterT = 0 #steps since inflation start
        self.inflatorDuration = 50 #duration of inflation
        self.inflatorEnd = 75

        self.mirror = False
        self.mirrorT = np.zeros(3)
        self.mirrorV = np.zeros(3)
        self.mirrorF = np.zeros(66)

        # handle node setup
        self.handleNode = None
        self.gripper = None

        #friction test variables
        self. collectFrictionTestData = False
        self.hapticsOn = True
        self.lowestDistFile = "lowestdistances_haptics_reacher.txt"
        self.maxDeformFile = "maxDeformation_haptics_reacher.txt"
        self.runtimesFile = "runtimes_haptics_reacher.txt"
        self.lowestDist = 9999.
        self.maxDeform = 0
        self.lowestDists = []
        self.maxDeforms = []
        self.runtimes = []

        #create cloth scene
        #clothScene = pyphysx.ClothScene(step=0.01, mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/test_sleeve_plane.obj", scale=1.6)
        #clothScene = pyphysx.ClothScene(step=0.01, mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/tshirt_m.obj", scale=1.6)
        clothScene = pyphysx.ClothScene(step=0.01,
                                        mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/tshirt_m.obj",
                                        state_path="/home/alexander/Documents/dev/1stSleeveState.obj",
                                        #state_path="/home/alexander/Documents/dev/end1stSleeveSuccess.obj",
                                        #state_path="/home/alexander/Documents/dev/2ndSleeveInitialState.obj",
                                        scale=1.6)

        #clothScene = pyphysx.ClothScene(step=0.01, mesh_path="/home/alexander/Documents/dev/tshirt_m.obj", scale=1.6)
        clothScene.togglePinned(0,0) #turn off auto-bottom pin
        #clothScene.togglePinned(0,9)
        #clothScene.togglePinned(0,10)
        #clothScene.togglePinned(0,37)
        #clothScene.togglePinned(0,42)
        #clothScene.togglePinned(0,44)
        #clothScene.togglePinned(0,48)
        #clothScene.togglePinned(0,51)
        #clothScene.togglePinned(0,54)
        #clothScene.togglePinned(0,58)
        #clothScene.togglePinned(0,64)
        
        '''clothScene.togglePinned(0,111) #collar
        clothScene.togglePinned(0,113) #collar
        clothScene.togglePinned(0,117) #collar
        clothScene.togglePinned(0,193) #collar
        clothScene.togglePinned(0,112) #collar
        clothScene.togglePinned(0,114) #collar
        clothScene.togglePinned(0,115) #collar
        clothScene.togglePinned(0,116) #collar
        clothScene.togglePinned(0,192) #collar
        clothScene.togglePinned(0,191) #collar'''
        
        '''clothScene.togglePinned(0,190) #collar
        clothScene.togglePinned(0,189) #collar
        clothScene.togglePinned(0,188) #collar
        clothScene.togglePinned(0,187) #collar
        clothScene.togglePinned(0,186) #collar
        clothScene.togglePinned(0,110) #collar
        clothScene.togglePinned(0,109) #collar
        clothScene.togglePinned(0,108) #collar
        clothScene.togglePinned(0,107) #collar'''
        
        #clothScene.togglePinned(0,144) #bottom
        #clothScene.togglePinned(0,147) #bottom
        #clothScene.togglePinned(0,149) #bottom
        #clothScene.togglePinned(0,153) #bottom
        #clothScene.togglePinned(0,155) #bottom
        #clothScene.togglePinned(0,161) #bottom
        #clothScene.togglePinned(0,165) #bottom
        #clothScene.togglePinned(0,224) #right sleeve
        #clothScene.togglePinned(0,229) #right sleeve
        #clothScene.togglePinned(0,233) #right sleeve
        #clothScene.togglePinned(0,236) #sleeve
        #clothScene.togglePinned(0,240) #sleeve
        #clothScene.togglePinned(0,246) #sleeve
        
        '''clothScene.togglePinned(0,250) #left sleeve
        clothScene.togglePinned(0,253) #left sleeve
        clothScene.togglePinned(0,257) #left sleeve
        clothScene.togglePinned(0,259) #left sleeve
        clothScene.togglePinned(0,262) #left sleeve
        clothScene.togglePinned(0,264) #left sleeve'''
        
        
        #intialize the parent env
        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4, observation_size=(66+66+6), action_bounds=self.control_bounds)#, disableViewer=True, visualize=False)
        #DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4,
        #                      observation_size=(66 + 66 + 6), action_bounds=self.control_bounds, visualize=False)

        #TODO: additional observation size for force
        utils.EzPickle.__init__(self)

        self.handleNode = HandleNode(self.clothScene)
        #self.handle_node.addTarget(t=1., pos=np.array([0., 0., 0.5]), orientation=np.normalized(np.quaternion(1,1,0,0)))
        #self.handle_node.addTarget(t=2., pos=np.array([0.25, 0.5, 0.5]), orientation=np.normalized(np.quaternion(1, 0, 1, 0)))
        #self.handle_node.targetSpline.insert(t=1., p=np.array([0.,0.,0.5]))
        #self.handle_node.targetSpline.insert(t=2., p=np.array([0.25, 0.5, 0.5]))
        #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
        self.handleNode.setTranslation(np.array([0.0, 0.2, -0.75]))
        #self.handle_node.addVertices([112,113,114,115,116,117,186,187,188,189,190,191,192,193]) #collar
        #self.handle_node.addVertex(0)
        #self.handle_node.addVertex(30)

        #self.gripper = pyutils.BoxFrame(c0=np.array([0.06, -0.075, 0.06]), c1=np.array([-0.06, -0.125, -0.06]))
        self.gripper = pyutils.EllipsoidFrame(c0=np.array([0,-0.1,0]), dim=np.array([0.05,0.025,0.05]))
        self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)

        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.4)
        #self.clothScene.setFriction(0, 0.0)
        
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = True
        self.sampleFromHemisphere = True
        self.rotateCloth = False
        self.randomRoll = False
        
        self.trackSuccess = False
        self.renderSuccess = False
        self.targetHistory = []
        self.successHistory = []
        
        self.renderDofs = True #if true, show dofs text 
        self.renderForceText = False
        
        self.random_dir = np.array([0,0,1.])
        
        self.reset_number = 0 #debugging

        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])

        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

        print("done init")

    def limits(self, dof_ix):
        return np.array([self.robot_skeleton.dof(dof_ix).position_lower_limit(), self.robot_skeleton.dof(dof_ix).position_upper_limit()])
        
    def poseInterpolate(self, q0, q1, t):
        'interpolate the pose q0->q1 over t=[0,1]'
        qpos = LERP(q0,q1,t)
        self.robot_skeleton.set_positions(qpos)
        
    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)
        
    def loadObjState(self):
        self.clothScene.loadObjState("objState", 0)

    def printPose(self):
        print("q = " + str(self.robot_skeleton.q))
        
    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        #fingertip = np.array([0.0, -0.25, 0.0])
        fingertip = np.array([0.0, -0.06, 0.0])
        #print("Transform = " + str(self.robot_skeleton.bodynodes[8].T))

        wFingertip1 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        if self.arm == 2 or self.mirror is True:
            wFingertip1 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vec1 = self.target-wFingertip1
        
        #execute special actions
        if self.doSettle and self.numSteps < self.settlePeriod:
            tau = np.zeros(len(tau))
        elif self.doInterpolation:
            t = self.numSteps/self.interpolationPeriod
            if self.doSettle:
                t = (self.numSteps-self.settlePeriod)/self.interpolationPeriod
            if t < 1:
                tau = np.zeros(len(tau))
                qpos = self.poseSpline.pos(t)
                self.robot_skeleton.set_positions(qpos)
                #self.poseInterpolate(self.interpolationStart, self.interpolationGoal, t)
            elif t<1.1:
                print(t)

        if self.handleNode is not None:
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.step()

        if self.gripper is not None:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)
        
        #apply action and simulate
        if self.arm == 1:
            tau = np.concatenate([tau, np.zeros(11)])
        else:
            if self.mirror is True:
                tau[0] *= -1
                tau[2] *= -1
            tau = np.concatenate([tau[:3], np.zeros(8), tau[3:], np.zeros(3)])
        self.do_simulation(tau, self.frame_skip)
        
        wFingertip2 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        if self.arm == 2 or self.mirror is True:
            wFingertip2 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vec2 = self.target-wFingertip2
        
        reward_dist = - np.linalg.norm(vec2)
        reward_ctrl = - np.square(tau).sum() * 0.001
        reward_progress = np.dot((wFingertip2 - wFingertip1), vec1/np.linalg.norm(vec1)) * 100
        alive_bonus = -0.001
        reward_prox = 0
        #if -reward_dist < 0.1:
        #    reward_prox += (0.1+reward_dist)*10
        reward = reward_ctrl + alive_bonus + reward_progress + reward_prox
        #reward = reward_dist + reward_ctrl
        
        ob = self._get_obs()

        s = self.state_vector()
        
        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        
        #check cloth deformation for termination
        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)

        if self.collectFrictionTestData:
            targetDist = np.linalg.norm(vec2)
            if targetDist < self.lowestDist:
                self.lowestDist = targetDist
            if clothDeformation > self.maxDeform:
                self.maxDeform = clothDeformation

        
        #check termination conditions
        done = False
        if not np.isfinite(s).all():
            done = True
            reward -= 500
        elif -reward_dist < 0.1:
            done = True
            reward += 100
        elif (clothDeformation > 14):
            if not self.doSettle or self.numSteps>self.settlePeriod:
                #done = True
                reward -= 500

        self.numSteps += 1
        if self.inflatorActive:
            self.inflaterT += 1
            if self.inflaterT > self.inflatorEnd:
                self.inflatorR1 = self.inflatorR2
                self.inflatorR2 = 0.
                self.inflatorEnd = 99999
                self.inflatorDuration = 25
                self.inflatorT = 0

        #self collision checking

        return ob, reward, done, {}

    def _get_obs(self):
        '''get_obs'''
        f_size = 66
        theta = self.robot_skeleton.q
        #fingertip = np.array([0.0, -0.25, 0.0])
        fingertip = np.array([0.0, -0.06, 0.0])
        vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
        if self.arm == 2 or self.mirror is True:
            vec = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target
        #'''
        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
            if self.mirror is True:
                #print(f)
                #mirror incoming force about y/z plane and mirror symmetric haptic sensors
                for i in range(int(len(f)/3)):
                    f[i*3] *= -1
                #now re-organize haptic sensors
                fp = np.array(f)
                for i in range(4, int((len(f) / 3)-9)):
                    fp[i * 3:i * 3+2] = f[((i * 3) + 9*3):((i * 3) + 9*3)+2]
                    fp[((i * 3) + 9 * 3):((i * 3) + 9 * 3) + 2] = f[i * 3:i * 3 + 2]
                f = fp
                #print(f)
        else:
            f = np.zeros(f_size)

        if self.hapticsOn is False:
            f = np.zeros(f_size)
        #'''
        #f = np.zeros(f_size)

        #print("ID getobs:" + str(self.clothScene.id))
        #print("f: " + str(f))
        #print("len f = " + str(len(f)))
        if self.mirror is True:
            #flip the x for mirroring
            vec[0] *= -1
            thetaMirror = np.array(theta)
            dqMirror = np.array(self.robot_skeleton.dq)
            #reflect the pose for mirroring
            thetaMirror[0] *= -1
            thetaMirror[2] *= -1
            thetaMirror[3:11] = theta[11:19]
            thetaMirror[11:19] = theta[3:11]
            dqMirror[0] *= -1
            dqMirror[2] *= -1
            dqMirror[3:11] = self.robot_skeleton.dq[11:19]
            dqMirror[11:19] = self.robot_skeleton.dq[3:11]
            targetMirror = np.array(self.target)
            targetMirror[0] *= -1
            return np.concatenate([np.cos(thetaMirror), np.sin(thetaMirror), dqMirror, vec, targetMirror, f]).ravel()
        #print("target info: " + str(vec) + " " + str(self.target))
        #print("force info: " + str(f))
        obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq, vec, self.target, f]).ravel()
        #print("observing: " + str(obs))
        return obs
        #return np.concatenate([theta, self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        '''reset_model'''
        print("reset " + str(self.reset_number))
        if self.collectFrictionTestData:
            if self.reset_number > 0:
                self.maxDeforms.append(self.maxDeform)
                self.lowestDists.append(self.lowestDist)
                self.runtimes.append(self.numSteps)
        self.lowestDist = 9999.
        self.maxDeform = 0
        self.inflaterT = 0
        self.numSteps = 0
        #self.clothScene.translateCloth(0, np.array([0,3.1,0]))
        self.dart_world.reset()
        self.clothScene.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)

        #1st sleeve start interpolation:

        qpos[0] -= 0
        qpos[1] -= 0.
        qpos[2] += 0
        qpos[3] += 0.
        qpos[4] -= 0.
        #qpos[5] += 1
        qpos[5] += 0.75
        qpos[6] += 0.25
        #qpos[7] += 0.0
        qpos[7] += 2.0
        qpos[8] += 2.9
        qpos[9] += 0.0
        #qpos[9] += 0.6
        qpos[10] += 0.6
        #qpos[10] += 0.0

        if self.collectFrictionTestData and self.reset_number > 0:
            if (self.reset_number)%10 == 0:
                print("Setting friction to " + str(self.clothScene.getFriction()+0.05))
                self.clothScene.setFriction(0, self.clothScene.getFriction()+0.05)
            pyutils.saveList(list=self.maxDeforms, filename=self.maxDeformFile)
            pyutils.saveList(list=self.lowestDists, filename=self.lowestDistFile)
            pyutils.saveList(list=self.runtimes, filename=self.runtimesFile)

        '''
        self.interpolationStart = np.array(qpos)
        self.interpolationGoal = np.array(qpos)
        self.interpolationGoal[5] = 0.75
        self.interpolationGoal[6] = 0.25
        self.interpolationGoal[7] = 2.0
        self.interpolationGoal[8] = 2.9
        self.interpolationGoal[9] = 0.0
        self.interpolationGoal[10] += 0.6
        '''

        #1st sleeve end -> 2nd sleeve start interpolation
        '''
        qpos[0] += 0.415
        qpos[1] += 0.192
        qpos[2] += 0.6
        qpos[3] -= 0.244
        qpos[4] += 0.288
        qpos[5] -= 0.464
        qpos[6] += 0.968
        qpos[7] += 2.101
        qpos[8] += 0
        qpos[9] -= 0.6
        qpos[10] += 0.573
        qpos[11] += 0.1
        qpos[12] += 0.251
        #qpos[13] += 0.149
        #qpos[14] += 0.19
        #qpos[15] += 0.0
        #qpos[16] += 0.3
        qpos[13] = 0.75
        qpos[14] = 0.2
        qpos[15] += 2.0
        qpos[16] += 2.9
        qpos[17] += 0.6
        qpos[18] += 0.6
        qpos[19] -= 0.192
        qpos[20] += 0.259
        qpos[21] += 0


        self.interpolationStart = np.array(qpos)
        self.interpolationGoal = np.zeros(len(qpos))
        #self.interpolationGoal = np.array(qpos)
        self.interpolationGoal[5] = -0.5
        self.interpolationGoal[13] = 0.75
        self.interpolationGoal[14] = 0.2
        self.interpolationGoal[15] = 2.0
        self.interpolationGoal[16] = 2.9
        self.interpolationGoal[17] = 0.6
        self.interpolationGoal[18] = 0.6
        #self.interpolationGoal[17] = 0.0
        '''

        # 2nd sleeve start
        '''
        qpos[5] = -0.5
        qpos[13] = 0.75
        qpos[14] = 0.2
        qpos[15] = 2.0
        qpos[16] = 2.9
        qpos[17] = 0.6
        qpos[18] = 0.6
        '''

        #uper body 1 arm fail #1 settings
        '''qpos[7] += 0.25
        qpos[8] += 2.0
        qpos[9] += 0.0
        qpos[10] += -0.6'''

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.025, high=.025, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        
        #reset cloth tube orientation and rotate sphere position
        v1 = np.array([0, 1., 0])
        v2 = np.array([0., 0, -1.])
        self.random_dir = np.array([-1., 0, 0])
        if self.arm == 2 or self.mirror is True:
            self.random_dir = np.array([1., 0, 0])
        if self.simulateCloth is True:   
            if self.rotateCloth is True:
                while True:
                    v2 = self.clothScene.sampleDirections()[0]
                    if np.dot(v2/np.linalg.norm(v2), np.array([0, -1, 0.])) < 1:
                        break
            M = self.clothScene.rotateTo(v1,v2)
            #self.clothScene.rotateCloth(0, M)
            #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=0.25, axis=np.array([0,1.,0.])))
            #self.clothScene.translateCloth(0, np.array([-0.042,-0.7,-0.035]))
        #self.clothScene.translateCloth(0, np.array([-0.75,0,0])) #shirt next to person
        #self.clothScene.translateCloth(0, np.array([-0., 1.0, 0]))  # shirt above to person
        #self.clothScene.translateCloth(0, np.array([-0., 0, -0.75]))  # shirt in front of person
        #self.clothScene.translateCloth(0, np.array([0,3.1,0]))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))
        #self.clothScene.rotateCloth(0, M)
        
        #load cloth state from ~/Documents/dev/objFile.obj
        #self.clothScene.loadObjState()
        
        #move cloth out of arm range
        #self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
        
        #old sampling in box
        #'''
        reacher_range = 1.1
        if not self.sampleFromHemisphere:
            while True:
                self.target = self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
                #print('target = ' + str(self.target))
                if np.linalg.norm(self.target) < reacher_range: break
        #'''
        
        #sample target from hemisphere
        if self.sampleFromHemisphere is True:
            self.target = self.hemisphereSample(maxradius=reacher_range, minradius=0.9, norm=self.random_dir)

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        #increment friction
        #if self.reset_number%4 == 0 and self.reset_number > 0:
        #    self.clothScene.setFriction(f=self.clothScene.getFriction()+0.1)

        self.reset_number += 1

        #self.handle_node.reset()
        if self.handleNode is not None:
            #self.handle_node.setTransform(self.robot_skeleton.bodynodes[8].T)
            self.handleNode.recomputeOffsets()

        if self.gripper is not None:
            self.gripper.setTransform(self.robot_skeleton.bodynodes[8].T)
        
        obs = self._get_obs()
        
        #self.render()
        #if np.linalg.norm(obs[-39:]) > 0.00001:
        #    print("COLLISION")
        #    self.reset_model()

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a=0
        #collision spheres creation
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

        if self.inflatorActive:
            p = self.inflatorP2
            r = self.inflatorR2
            if self.inflaterT < self.inflatorDuration:
                p = LERP(self.inflatorP1, self.inflatorP2, self.inflaterT/self.inflatorDuration)
                r = LERP(self.inflatorR1, self.inflatorR2, self.inflaterT/self.inflatorDuration)
            csVarsInflator = np.array([r, -1, -1, 0, 0, 0])
            collisionSpheresInfo = np.concatenate(
                [cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7,
                 csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13,
                 csVars13, p, csVarsInflator]).ravel()

        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((14,14))
            if self.inflatorActive:
                collisionCapsuleInfo = np.zeros((15, 15))
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
        
    def hemisphereSample(self, maxradius=1, minradius = 0, norm=np.array([0,0,1.]), frustrum = 0.7):
        p = norm
        while True:
            p = self.np_random.uniform(low=-maxradius, high=maxradius, size=3)
            p_n = np.linalg.norm(p)
            if p_n <= maxradius and p_n >= minradius:
                if(np.dot(p/p_n, norm) > frustrum):
                    return p

        
    def extraRenderFunction(self):
        #print("extra render function")
        
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        '''GL.glBegin(GL.GL_POLYGON)
        GL.glVertex3d(-0.53636903, 0.0341332, -0.0731871)
        GL.glVertex3d(-0.54120499, 0.038279, -0.0522403)
        GL.glVertex3d(-0.53161001, 0.0271241, -0.0933676)
        GL.glVertex3d(-0.50181198, -0.138647,   -0.0973828)
        GL.glVertex3d(-0.49843001, -0.167151,   -0.0937369)
        #GL.glVertex3d(-0.47647399, -0.174687, -0.0643502)
        GL.glVertex3d(-0.50627899, -0.175074, -0.0748381)
        GL.glVertex3d(-0.51507902, -0.14102399, -0.0483567)
        GL.glVertex3d(-0.52044398, -0.0982356, -0.0367543)
        GL.glVertex3d(-0.52189898, -0.0590227, -0.0141631)
        GL.glVertex3d(-0.527426, -0.0366101, -0.00950058)
        GL.glVertex3d(-0.53686303, 0.00799755, -0.0190606)
        GL.glVertex3d(-0.54120499, 0.038279, -0.0522403)
        GL.glEnd()'''

        #render the vertrex handle_node(s)/Handle(s)
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

        if self.renderInflator and self.inflatorActive:
            p = self.inflatorP2
            r = self.inflatorR2
            if self.inflaterT < self.inflatorDuration:
                p=LERP(self.inflatorP1, self.inflatorP2, self.inflaterT/self.inflatorDuration)
                r=LERP(self.inflatorR1, self.inflatorR2, self.inflaterT/self.inflatorDuration)
            GL.glPushMatrix()
            GL.glTranslated(p[0], p[1], p[2])
            GLUT.glutWireSphere(r, 10, 10)
            GL.glPopMatrix()
        
        if self.renderSuccess is True:
            for i in range(len(self.targetHistory)):
                p = self.targetHistory[i]
                s = self.successHistory[i]
                GL.glColor3d(1,0.,0)
                if s is True:
                    GL.glColor3d(0,1.,0)
                GL.glPushMatrix()
                GL.glTranslated(p[0], p[1], p[2])
                GLUT.glutSolidSphere(0.01, 10,10)
                GL.glPopMatrix()
        
        #draw hemisphere samples for target sampling
        '''
        GL.glColor3d(0,1,0)
        for i in range(1000):
            normVec = self.random_dir
            p = self.hemisphereSample(maxradius=0.95, minradius=0.7, norm=normVec)

            #p=np.array([0,0,0.])
            #while True:
            #    p = self.np_random.uniform(low=-1.5, high=1.5, size=3)
            #    if np.linalg.norm(p) < 1.5: break
            GL.glPushMatrix()
            GL.glTranslated(p[0], p[1], p[2])
            GLUT.glutSolidSphere(0.01, 10,10)
            GL.glPopMatrix()
        '''
        
        #print("ID:" + str(self.clothScene.id))
            
        m_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        
        textX = 15.
        if self.renderForceText:
            HSF = self.clothScene.getHapticSensorObs()
            #print("HSF: " + str(HSF))
            for i in range(self.clothScene.getNumHapticSensors()):
                #print("i = " + str(i))
                #print("HSL[i] = " + str(HSL[i*3:i*3+3]))
                #print("HSF[i] = " + str(HSF[i*3:i*3+3]))
                self.clothScene.drawText(x=textX, y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
            textX += 160
        
        #draw 2d HUD setup
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, m_viewport[2], 0, m_viewport[3], -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_CULL_FACE);
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT);
        
        #draw the load bars
        if self.renderDofs:
            #draw the load bar outlines
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor3d(0,0,0)
            GL.glBegin(GL.GL_QUADS)
            for i in range(len(self.robot_skeleton.q)):
                y = 58+18.*i
                x0 = 120+70
                x1 = 210+70
                GL.glVertex2d(x0, y)
                GL.glVertex2d(x0, y+15)
                GL.glVertex2d(x1, y+15)
                GL.glVertex2d(x1, y)
            GL.glEnd()
            #draw the load bar fills
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            for i in range(len(self.robot_skeleton.q)):
                qlim = self.limits(i)
                qfill = (self.robot_skeleton.q[i]-qlim[0])/(qlim[1]-qlim[0])
                y = 58+18.*i
                x0 = 121+70
                x1 = 209+70
                x = LERP(x0,x1,qfill)
                xz = LERP(x0,x1,(-qlim[0])/(qlim[1]-qlim[0]))
                GL.glColor3d(0,2,3)
                GL.glBegin(GL.GL_QUADS)
                GL.glVertex2d(x0, y+1)
                GL.glVertex2d(x0, y+14)
                GL.glVertex2d(x, y+14)
                GL.glVertex2d(x, y+1)
                GL.glEnd()
                GL.glColor3d(2,0,0)
                GL.glBegin(GL.GL_QUADS)
                GL.glVertex2d(xz-1, y+1)
                GL.glVertex2d(xz-1, y+14)
                GL.glVertex2d(xz+1, y+14)
                GL.glVertex2d(xz+1, y+1)
                GL.glEnd()
                GL.glColor3d(0,0,2)
                GL.glBegin(GL.GL_QUADS)
                GL.glVertex2d(x-1, y+1)
                GL.glVertex2d(x-1, y+14)
                GL.glVertex2d(x+1, y+14)
                GL.glVertex2d(x+1, y+1)
                GL.glEnd()
                GL.glColor3d(0,0,0)
                
                textPrefix = "||q[" + str(i) + "]|| = "
                if i < 10:
                    textPrefix = "||q[0" + str(i) + "]|| = "
                    
                self.clothScene.drawText(x=30, y=60.+18*i, text=textPrefix + '%.2f' % qlim[0], color=(0.,0,0))
                self.clothScene.drawText(x=x0, y=60.+18*i, text='%.3f' % self.robot_skeleton.q[i], color=(0.,0,0))
                self.clothScene.drawText(x=x1+2, y=60.+18*i, text='%.2f' % qlim[1], color=(0.,0,0))

        self.clothScene.drawText(x=15 , y=600., text='Friction: %.2f' % self.clothScene.getFriction(), color=(0., 0, 0))
        f = self.clothScene.getHapticSensorObs()
        maxf_mag = 0

        for i in range(int(len(f)/3)):
            fi = f[i*3:i*3+3]
            #print(fi)
            mag = np.linalg.norm(fi)
            #print(mag)
            if mag > maxf_mag:
                maxf_mag = mag
        #exit()
        #self.clothScene.drawText(x=15, y=620., text='Max force (1 dim): %.2f' % np.amax(f), color=(0., 0, 0))
        #self.clothScene.drawText(x=15, y=640., text='Max force (3 dim): %.2f' % maxf_mag, color=(0., 0, 0))

        self.clothScene.drawText(x=15, y=620., text='Best Target Distance: %.2f' % self.lowestDist, color=(0., 0, 0))
        self.clothScene.drawText(x=15, y=640., text='Max Deformation: %.2f' % self.maxDeform, color=(0., 0, 0))


        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        a=0

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
            self.track_skeleton_id = 0
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t
