import numpy as np
from gym import utils, error
from gym.envs.dart import dart_env

from gym.envs.dart.static_sawyer_window import *
from gym.envs.dart.norender_window import *

from gym.envs.dart.parameter_managers import *

try:
    import pydart2 as pydart
    import pydart2.joint as Joint
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))

import pybullet as p
import time
import pybullet_data
import os
import random

try:
    import pyPhysX as pyphysx
    pyphysx.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pyphysx.)".format(e))

import pyPhysX.pyutils as pyutils
import pyPhysX.renderUtils as renderUtils

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
    def __init__(self, env, target=None, timestep=0.01):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = timestep
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs-6
        self.qhat = self.skel.q
        self.Kp = np.diagflat([30000.0] * (ndofs))
        self.Kd = np.diagflat([300.0] * (ndofs))

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
        self.target = np.array(self.skel.q[6:])
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

    def query(self, obs):
        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q[6:] + skel.dq[6:] * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq[6:])
        b = -skel.c[6:] + p + d + skel.constraint_forces()[6:]
        A = skel.M[6:, 6:] + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        #tau = p - self.Kd.dot(skel.dq[6:] + x * self.h)
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class DartIiwaSPDBezierEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        rendering = True
        self.renderIKShadow = False #if true, render the IK solution skel also
        self.renderViewWindow = True #if true, square window with special orientation
        self.control_bounds = np.array([np.ones(13), -1*np.ones(13)])
        #self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        obs_dim = 14
        self.param_manager = hopperContactMassManager(self)
        self.kinematicIK = True #if false, dynamic SPD
        self.freezeTracking = False #if true, arm is stationary
        self.previousIKResult = np.zeros(7)
        self.viewer = None

        #SPD error graphing per dof
        self.graphSPDError = False
        self.SPDErrorGraph = None
        if self.graphSPDError:
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)

        # setup pybullet for IK
        print("Setting up pybullet")
        self.pyBulletPhysicsClient = p.connect(p.DIRECT)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # print(dir_path)
        self.pyBulletIiwa = p.loadURDF(dir_path + '/assets/iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf', flags=p.URDF_USE_SELF_COLLISION)
        print("Iiwa bodyID: " + str(self.pyBulletIiwa))
        self.ikPath = pyutils.CubicBezier()
        self.ikPathTimeScale = 0.017 #relationship between number of steps and spline time
        self.ikTarget = np.array([0.5, 0, 0])
        self.ikOrientationFrame = pyutils.ShapeFrame() #reset in reset, go figure...
        self.rigidClothTargetFrame = pyutils.BoxFrame(c0=np.array([0.1, 0.2, 0.001]), c1=np.array([-0.1, 0, -0.001]))
        self.orientationEndPoints = [pyutils.ShapeFrame(), pyutils.ShapeFrame()]
        self.orientationTarget = pyutils.ShapeFrame()  # only used for orientation
        self.maxIiwaReach = 1.0
        self.robotPathParams = {'p0_depth_range': 0.05, 'p0_depth_offset': -0.15, 'p0_disk_rad': self.maxIiwaReach * 0.7,
                                'p3_el_dim': np.array([0.2, 0.1, 0.1]), 'p3_el_org': np.array([0.15, -0.025, -0.8]),
                                'b_tan_dot_cone': 0.2, 'b_tan_len': 0.5,
                                'orient_dot_cone': 0.8}
        print("Number of joint: " + str(p.getNumJoints(self.pyBulletIiwa)))
        for i in range(p.getNumJoints(self.pyBulletIiwa)):
            jinfo = p.getJointInfo(self.pyBulletIiwa, i)
            print(" " + str(jinfo[0]) + " " + str(jinfo[1]) + " " + str(jinfo[2]) + " " + str(jinfo[3]) + " " + str(jinfo[12]))

        self.numSteps = 0

        #initialize the variable
        self.SPDController = None

        self.iiwa_dof_llim = np.zeros(7)
        self.iiwa_dof_ulim = np.zeros(7)
        self.iiwa_dof_jr = np.zeros(7)
        self.iiwa_ranges_initialized = False
        self.iiwa_root_dofs = np.array([-1.2, -1.2, -1.2, 0, 0, 0]) #values for the fixed 6 dof root transformation
        self.iiwa_rest = np.array([0, 0, 0, 0, 0, 0, 0])

        #capability mapping variables:
        self.capable_range = 1.3 #radius of the sphere/cube for sampling
        self.orientationCapability  = True #if true, test orientations at each point
        self.clothTaskOrientations  = False #TODO:if true (and orientationCapability), then keep end effector level
        self.liveCapabilitySample   = False #if true, try one sample per step and render cycle
        self.saveCapabilitySample   = False #if true, saving the error results
        self.loadCapabilityMap      = False #if true, load a capabaility map immediately
        self.numRandInitialPoses    = 0 #if 0, don't use. Otherwise, re-try the IK from this many random poses
        self.maxIKIterations = 4 #number of iterations for the IK solver
        self.sample_density = 25 #how many samples in each dimension within the cube
        self.sampleRenderRange = [[0, 0, 0], [self.sample_density-1,self.sample_density-1,self.sample_density-1]]
        self.capabilitySamples = []
        self.capabilityStats = {"best":9999, "worst":0, "total":0, "avg":0, "numSamples":0} #this initialization is repeated in "initializeCapabilitySamples"
        bottomCorner = np.array([-self.capable_range, -self.capable_range, -self.capable_range])
        self.numCapabilitySamples = 0
        self.numCapabilityEntries = 0
        self.capabilityTarget = [0,0,0]
        for ix in range(self.sample_density):
            self.capabilitySamples.append([])
            for iy in range(self.sample_density):
                self.capabilitySamples[-1].append([])
                for iz in range(self.sample_density):
                    dx = self.capable_range*2.0*(ix/(self.sample_density-1))
                    dy = self.capable_range*2.0*(iy/(self.sample_density-1))
                    dz = self.capable_range*2.0*(iz/(self.sample_density-1))
                    point = bottomCorner + np.array([dx, dy, dz])
                    if(np.linalg.norm(point) <= self.capable_range):
                        self.capabilitySamples[-1][-1].append([point])
                        self.numCapabilitySamples += 1
                    else: #if the sample point is not in the sphere, don't bother with it
                        self.capabilitySamples[-1][-1].append([None])
                    self.numCapabilityEntries += 1

        if self.loadCapabilityMap:
            print("loading capability data")
            self.loadCapabilityData()
        print("capability sample points initialized: " + str(self.numCapabilitySamples) + "/" + str(self.numCapabilityEntries))

        #dart_env.DartEnv.__init__(self, 'iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf', 5, obs_dim, self.control_bounds, disableViewer=(not rendering))
        if self.renderViewWindow:
            dart_env.DartEnv.__init__(self, 'iiwa_description/urdf/iiwa7_simplified.urdf', 5, obs_dim, self.control_bounds, disableViewer=(not rendering), screen_width=1000, screen_height=1000)
        else:
            dart_env.DartEnv.__init__(self, 'iiwa_description/urdf/iiwa7_simplified_collision_complete.urdf', 5, obs_dim, self.control_bounds, disableViewer=(not rendering))

        #initialize the controller
        self.SPDController = SPDController(self)

        #self.dart_world.set_collision_detector(3) # 3 is ode collision detector

        utils.EzPickle.__init__(self)

        # enable DART collision testing (BROKEN)
        #self.robot_skeleton.set_self_collision_check(True)
        #self.robot_skeleton.set_adjacent_body_check(False)

        # lock the first 6 dofs
        for i in range(6):
            self.robot_skeleton.dof(i).set_position_lower_limit(0)
            self.robot_skeleton.dof(i).set_position_upper_limit(0)
        #self.robot_skeleton.dof(3).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(3).set_position_upper_limit(0.01)
        #self.robot_skeleton.dof(4).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(4).set_position_upper_limit(0.01)
        #self.robot_skeleton.dof(5).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(5).set_position_upper_limit(0.01)

        print("BodyNodes: ")
        for bodynode in self.robot_skeleton.bodynodes:
            print("     : " + bodynode.name)

        print("Joints: ")
        for joint in self.robot_skeleton.joints:
            print("     : " + joint.name)
            joint.set_position_limit_enforced()

        print("Dofs: ")
        for dof in self.robot_skeleton.dofs:
            print("     : " + dof.name)
            #print("         damping: " + str(dof.damping_coefficient()))
            dof.set_damping_coefficient(2.0)
        self.robot_skeleton.joints[0].set_actuator_type(Joint.Joint.LOCKED)
        #self.robot_skeleton.set_self_collision_check(True)

        # compute the joint ranges for null space IK
        for i in range(7):
            # print(i)
            self.iiwa_dof_llim[i] = self.robot_skeleton.dofs[i + 6].position_lower_limit()
            self.iiwa_dof_ulim[i] = self.robot_skeleton.dofs[i + 6].position_upper_limit()
            self.iiwa_dof_jr[i] = self.iiwa_dof_ulim[i] - self.iiwa_dof_llim[i]
        self.iiwa_ranges_initialized = True
        self.capabilityTarget = [0, 0, 0]
        print("Done initializing")

    def _step(self, a):

        #print("-----------------")
        #print(self.robot_skeleton.q)
        #print("a: " + str(a))

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau = clamped_control * self.action_scale
        #self.do_simulation(tau, self.frame_skip)

        #test IK
        randDir = np.random.random(3)
        randDir *= 2.0
        randDir -= np.ones(3)
        while(np.linalg.norm(randDir) > 1.0):
            randDir = np.random.random(3)
            randDir *= 2.0
            randDir -= np.ones(3)
        #self.ikTarget += randDir*0.025
        #print(self.numSteps)
        self.ikTarget = self.ikPath.pos(self.numSteps*self.ikPathTimeScale)
        #self.ikTarget = np.array(self.ikPath.points[-1].p)

        #compute capability sample under the current rules
        if self.liveCapabilitySample:
            self.ikTarget = None
            while self.ikTarget is None:
                self.capabilityTarget[2] += 1
                if self.capabilityTarget[2] >= (self.sample_density):
                    self.capabilityTarget[2] = 0
                    self.capabilityTarget[1] += 1
                if self.capabilityTarget[1] >= (self.sample_density):
                    self.capabilityTarget[1] = 0
                    self.capabilityTarget[0] += 1
                    print("wrapping y: " + str(self.capabilityTarget[0])+"/"+str(self.sample_density - 1))
                if self.capabilityTarget[0] >= (self.sample_density):
                    self.capabilityTarget[0] = 0
                    #if wrapping in [0], then save the map
                    self.saveCapabilityData()
                    print("Capability data collected and saved. Exiting.")
                    exit()
                self.ikTarget = self.capabilitySamples[self.capabilityTarget[0]][self.capabilityTarget[1]][self.capabilityTarget[2]][0]


        result = None
        if self.iiwa_ranges_initialized:
            self.rigidClothTargetFrame.setQuaternion(
                pyutils.qSLERP(q0=self.orientationEndPoints[0].quat, q1=self.orientationEndPoints[1].quat,
                               t=min(1.0, self.numSteps * self.ikPathTimeScale)))
            self.rigidClothTargetFrame.setOrg(org=self.ikTarget)

            tar_quat = self.rigidClothTargetFrame.quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            tar_dir = -self.ikTarget / np.linalg.norm(self.ikTarget)

            #tar_quat = self.ikOrientationFrame.quat
            #tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            if self.orientationCapability:
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                             endEffectorLinkIndex=8,
                                             targetPosition=self.ikTarget,
                                             targetOrientation=tar_quat,
                                             lowerLimits=self.iiwa_dof_llim.tolist(),
                                             upperLimits=self.iiwa_dof_ulim.tolist(),
                                             jointRanges=self.iiwa_dof_jr.tolist(),
                                             restPoses=self.robot_skeleton.q[6:].tolist(),
                                             maxNumIterations = self.maxIKIterations
                                             )
            else:
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                      endEffectorLinkIndex=8,
                                                      targetPosition=self.ikTarget,
                                                      # targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.iiwa_dof_llim.tolist(),
                                                      upperLimits=self.iiwa_dof_ulim.tolist(),
                                                      jointRanges=self.iiwa_dof_jr.tolist(),
                                                      restPoses=self.robot_skeleton.q[6:].tolist(),
                                                      maxNumIterations=self.maxIKIterations
                                                      )
        else:
            result = p.calculateInverseKinematics(self.pyBulletIiwa, 8, self.ikTarget)

        #collect a number of re-tries at IK from random poses
        rand_results = []
        if self.liveCapabilitySample:
            for i in range(self.numRandInitialPoses):
                #compute and set random pose
                rand_pose = np.random.uniform(low=self.iiwa_dof_llim, high=self.iiwa_dof_ulim)
                #print("random pose: " + str(rand_pose))
                self.setPosePyBullet(rand_pose)
                if self.iiwa_ranges_initialized:
                    tar_quat = self.ikOrientationFrame.quat
                    tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
                    if self.orientationCapability:
                        rand_results.append(p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                              endEffectorLinkIndex=8,
                                                              targetPosition=self.ikTarget,
                                                              targetOrientation=tar_quat,
                                                              lowerLimits=self.iiwa_dof_llim.tolist(),
                                                              upperLimits=self.iiwa_dof_ulim.tolist(),
                                                              jointRanges=self.iiwa_dof_jr.tolist(),
                                                              restPoses=rand_pose.tolist(),
                                                              maxNumIterations=self.maxIKIterations
                                                              ))
                    else:
                        rand_results.append(p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                              endEffectorLinkIndex=8,
                                                              targetPosition=self.ikTarget,
                                                              # targetOrientation=tar_quat,
                                                              # targetOrientation=tar_dir,
                                                              lowerLimits=self.iiwa_dof_llim.tolist(),
                                                              upperLimits=self.iiwa_dof_ulim.tolist(),
                                                              jointRanges=self.iiwa_dof_jr.tolist(),
                                                              restPoses=rand_pose.tolist(),
                                                              maxNumIterations=self.maxIKIterations
                                                              ))
                else:
                    rand_results.append(p.calculateInverseKinematics(self.pyBulletIiwa, 8, self.ikTarget))

        self.previousIKResult = np.array(result)
        self.setPosePyBullet(result)
        #print(p.getLinkState(self.pyBulletSawyer, 12, computeForwardKinematics=True)[0])
        #result = p.calculateInverseKinematics(self.pyBulletSawyer, 12, self.ikTarget, lowerLimits=lowerLimits, upperLimits=upperLimits, jointRanges=jointRanges, restPoses=restPoses)

        clamped_result = np.array(result)
        if self.freezeTracking:
            fixed_pose = np.array([-1.2, -1.2, -1.2, 0., 0., 0., -0.57619293, 1.59480151, 0.15745552, -0.14593577, -0.16227263, 1.36463517, 1.02508058])
            fixed_pose = np.array([-1.2, -1.2, -1.2, 0., 0., 0., -2.45673167, 1.69136021, 0.19637591, -0.00873531, -0.18845419, 1.43566458, -0.87881501])
            fixed_pose = np.concatenate([self.iiwa_root_dofs, self.iiwa_rest])
            self.robot_skeleton.set_positions(fixed_pose)
            a=0 #nothing happens
        elif self.kinematicIK:
            for dof,val in enumerate(result):
                if(val < self.iiwa_dof_llim[dof]):
                    clamped_result[dof] = self.iiwa_dof_llim[dof]
                elif(val > self.iiwa_dof_ulim[dof]):
                    clamped_result[dof] = self.iiwa_dof_ulim[dof]


            #kinematic
            self.robot_skeleton.set_positions(np.concatenate([self.iiwa_root_dofs, clamped_result]))
        else:
            #SPD (dynamic)
            if self.SPDController is not None:
                self.SPDController.target = result
                tau = np.concatenate([np.zeros(6), self.SPDController.query(obs=None)])

                # check the arm for joint, velocity and torque limits
                #tau = self.iiwa_skel.forces()
                if False:
                    tau_upper_lim = self.robot_skeleton.force_upper_limits()
                    tau_lower_lim = self.robot_skeleton.force_lower_limits()
                    vel = self.robot_skeleton.velocities()
                    pos = self.robot_skeleton.positions()
                    pos_upper_lim = self.robot_skeleton.position_upper_limits()
                    pos_lower_lim = self.robot_skeleton.position_lower_limits()
                    for i in range(len(tau)):
                        if (tau[i] > tau_upper_lim[i]):
                            tau[i] = tau_upper_lim[i]
                        if (tau[i] < tau_lower_lim[i]):
                            tau[i] = tau_lower_lim[i]

                self.do_simulation(tau, self.frame_skip)
        #print(self.robot_skeleton.q)

        #track ik error in capability map
        clamped_rand_results = []
        if self.saveCapabilitySample and self.iiwa_ranges_initialized:
            best_error = 9999999
            best_solution = 0
            for iter in range(len(rand_results)+1):
                if(iter > 0):
                    #self.setPosePyBullet(rand_results[iter-1])
                    clamped_rand_result = np.array(rand_results[iter-1])
                    for dof, val in enumerate(rand_results[iter-1]):
                        if (val < self.iiwa_dof_llim[dof]):
                            clamped_rand_result[dof] = self.iiwa_dof_llim[dof]
                        elif (val > self.iiwa_dof_ulim[dof]):
                            clamped_rand_result[dof] = self.iiwa_dof_ulim[dof]
                    clamped_rand_results.append(clamped_rand_result)
                    self.robot_skeleton.set_positions(np.concatenate([self.iiwa_root_dofs, clamped_rand_result]))

                #ik_error = np.linalg.norm(p.getLinkState(self.pyBulletIiwa, 8)[0] - self.ikTarget) #sometimes outside the joint limits...
                ik_error = np.linalg.norm(self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)) - self.ikTarget)
                if(ik_error < best_error):
                    best_solution = iter
                    best_error = ik_error

            #set robot pose to the best solution
            if best_solution == 0:
                self.previousIKResult = np.array(clamped_result)
                self.setPosePyBullet(clamped_result)
                self.robot_skeleton.set_positions(np.concatenate([self.iiwa_root_dofs, clamped_result]))
            else:
                self.previousIKResult = np.array(clamped_rand_results[best_solution-1])
                self.setPosePyBullet(clamped_rand_results[best_solution-1])
                self.robot_skeleton.set_positions(np.concatenate([self.iiwa_root_dofs, clamped_rand_results[best_solution-1]]))


            self.capabilitySamples[self.capabilityTarget[0]][self.capabilityTarget[1]][self.capabilityTarget[2]].append(best_error)
            self.capabilityStats["best"] = min(self.capabilityStats["best"], best_error)
            self.capabilityStats["worst"] = max(self.capabilityStats["worst"], best_error)
            self.capabilityStats["numSamples"] += 1
            self.capabilityStats["total"] += best_error
            self.capabilityStats["avg"] = self.capabilityStats["total"]/self.capabilityStats["numSamples"]

        pose_error = self.robot_skeleton.q[6:]-result
        pose_error_mag = np.linalg.norm(self.robot_skeleton.q[6:]-result)

        if self.graphSPDError:
            self.SPDErrorGraph.addToLinePlot(data=pose_error.tolist())

        #print("Errors: IK=" + str(ik_error) + ", SPD=" + str(pose_error_mag))


        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()


        alive_bonus = 1.0
        reward = 0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = False
        ob = self._get_obs()

        self.numSteps += 1

        return ob, reward, done, {'model_parameters':self.param_manager.get_simulator_parameters(), 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[6:],
            self.robot_skeleton.dq[6:]
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.numSteps = 0
        self.dart_world.reset()
        iiwa_pose = np.array(self.robot_skeleton.q)
        iiwa_pose[:6] = np.array(self.iiwa_root_dofs)
        iiwa_pose[6:] = np.array(self.iiwa_rest)
        self.robot_skeleton.set_positions(iiwa_pose)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        # self.hoop.set_positions(np.array([0,0,0,0,2.0,0, 0 ,0 ,0])) #get the hoop out of the way
        T = self.robot_skeleton.bodynodes[0].world_transform()
        tempFrame = pyutils.ShapeFrame()
        tempFrame.setTransform(T)
        root_quat = tempFrame.quat
        root_quat = (root_quat.x, root_quat.y, root_quat.z, root_quat.w)
        p.resetBasePositionAndOrientation(self.pyBulletIiwa, posObj=np.zeros(3), ornObj=root_quat)
        self.setPosePyBullet(self.robot_skeleton.q[6:])

        #test IK reset
        self.ikPath = pyutils.Spline()
        dart_ef = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
        #self.ikPath.addPoint(0, 0.0)
        #self.ikPath.addPoint(0.5, 0.5)
        #self.ikPath.addPoint(1.0, 0.5)
        #self.ikPath.points[0].p = np.array(dart_ef)
        #print("checking IK spline...")

        #setup the orientation frame
        self.ikOrientationFrame.setFromDirectionandUp(dir=np.array([0, -1.0, 0]),
                                                           up=np.array([0, 0, 1.0]),
                                                           org=np.zeros(3))

        # ikPath: setup the Bezier curve with start and end point distribution and "in-facing" tangents
        # p0 sample from planar disk 90% size of iiwa reach for xy,
        # then sample from depth and move to iiwa location
        depthRange = self.robotPathParams['p0_depth_range']
        diskRad = self.robotPathParams['p0_disk_rad']
        # rejection sample for cylinder
        diskPoint = np.array([(random.random() * 2 - 1) * diskRad, (random.random() * 2 - 1) * diskRad])
        while (np.linalg.norm(diskPoint) > diskRad):
            diskPoint = np.array([(random.random() * 2 - 1) * diskRad, (random.random() * 2 - 1) * diskRad])
        depth = random.random() * depthRange + self.robotPathParams['p0_depth_offset']
        p0 = self.robot_skeleton.bodynodes[3].to_world(np.zeros(3)) + np.array([diskPoint[0], diskPoint[1], depth])

        # p3 ellipsoid sampling about the shoulder region
        p3_distribution = pyutils.EllipsoidFrame(dim=self.robotPathParams['p3_el_dim'],
                                                 org=self.robotPathParams['p3_el_org'])
        p3 = p3_distribution.sample()[0]

        # setup tangent vectors constrained to conical region between chosen end points
        dot_constraint = self.robotPathParams['b_tan_dot_cone']
        tan_length = self.robotPathParams['b_tan_len']
        v03 = p3 - p0
        v03n = v03 / np.linalg.norm(v03)
        v1 = pyutils.sampleDirections()[0]
        while (v1.dot(v03n) < dot_constraint):
            v1 = pyutils.sampleDirections()[0]
        v2 = pyutils.sampleDirections()[0]
        while (v2.dot(-v03n) < dot_constraint):
            v2 = pyutils.sampleDirections()[0]

        p1 = p0 + v1 * tan_length
        p2 = p3 + v2 * tan_length
        self.ikPath = pyutils.CubicBezier(p0, p1, p2, p3)

        # compute the orientation targets by picking a start and end quaternion and slerping over the path
        # direction 1 should point toward the character
        # direction 2 should point in the tangent of the curve at the end
        # both distribution means should be projected into the
        quat_dot_constraint = self.robotPathParams['orient_dot_cone']
        d1 = pyutils.sampleDirections()[0]
        while (d1.dot(v03n) < quat_dot_constraint):  # in the direction of the endpoint from startpoint
            d1 = pyutils.sampleDirections()[0]
        d2 = pyutils.sampleDirections()[0]
        while (d2.dot(-v2) < quat_dot_constraint):  # in the tangent direction at the end of the curve
            d2 = pyutils.sampleDirections()[0]
        # now compute the quaternions from these directions
        '''
        self.orientationEndPoints[0].setFromDirectionandUp(dir=d1,
                                                           up=np.array([0, -1.0, 0]),
                                                           org=p0)
        self.orientationEndPoints[1].setFromDirectionandUp(dir=d2,
                                                           up=np.array([0, -1.0, 0]),
                                                           org=p3)
        '''

        self.orientationEndPoints[0].setFromDirectionandUp(dir=np.array([0, -1.0, 0]),
                                                           up=d1,
                                                           org=p0)
        self.orientationEndPoints[1].setFromDirectionandUp(dir=np.array([0, -1.0, 0]),
                                                           up=d2,
                                                           org=p3)
        self.rigidClothTargetFrame.setQuaternion(self.orientationEndPoints[0].quat)  # set initial target

        # self.checkIKSplineValidity()

        # initial IK target is the first spline point
        # self.ikTarget = self.ikPath.points[0].p
        self.ikTarget = self.ikPath.pos(t=0.0)

        # self.rigidClothTargetFrame.setFromDirectionandUp(dir=-self.ikTarget, up=np.array([0, -1.0, 0]),
        #                                                 org=self.ikTarget)
        tar_quat = self.rigidClothTargetFrame.quat
        tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
        tar_dir = -self.ikTarget / np.linalg.norm(self.ikTarget)

        result = None
        result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                  endEffectorLinkIndex=8,
                                                  targetPosition=self.ikTarget - self.iiwa_root_dofs[3:],
                                                  targetOrientation=tar_quat,
                                                  # targetOrientation=tar_dir,
                                                  lowerLimits=self.iiwa_dof_llim.tolist(),
                                                  upperLimits=self.iiwa_dof_ulim.tolist(),
                                                  jointRanges=self.iiwa_dof_jr.tolist(),
                                                  restPoses=self.robot_skeleton.q[6:].tolist()
                                                  )


        self.previousIKResult = result
        self.setPosePyBullet(result)
        self.robot_skeleton.set_velocities(np.zeros(len(self.robot_skeleton.dq)))
        self.robot_skeleton.set_positions(np.concatenate([np.array(self.iiwa_root_dofs), result]))

        hn = self.robot_skeleton.bodynodes[8]  # hand node

        ef_accuracy = np.linalg.norm(hn.to_world(np.zeros(3)) - self.ikTarget)
        retry_count = 0
        while (ef_accuracy > 0.05 and retry_count < 10):
            retry_count += 1
            print("retry " + str(retry_count))
            result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletIiwa,
                                                      endEffectorLinkIndex=8,
                                                      targetPosition=self.ikTarget - self.iiwa_root_dofs[3:],
                                                      targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.iiwa_dof_llim.tolist(),
                                                      upperLimits=self.iiwa_dof_ulim.tolist(),
                                                      jointRanges=self.iiwa_dof_jr.tolist(),
                                                      restPoses=self.robot_skeleton.q[6:].tolist()
                                                      )

            self.previousIKResult = result
            self.setPosePyBullet(result)
            self.robot_skeleton.set_positions(np.concatenate([np.array(self.iiwa_root_dofs), result]))
            ef_accuracy = np.linalg.norm(self.robot_skeleton.bodynodes[8].to_world(np.zeros(3)) - self.ikTarget)
        # DONE: IK setup

        #self.rigidClothFrame.setTransform(hn.world_transform())

        startTest = time.time()
        #self.checkIKSplineSmoothness()
        print("... took " + str(time.time()-startTest) + " time.")

        if self.graphSPDError:
            self.SPDErrorGraph.close()
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)
        state = self._get_obs()

        return state

    def envExtraRender(self):

        #print("party time")
        renderUtils.setColor(color=[1.0,0,0])
        #renderUtils.drawSphere(self.ikTarget)
        pybullet_state = p.getLinkState(self.pyBulletIiwa, 8)[0]
        renderUtils.setColor(color=[0, 1.0, 0])
        #renderUtils.drawSphere(pybullet_state)
        dart_ef = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
        renderUtils.setColor(color=[0, 0, 1.0])
        #renderUtils.drawSphere(dart_ef)

        #self.ikOrientationFrame.setOrg(dart_ef)
        #self.ikOrientationFrame.drawFrame()

        renderUtils.setColor(color=[0, 0, 0])
        hoop_norm = self.rigidClothTargetFrame.toGlobal(np.array([0, 1, 0])) - self.rigidClothTargetFrame.toGlobal(np.zeros(3))
        hoop_norm /= np.linalg.norm(hoop_norm)
        renderUtils.drawArrow(p0=self.rigidClothTargetFrame.org,
                              p1=self.rigidClothTargetFrame.org + hoop_norm * 0.2)

        robo_frame = pyutils.ShapeFrame()
        robo_frame.setTransform(T=self.robot_skeleton.bodynodes[9].T)
        robo_frame.drawFrame()

        renderUtils.setColor([1,0,0])
        for i in range(6,len(self.robot_skeleton.q)):
            if (self.robot_skeleton.q[i] - self.robot_skeleton.dofs[i].position_lower_limit()) < 0.001 or (self.robot_skeleton.dofs[i].position_upper_limit() - self.robot_skeleton.q[i]) < 0.001:
                renderUtils.drawSphere(pos=self.robot_skeleton.bodynodes[i-4].to_world(np.zeros(3)), rad=0.1)

        #draw capability range sphere
        #renderUtils.drawSphere(pos=np.zeros(3), rad=self.capable_range, solid=False)
        #draw capability samples
        for ix,row in enumerate(self.capabilitySamples):
            for iy,col in enumerate(row):
                for iz,sample in enumerate(col):
                    if(sample[0] is not None):
                        if(len(sample) > 1): #we have data
                            if(ix >= self.sampleRenderRange[0][0] and ix <= self.sampleRenderRange[1][0]):
                                if (iy >= self.sampleRenderRange[0][1] and iy <= self.sampleRenderRange[1][1]):
                                    if (iz >= self.sampleRenderRange[0][2] and iz <= self.sampleRenderRange[1][2]):
                                        #norm_error = (sample[1]-self.capabilityStats["best"])/(self.capabilityStats["worst"]-self.capabilityStats["best"])
                                        #renderUtils.setColor(color=[])
                                        cap_color = renderUtils.heatmapColor(minimum=self.capabilityStats["best"], maximum=self.capabilityStats["worst"], value=sample[1])
                                        renderUtils.setColor(color=cap_color)
                                        renderUtils.drawSphere(pos=sample[0], rad=self.capable_range/self.sample_density)


        diskRad = self.robotPathParams['p0_disk_rad']
        depthRange = self.robotPathParams['p0_depth_range']
        org = self.robot_skeleton.bodynodes[3].to_world(np.zeros(3))
        renderUtils.setColor(color=[0.0, 0.0, 0])
        renderUtils.drawCylinder(p0=org + np.array([0, 0, self.robotPathParams['p0_depth_offset']]),
                                     p1=org + np.array([0, 0, depthRange + self.robotPathParams['p0_depth_offset']]),
                                     rad=diskRad)

        # p3 spherical distribution
        p3_distribution = pyutils.EllipsoidFrame(dim=self.robotPathParams['p3_el_dim'],
                                                 org=self.robotPathParams['p3_el_org'])
        p3_distribution.draw()

        self.ikPath.draw()

        renderUtils.setColor(color=[1.0, 0, 0])
        renderUtils.drawSphere(self.ikTarget)
        renderUtils.setColor(color=[0, 1.0, 0])
        # renderUtils.drawLines(lines=[[np.zeros(3), self.sawyer_skel.bodynodes[3].to_world(np.zeros(3))]])
        renderUtils.drawSphere(self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)))

        d_frame = pyutils.BoxFrame(c0=np.array([0.1, 0.2, 0.001]), c1=np.array([-0.1, 0, -0.001]))
        for i in range(5):
            t = i / (4.0)
            d_frame.setQuaternion(
                pyutils.qSLERP(q0=self.orientationEndPoints[0].quat, q1=self.orientationEndPoints[1].quat, t=t))
            d_frame.setOrg(org=self.ikPath.pos(t=t))
            # d_frame.draw()
            d_frame.drawFrame(size=0.1)

        if not self.renderViewWindow:
            renderUtils.drawText(10, 80, "SPD Gains:")
            renderUtils.drawText(10, 60, "  P="+str(self.SPDController.Kp[0][0]))
            renderUtils.drawText(10, 40, "  D="+str(self.SPDController.Kd[0][0]))
            renderUtils.drawText(10, 20, "      balance="+str(self.SPDController.Kd[0][0])+" > " + str(self.SPDController.Kp[0][0]*self.SPDController.h))

        # render target pose
        if self.viewer is not None and self.renderIKShadow:
            q = np.array(self.robot_skeleton.q)
            dq = np.array(self.robot_skeleton.dq)
            self.robot_skeleton.set_positions(np.concatenate([self.iiwa_root_dofs, self.previousIKResult]))
            # self.viewer.scene.render(self.viewer.sim)
            self.robot_skeleton.render()
            self.robot_skeleton.set_positions(q)

        m_viewport = self.viewer.viewport
        textHeight = 15
        textLines = 2

        # draw Sawyer positions vs. limits
        if True:
            for d in range(7):
                #renderUtils.drawText()
                renderUtils.drawText(x=15., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.dofs[6 + d].position_lower_limit(),),
                                         color=(0., 0, 0))
                renderUtils.drawText(x=100., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.q[6 + d],), color=(0., 0, 0))
                renderUtils.drawText(x=200., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.dofs[6 + d].position_upper_limit(),),
                                         color=(0., 0, 0))

                val = (self.robot_skeleton.q[6 + d] - self.robot_skeleton.dofs[6 + d].position_lower_limit()) / (
                self.robot_skeleton.dofs[6 + d].position_upper_limit() - self.robot_skeleton.dofs[6 + d].position_lower_limit())
                tar = (self.previousIKResult[d] - self.robot_skeleton.dofs[6 + d].position_lower_limit()) / (
                self.robot_skeleton.dofs[6 + d].position_upper_limit() - self.robot_skeleton.dofs[6 + d].position_lower_limit())
                renderUtils.drawProgressBar(topLeft=[75, self.viewer.viewport[3] - 450 - d * 20], h=16, w=120, progress=val,
                                            origin=0.5, features=[tar], color=[1.0, 0.0, 0])

                if not self.kinematicIK:
                    renderUtils.drawText(x=250., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.force_lower_limits()[6 + d],),
                                         color=(0., 0, 0))

                    renderUtils.drawText(x=335., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.forces()[6 + d],), color=(0., 0, 0))
                    renderUtils.drawText(x=435., y=self.viewer.viewport[3] - 463 - d * 20,
                                         text="%0.2f" % (self.robot_skeleton.force_upper_limits()[6 + d],), color=(0., 0, 0))

                if not self.kinematicIK:
                    tval = (self.robot_skeleton.forces()[6 + d] - self.robot_skeleton.force_lower_limits()[6 + d]) / (
                    self.robot_skeleton.force_upper_limits()[6 + d] - self.robot_skeleton.force_lower_limits()[6 + d])
                    renderUtils.drawProgressBar(topLeft=[310, self.viewer.viewport[3] - 450 - d * 20], h=16, w=120,
                                                progress=tval, origin=0.5, color=[1.0, 0.0, 0])

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -4
            self._get_viewer().scene.tb.trans[2] = -3.2
            #self._get_viewer().scene.tb._set_orientation(-5.506431209546595, -1.533041108990674)

            #self._get_viewer().scene.tb._trans = [-0.40000000000000019, 0.0, -2.0999999999999988]
            rot = [-0.049641316310301165, -0.1771298414587387, -0.021697643188284738, 0.98269525859233486]
            rot = [0.094279722771759944, -0.86099711940024448, -0.14855547508497791, -0.47720704634449473]
            pyutils.setTrackballOrientation(self.viewer.scene.tb, rot)
        a=0
        #self._get_viewer().scene.tb.trans[2] = -5.5

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = NoRenderWindow(sim, title)
        if not self.disableViewer:
            win = StaticSawyerWindow(sim, title, self, self.envExtraRender)
            win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.2), 'gym_camera')
            win.scene.set_camera(win.scene.num_cameras()-1)

        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            if self.renderViewWindow:
                win.window_size = [self.screen_width, self.screen_height]
            win.run(_show_window=self.visualize)
        return win

    #set a pose in the pybullet simulation env
    def setPosePyBullet(self, pose):
        count = 0
        for i in range(p.getNumJoints(self.pyBulletIiwa)):
            jinfo = p.getJointInfo(self.pyBulletIiwa, i)
            if(jinfo[3] > -1):
                p.resetJointState(self.pyBulletIiwa, i, pose[count])
                count += 1

    #return the pyBullet list of dof positions
    def getPosePyBullet(self):
        pose = []
        for i in range(p.getNumJoints(self.pyBulletIiwa)):
            jinfo = p.getJointInfo(self.pyBulletIiwa, i)
            if (jinfo[3] > -1):
                pose.append(p.getJointState(self.pyBulletIiwa, i)[0])

    #return a scalar metric for the pose trajectory smoothness resulting from following an IK spline
    def checkIKSplineSmoothness(self, samples=100):
        print("testing ik spline")
        self.setPosePyBullet(np.zeros(7)) #reset pose to eliminate variation due to previous spline
        ik_error_info = {'avg': 0, 'sum': 0, 'max': 0, 'min': 0, 'history': []}
        pose_drift_info = {'avg': 0, 'sum': 0, 'max': 0, 'min': 0}

        splineTime = self.ikPath.points[-1].t - self.ikPath.points[0].t
        startTime = self.ikPath.points[0].t
        results = []
        ik_errors = []
        for i in range(samples):
            #iterationStartTime = time.time()
            t = (i/samples)*splineTime + startTime
            self.ikTarget = self.ikPath.pos(t)
            results.append(p.calculateInverseKinematics(self.pyBulletIiwa, 12, self.ikTarget))
            self.setPosePyBullet(results[-1])
            ik_error_info['history'].append(np.linalg.norm(p.getLinkState(self.pyBulletIiwa, 12)[0] - self.ikTarget))
            #print(" ik_error " + str(i) + ": " + str(ik_error_info['history'][-1]))
            ik_error_info['sum'] += ik_error_info['history'][-1]
            if(i==0):
                ik_error_info['max'] = ik_error_info['history'][-1]
                ik_error_info['min'] = ik_error_info['history'][-1]
            else:
                if(ik_error_info['max'] < ik_error_info['history'][-1]):
                    ik_error_info['max'] = ik_error_info['history'][-1]
                if(ik_error_info['min'] > ik_error_info['history'][-1]):
                    ik_error_info['min'] = ik_error_info['history'][-1]
            #print(" iteration time: " + str(time.time()-iterationStartTime))

        ik_error_info['avg'] = ik_error_info['sum']/samples

        print(" ik_error_info: " + str(ik_error_info))

        #compute pose_drift
        for i in range(1,len(results)):
            pose_drift = np.linalg.norm(np.array(results[i])-np.array(results[i-1]))
            pose_drift_info['sum'] += pose_drift
            if (i == 0):
                pose_drift_info['max'] = pose_drift
                pose_drift_info['min'] = pose_drift
            else:
                if (pose_drift_info['max'] < pose_drift):
                    pose_drift_info['max'] = pose_drift
                if (pose_drift_info['min'] > pose_drift):
                    pose_drift_info['min'] = pose_drift
        pose_drift_info['avg'] = pose_drift_info['sum']/(samples-1)
        print(" pose_drift_info = " + str(pose_drift_info))
        self.setPosePyBullet(np.zeros(7))  # reset pose to eliminate variation due to previous spline

    def saveCapabilityData(self, filename="capability_data"):
        f = open(filename, 'w')
        f.write("org " + str(0) + " " + str(0) + " " + str(0) + "\n") #TODO: other orgs relative to robot?
        f.write("dim " + str(self.capable_range) + "\n")
        f.write("sample_density " + str(self.sample_density) + "\n")
        for ix,row in enumerate(self.capabilitySamples):
            for iy,col in enumerate(row):
                for iz,sample in enumerate(col):
                    if(sample[0] is not None): #this sample is in the sphere
                        f.write("s ")
                        if(len(sample) > 1): #we have data
                            f.write(str(sample[1]))
                        f.write("\n")
                    else: #placeholder
                        f.write("none \n")
        f.close()

    #initialize the capability sample array given the number of samples and dimension
    def initializeCapabilitySamples(self):
        self.capabilitySamples = []
        self.capabilityStats = {"best":9999, "worst":0, "total":0, "avg":0, "numSamples":0}
        bottomCorner = np.array([-self.capable_range, -self.capable_range, -self.capable_range])
        self.numCapabilitySamples = 0
        self.capabilityTarget = [0, 0, 0]
        for ix in range(self.sample_density):
            self.capabilitySamples.append([])
            for iy in range(self.sample_density):
                self.capabilitySamples[-1].append([])
                for iz in range(self.sample_density):
                    dx = self.capable_range * 2.0 * (ix / (self.sample_density - 1))
                    dy = self.capable_range * 2.0 * (iy / (self.sample_density - 1))
                    dz = self.capable_range * 2.0 * (iz / (self.sample_density - 1))
                    point = bottomCorner + np.array([dx, dy, dz])
                    if (np.linalg.norm(point) <= self.capable_range):
                        self.capabilitySamples[-1][-1].append([point])
                        self.numCapabilitySamples += 1
                    else:  # if the sample point is not in the sphere, don't bother with it
                        self.capabilitySamples[-1][-1].append([None])

    def loadCapabilityData(self, filename="capability_data_orientation"):
        f = open(filename, 'r')
        lIxs = [0,0,0] #datastructure load indices
        initializedCapabilitySamples = False
        for ix, line in enumerate(f):
            words = line.split()
            if(words[0] == "org"):
                #TODO
                a=0
            elif(words[0] == "dim"):
                self.capable_range = float(words[1])
            elif(words[0] == "sample_density"):
                self.sample_density = int(words[1])
            else:
                if not initializedCapabilitySamples: #do this the first time
                    self.initializeCapabilitySamples()
                    initializedCapabilitySamples = True
                if(words[0] == "s"):
                    val = float(words[1])
                    #print("line " + str(ix))
                    #print("lIxs: " + str(lIxs))
                    #print("||self.capabilitySamples||: " + str(len(self.capabilitySamples)) + ", " + str(len(self.capabilitySamples[lIxs[0]])) + ", " + str(len(self.capabilitySamples[lIxs[0]][lIxs[1]])))
                    self.capabilitySamples[lIxs[0]][lIxs[1]][lIxs[2]].append(val)
                    self.capabilityStats["best"] = min(self.capabilityStats["best"], val)
                    self.capabilityStats["worst"] = max(self.capabilityStats["worst"], val)
                    self.capabilityStats["numSamples"] += 1
                    self.capabilityStats["total"] += val
                    self.capabilityStats["avg"] = self.capabilityStats["total"] / self.capabilityStats["numSamples"]

                lIxs[2] += 1
                if lIxs[2] >= (self.sample_density):
                    lIxs[2] = 0
                    lIxs[1] += 1
                if lIxs[1] >= (self.sample_density):
                    lIxs[1] = 0
                    lIxs[0] += 1
        f.close()

        #print capability map statistics:
        print("Capability map stats:")
        print(self.capabilityStats)