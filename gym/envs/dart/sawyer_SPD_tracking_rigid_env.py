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

try:
    import pyPhysX as pyphysx
    pyphysx.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pyphysx.)".format(e))

import pyPhysX.pyutils as pyutils
import pyPhysX.renderUtils as renderUtils

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

        self.h = 0.01
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

class DartSawyerSPDTrackingRigidEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([np.ones(13), -1*np.ones(13)])
        #self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        obs_dim = 14
        self.param_manager = hopperContactMassManager(self)
        self.kinematicIK = False #if false, dynamic SPD
        self.previousIKResult = np.zeros(7)
        self.viewer = None

        #SPD error graphing per dof
        self.graphSPDError = True
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
        self.pyBulletSawyer = p.loadURDF(dir_path + '/assets/sawyer_description/urdf/sawyer_arm.urdf')
        print("Sawyer bodyID: " + str(self.pyBulletSawyer))
        self.ikPath = pyutils.Spline()
        self.ikPath.addPoint(0, 0.5)
        self.ikPath.addPoint(0.5, 0.5)
        self.ikPath.addPoint(1.0, 0.5)
        self.ikPathTimeScale = 0.01 #relationship between number of steps and spline time
        self.ikTarget = np.array([0.5, 0, 0])
        print("Number of joint: " + str(p.getNumJoints(self.pyBulletSawyer)))
        for i in range(p.getNumJoints(self.pyBulletSawyer)):
            jinfo = p.getJointInfo(self.pyBulletSawyer, i)
            print(" " + str(jinfo[0]) + " " + str(jinfo[1]) + " " + str(jinfo[2]) + " " + str(jinfo[3]) + " " + str(jinfo[12]))
        #result = p.calculateInverseKinematics(self.pyBulletSawyer, 15, self.ikTarget)
        #print(result)

        self.numSteps = 0

        #initialize the variable
        self.SPDController = None

        dart_env.DartEnv.__init__(self, 'sawyer_description/urdf/sawyer_arm.urdf', 5, obs_dim, self.control_bounds, disableViewer=True)

        #initialize the controller
        self.SPDController = SPDController(self)

        #self.dart_world.set_collision_detector(3) # 3 is ode collision detector

        utils.EzPickle.__init__(self)

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
        self.ikTarget = self.ikPath.pos(self.numSteps*self.ikPathTimeScale)
        #self.ikTarget = np.array(self.ikPath.points[-1].p)
        lowerLimits = (np.ones(7)*-4).tolist()
        upperLimits = (np.ones(7)*4).tolist()
        jointRanges = (np.ones(7)*4).tolist()
        #restPoses = (np.zeros(7)).tolist()
        restPoses = self.robot_skeleton.q[6:].tolist()
        result = p.calculateInverseKinematics(self.pyBulletSawyer, 12, self.ikTarget)
        self.previousIKResult = np.array(result)
        self.setPosePyBullet(result)
        #print(p.getLinkState(self.pyBulletSawyer, 12, computeForwardKinematics=True)[0])
        #result = p.calculateInverseKinematics(self.pyBulletSawyer, 12, self.ikTarget, lowerLimits=lowerLimits, upperLimits=upperLimits, jointRanges=jointRanges, restPoses=restPoses)

        if(self.kinematicIK):
            #kinematic
            self.robot_skeleton.set_positions(np.concatenate([np.zeros(6), result]))
        else:
            #SPD (dynamic)
            if self.SPDController is not None:
                self.SPDController.target = result
                tau = np.concatenate([np.zeros(6), self.SPDController.query(obs=None)])
                self.do_simulation(tau, self.frame_skip)
                print(self.robot_skeleton.q)

        ik_error = np.linalg.norm(p.getLinkState(self.pyBulletSawyer, 12)[0] - self.ikTarget)
        pose_error = self.robot_skeleton.q[6:]-result
        pose_error_mag = np.linalg.norm(self.robot_skeleton.q[6:]-result)

        if self.graphSPDError:
            self.SPDErrorGraph.addToLinePlot(data=pose_error.tolist())

        print("Errors: IK=" + str(ik_error) + ", SPD=" + str(pose_error_mag))


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
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        #test IK reset
        self.ikPath = pyutils.Spline()
        dart_ef = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        self.ikPath.addPoint(0, 0.0)
        self.ikPath.addPoint(0.5, 0.5)
        self.ikPath.addPoint(1.0, 0.5)
        self.ikPath.points[0].p = np.array(dart_ef)
        print("checking IK spline...")
        startTest = time.time()
        self.checkIKSplineSmoothness()
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
        renderUtils.drawSphere(self.ikTarget)
        pybullet_state = p.getLinkState(self.pyBulletSawyer, 12)[0]
        renderUtils.setColor(color=[0, 1.0, 0])
        renderUtils.drawSphere(pybullet_state)
        dart_ef = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        renderUtils.setColor(color=[0, 0, 1.0])
        renderUtils.drawSphere(dart_ef)

        self.ikPath.draw()
        renderUtils.drawText(10, 80, "SPD Gains:")
        renderUtils.drawText(10, 60, "  P="+str(self.SPDController.Kp[0][0]))
        renderUtils.drawText(10, 40, "  D="+str(self.SPDController.Kd[0][0]))
        renderUtils.drawText(10, 20, "      balance="+str(self.SPDController.Kd[0][0])+" > " + str(self.SPDController.Kp[0][0]*self.SPDController.h))

        # render target pose
        if self.viewer is not None:
            q = np.array(self.robot_skeleton.q)
            dq = np.array(self.robot_skeleton.dq)
            self.robot_skeleton.set_positions(np.concatenate([np.zeros(6), self.previousIKResult]))
            # self.viewer.scene.render(self.viewer.sim)
            self.robot_skeleton.render()
            self.robot_skeleton.set_positions(q)

    def viewer_setup(self):
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
            win.run(_show_window=self.visualize)
        return win

    #set a pose in the pybullet simulation env
    def setPosePyBullet(self, pose):
        count = 0
        for i in range(p.getNumJoints(self.pyBulletSawyer)):
            jinfo = p.getJointInfo(self.pyBulletSawyer, i)
            if(jinfo[3] > -1):
                p.resetJointState(self.pyBulletSawyer, i, pose[count])
                count += 1

    #return the pyBullet list of dof positions
    def getPosePyBullet(self):
        pose = []
        for i in range(p.getNumJoints(self.pyBulletSawyer)):
            jinfo = p.getJointInfo(self.pyBulletSawyer, i)
            if (jinfo[3] > -1):
                pose.append(p.getJointState(self.pyBulletSawyer, i)[0])

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
            results.append(p.calculateInverseKinematics(self.pyBulletSawyer, 12, self.ikTarget))
            self.setPosePyBullet(results[-1])
            ik_error_info['history'].append(np.linalg.norm(p.getLinkState(self.pyBulletSawyer, 12)[0] - self.ikTarget))
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

