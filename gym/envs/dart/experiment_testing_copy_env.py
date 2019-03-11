from gym.envs.dart.dart_cloth_iiwa_env import *

class DartClothExperimentTestingEnv(DartClothIiwaEnv):
    def __init__(self):
        dual_policy = True
        is_human = True

        self.limbNodesR = [3, 4, 5, 6, 7]
        self.limbNodesL = [8, 9, 10, 11, 12]

        #setup robot root dofs
        # setup robot base location
        self.dFromRoot = 0.75
        #self.dFromRoot = 2.75
        self.aFrom_invZ = 1.1
        self.iiwa_root_dofs = []  # values for the fixed 6 dof root transformation

        # robots mirrored around z
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot * math.sin(self.aFrom_invZ), -0.2, -self.dFromRoot * math.cos(self.aFrom_invZ)]))
        #self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot * math.sin(-self.aFrom_invZ), -0.2, -self.dFromRoot * math.cos(-self.aFrom_invZ)]))

        #initialize the base env
        cloth_mesh_file = "fullgown1.obj"
        cloth_mesh_state_file = "hanginggown.obj"
        DartClothIiwaEnv.__init__(self, robot_root_dofs=self.iiwa_root_dofs, active_compliance=False, cloth_mesh_file=cloth_mesh_file, cloth_mesh_state_file=cloth_mesh_state_file, cloth_scale=1.3, dual_policy=dual_policy, is_human=is_human)

        #setup features
        self.sleeveRVerts = [532, 451, 251, 252, 253, 1334, 1320, 1184, 945, 985, 1062, 1607, 1037, 484, 1389, 679, 1230, 736, 1401, 1155, 486, 1410]
        self.sleeveLVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.cloth_features.append(self.sleeveLSeamFeature)
        #self.cloth_features.append(self.sleeveRSeamFeature)

        #setup separated meshes
        # setup specific separated meshes and dressing targets for this task
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # left sleeve

        #self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # right sleeve
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveLSeamFeature, limb_sequence=self.limbNodesL, distal_offset=self.fingertip))
        #self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveRSeamFeature, limb_sequence=self.limbNodesR, distal_offset=self.fingertip))

        #manual control target
        #self.iiwas[0].iiwa_frame_controller = IiwaLimbTraversalController(env=self, skel=self.human_skel, iiwa=self.iiwas[0], limb=self.limbNodesL, ef_offset=self.fingertip, offset_dists=[0.1, 0.1, 0.1, 0.15, 0.18, 0.18])
        #self.iiwas[0].iiwa_frame_controller = IiwaApproachHoverProceedAvoidController(self, self.iiwas[0], dressingTargets=self.dressing_targets, target_node=8, node_offset=np.array([0.21, 0.1, 0]), distance=0.1, noise=0.0, control_fraction=0.3, slack=(0.1, 0.075), hold_time=0.75, avoidDist=0.05, hold_elevation_node=11, hold_elevation_node_offset=np.array([0,00.05,-0.15]))
        #self.iiwas[0].iiwa_frame_controller = IiwaApproachHoverProceedAvoidMultistageController(self, self.iiwas[0], dressing_targets=self.dressing_targets, target_nodes=[11,10,8], node_offsets=[np.array([0,0,-0.15]), np.array([0,0.05,-0.1]), np.array([0.15, 0.12, 0]) ], distances=[0.1, 0.15, 0.1], control_fraction=0.3, slack=(0.05, 0.075), hold_time=0.75, avoid_dist=0.08)


        #setup handle nodes
        self.iiwas[0].addClothHandle(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131], offset=np.array([0, 0, 0.05]))
        #self.iiwas[0].addClothHandle(verts=[1552], offset=np.array([0, 0, 0.05]))
        #self.iiwas[0].addClothHandle(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131], offset=np.array([0, 0, 0]))

        #setup human obs
        self.human_obs_manager.addObsFeature(feature=ProprioceptionObsFeature(skel=self.human_skel, name="human proprioception"))
        self.human_obs_manager.addObsFeature(feature=HumanHapticObsFeature(self,render=True))
        self.human_obs_manager.addObsFeature(feature=JointPositionObsFeature(self.human_skel, name="human joint positions"))
        self.human_obs_manager.addObsFeature(feature=SPDTargetObsFeature(self))
        #self.human_obs_manager.addObsFeature(feature=DataDrivenJointLimitsObsFeature(self))
        #self.human_obs_manager.addObsFeature(feature=CollisionMPCObsFeature(env=self,is_human=True))
        self.human_obs_manager.addObsFeature(feature=WeaknessScaleObsFeature(self,self.limbDofs[1],scale_range=(0.1,0.3)))
        self.human_obs_manager.addObsFeature(feature=OracleObsFeature(env=self,sensor_ix=21,dressing_target=self.dressing_targets[-1],sep_mesh=self.separated_meshes[-1]))
        for iiwa in self.iiwas:
            self.human_obs_manager.addObsFeature(feature=JointPositionObsFeature(iiwa.skel, ignored_joints=[1], name="iiwa " + str(iiwa.index) + " joint positions"))

        #setup robot obs
        for iiwa in self.iiwas:
            self.robot_obs_manager.addObsFeature(feature=ProprioceptionObsFeature(skel=iiwa.skel, start_dof=6, name="iiwa " + str(iiwa.index) + "proprioception"))
            self.robot_obs_manager.addObsFeature(feature=JointPositionObsFeature(iiwa.skel, ignored_joints=[1], name="iiwa " + str(iiwa.index) + " joint positions"))
            self.robot_obs_manager.addObsFeature(feature=RobotFramesObsFeature(iiwa, name="iiwa " + str(iiwa.index) + " frame"))
            self.robot_obs_manager.addObsFeature(feature=CapacitiveSensorObsFeature(iiwa, name="iiwa " + str(iiwa.index) + " cap sensor"))
            self.robot_obs_manager.addObsFeature(feature=FTSensorObsFeature(self, iiwa, name="iiwa " + str(iiwa.index) + " FT sensor"))
        #self.robot_obs_manager.addObsFeature(feature=CollisionMPCObsFeature(env=self, is_human=False))
        self.robot_obs_manager.addObsFeature(feature=JointPositionObsFeature(self.human_skel, name="human joint positions"))

        #setup rewards
        rest_pose_weights = np.ones(self.human_skel.ndofs)
        rest_pose_weights[:2] *= 40 #stable torso
        rest_pose_weights[2] *= 4 #spine
        rest_pose_weights[3:11] *= 4 #passive arm
        rest_pose_weights[11:19] *= 0.5 #active arm
        #rest_pose_weights[3:19] *= 0 #ignore rest pose
        rest_pose_weights[19:] *= 8 #stable head
        self.reward_manager.addTerm(term=RestPoseRewardTerm(self.human_skel, pose=np.zeros(self.human_skel.ndofs), weights=rest_pose_weights))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[0], terminal=True, weight=50))
        self.reward_manager.addTerm(term=ClothDeformationRewardTerm(self, weight=1))
        self.reward_manager.addTerm(term=HumanContactRewardTerm(self, weight=1, tanh_params=(2, 0.15, 10))) #saturates at ~10 and ~38

        #set the observation space
        self.obs_dim = self.human_obs_manager.obs_size
        if self.dual_policy:
            self.obs_dim += self.robot_obs_manager.obs_size
        elif not self.is_human:
            self.obs_dim = self.robot_obs_manager.obs_size

        self.observation_space = spaces.Box(np.inf * np.ones(self.obs_dim) * -1.0, np.inf * np.ones(self.obs_dim))
        print(self.observation_space)

    def additionalResets(self):
        # setting the orientation of the pyBulletIiwa, other settings are unnecessary as we give rest poses for IK

        #set manual target to random pose
        if self.manual_human_control:
            self.human_manual_target = self.getValidRandomPose(verbose=False)
            self.human_manual_target = np.array([0.0, 0.0, 0.0, -0.09486478804170062, 0.16919563098552753, -0.4913244737893412, -1.371164742525659, -0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, -0.09486478804170062, 0.16919563098552753, 0.4913244737893412, -1.371164742525659, 0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, 0.48155552859527917, -0.13660824713013747, 0.6881130165905589])

            print("chose manual target = " + str(self.human_manual_target.tolist()))
            for iiwa in self.iiwas:
                iiwa.setRestPose()

            #TODO: remove after testing
            #self.human_skel.set_positions([0.0, 0.0, 0.0, -0.21890184289240233, 0.1618533105311784, -0.03417282760690066, 0.670498809614021, -0.16780524349209935, 1.8045016700105585, -0.3012597961534294, 0.4064480138415224, -0.21890184289240233, 0.1618533105311784, 0.03417282760690066, 0.670498809614021, 0.16780524349209935, 1.8045016700105585, -0.3012597961534294, 0.4064480138415224, 0.2530563478930248, -0.5648952906859239, 0.9915228996786887])
            #self.human_skel.set_positions([0.0, 0.0, 0.0, -0.09486478804170062, 0.16919563098552753, -0.4913244737893412, -1.371164742525659, -0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, -0.09486478804170062, 0.16919563098552753, 0.4913244737893412, -1.371164742525659, 0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, 0.48155552859527917, -0.13660824713013747, 0.6881130165905589])
            #self.humanSPDIntperolationTarget = np.array(self.human_skel.q)

        if self.manual_robot_control:
            for iiwa in self.iiwas:
                valid = False
                while not valid:
                    valid = True
                    cen = iiwa.skel.bodynodes[3].to_world(np.zeros(3))
                    root = iiwa.skel.bodynodes[0].to_world(np.zeros(3))
                    iiwa.manualFrameTarget.org = pyutils.sampleDirections(num=1)[0]*random.uniform(0.2,0.7) + cen
                    rand_eulers = np.random.uniform(low=iiwa.orientationEulerStateLimits[0], high=iiwa.orientationEulerStateLimits[1])
                    iiwa.manualFrameTarget.orientation = pyutils.euler_to_matrix(rand_eulers)
                    iiwa.manualFrameTarget.updateQuaternion()
                    if iiwa.manualFrameTarget.org[0] > cen[0]:
                        valid = False
                    if iiwa.manualFrameTarget.org[2] > -0.2:
                        valid = False
                    if abs(iiwa.manualFrameTarget.org[1]) > 0.2:
                        valid = False
                    if np.linalg.norm(iiwa.manualFrameTarget.org-root) < 0.2:
                        valid = False


        T = self.iiwas[0].skel.bodynodes[0].world_transform()
        tempFrame = pyutils.ShapeFrame()
        tempFrame.setTransform(T)
        root_quat = tempFrame.quat
        root_quat = (root_quat.x, root_quat.y, root_quat.z, root_quat.w)
        p.resetBasePositionAndOrientation(self.pyBulletIiwa, posObj=np.zeros(3), ornObj=root_quat)

        #initialize the robot pose
        # pick p0 with rejection sampling
        diskRad = 0.7
        good = False
        r_pivot = self.iiwas[0].skel.bodynodes[3].to_world(np.zeros(3))
        p0 = r_pivot + pyutils.sampleDirections(num=1)[0] * diskRad
        depth_offset = 0.15
        while (not good):
            good = True
            # xz0 = np.array([p0[0], p0[2]])

            # cut off points too close or behind the robot in x
            if p0[0] > (r_pivot[0] - depth_offset):
                good = False

            # cut off points too close or behind the robot in z
            if p0[2] > (r_pivot[2] - depth_offset):
                good = False

            if not good:
                p0 = r_pivot + pyutils.sampleDirections(num=1)[0] * diskRad

        self.iiwas[0].ik_target.setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=p0)
        self.iiwas[0].computeIK(maxIter=300)
        self.iiwas[0].skel.set_velocities(np.zeros(len(self.iiwas[0].skel.dq)))
        self.iiwas[0].setIKPose() #frame set in here too

        #initialize the garment location
        hn = self.iiwas[0].skel.bodynodes[self.iiwas[0].handle_bodynode]

        self.iiwas[0].handle_node.setOrgToCentroid()
        self.iiwas[0].handle_node.setOrientation(R=hn.T[:3, :3])
        self.clothScene.translateCloth(0, -self.iiwas[0].handle_node.org)

        self.clothScene.rotateCloth(cid=0, R=pyutils.rotateX(-math.pi / 2.0))
        self.clothScene.rotateCloth(cid=0, R=hn.T[:3, :3])

        self.clothScene.translateCloth(0, hn.to_world(self.iiwas[0].handle_node_offset))
        self.iiwas[0].handle_node.setOrgToCentroid()

        self.iiwas[0].handle_node.recomputeOffsets()
        self.iiwas[0].handle_node.updatePrevConstraintPositions()

        #ensure feature normal direction
        self.sleeveLSeamFeature.fitPlane()
        from_origin = self.sleeveLSeamFeature.plane.org
        from_origin = from_origin / np.linalg.norm(from_origin)
        self.sleeveLSeamFeature.fitPlane(normhint=from_origin)

        #using the expected normal of the plane, so do this here only once
        if self.reset_number == 0:
            self.separated_meshes[0].initSeparatedMeshGraph()
            self.separated_meshes[0].updateWeights()
            self.separated_meshes[0].computeGeodesic(feature=self.sleeveLSeamFeature, oneSided=True, side=0, normalSide=1)

    def _getFile(self):
        return __file__