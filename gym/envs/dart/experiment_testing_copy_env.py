from gym.envs.dart.dart_cloth_iiwa_env import *

class DartClothExperimentTestingEnv(DartClothIiwaEnv):
    def __init__(self):
        dual_policy = True
        is_human = True

        self.limbNodesR = [3, 4, 5, 6, 7]
        self.limbNodesL = [8, 9, 10, 11, 12]
        self.limbNodesH = [13,14] #neck/head limb
        self.limbNodesB = [1,13,14] #torso/neck/head limb
        self.limbNodesRT = [1,3, 4, 5, 6, 7] #rigt arm and torso
        self.limbNodesLT = [1,8, 9, 10, 11, 12] #left arm and torso

        #setup robot root dofs
        # setup robot base location
        self.dFromRoot = 0.75
        #self.dFromRoot = 2.75
        self.aFrom_invZ = 1.1
        self.iiwa_root_dofs = []  # values for the fixed 6 dof root transformation

        # robots mirrored around z
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot * math.sin(self.aFrom_invZ), -0.2, -self.dFromRoot * math.cos(self.aFrom_invZ)]))
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot * math.sin(-self.aFrom_invZ), -0.2, -self.dFromRoot * math.cos(-self.aFrom_invZ)]))

        #initialize the base env
        cloth_mesh_file = "tshirt_m.obj"
        #cloth_mesh_state_file = "hanginggown.obj"
        #cloth_mesh_state_file = "tshirt_m.obj"
        cloth_mesh_state_file = "twoArmTshirtHang.obj"
        DartClothIiwaEnv.__init__(self, robot_root_dofs=self.iiwa_root_dofs, active_compliance=False, cloth_mesh_file=cloth_mesh_file, cloth_mesh_state_file=cloth_mesh_state_file, cloth_scale=1.5, dual_policy=dual_policy, is_human=is_human)

        #setup features
        self.sleeveRVerts = [2580, 2495, 2508, 2586, 2518, 2560, 2621, 2529, 2559, 2593, 272, 2561, 2658, 2582, 2666, 2575, 2584, 2625, 2616, 2453, 2500, 2598, 2466]
        self.sleeve_R_border_vertices = [264, 263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 10, 265]
        self.sleeveLVerts = [211, 2305, 2364, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2276, 2326, 2334, 2288, 2346, 2314, 2251, 2347, 2304, 2245, 2376, 2315]
        self.sleeve_L_border_vertices = [229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 9, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230]
        self.collarVertices = [1674, 2011, 1371, 1836, 2030, 1564, 106, 901, 306, 460, 1052, 900, 458, 478, 663, 761, 611, 1067, 429, 657, 1179, 428, 427, 884, 123, 2192, 1720, 2034, 1379, 1226, 1994, 1858, 1322, 2033, 1857]
        #self.collar_border_vertices can be excluded from geodesic search
        self.collar_border_vertices = [8, 192, 193, 194, 195, 196, 197, 4, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 5, 186, 187, 188, 189, 190, 191]
        self.waistVertices = [69, 454, 1028, 827, 982, 1171, 398, 1024, 421, 707, 1217, 987, 638, 1155, 361, 924, 774, 637, 347, 1043, 738, 1156, 943, 572, 1058, 472, 736, 964, 573, 956, 823, 442, 890, 30, 1853, 1979, 1919, 1250, 2113, 1418, 1315, 1888, 1805, 1400, 2027, 2060, 1966, 1412, 1633, 1429, 2117, 1457, 2088, 1332, 1529, 1880, 1516, 1783, 1307, 2193, 1462, 1864, 1742, 2066, 1863, 2057]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.waistFeature = ClothFeature(verts=self.waistVertices, clothScene=self.clothScene)
        self.cloth_features.append(self.sleeveLSeamFeature)
        self.cloth_features.append(self.sleeveRSeamFeature)
        self.cloth_features.append(self.collarFeature)
        self.cloth_features.append(self.waistFeature)

        #setup separated meshes
        # setup specific separated meshes and dressing targets for this task
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # left sleeve
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # right sleeve
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # collar
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # waist

        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveLSeamFeature, limb_sequence=self.limbNodesL, distal_offset=self.fingertip))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveRSeamFeature, limb_sequence=self.limbNodesR, distal_offset=self.fingertip))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.collarFeature, limb_sequence=self.limbNodesH, distal_offset=np.array([0,0.25,0])))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.waistFeature, limb_sequence=self.limbNodesB, distal_offset=np.array([0,0.25,0])))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.waistFeature, limb_sequence=self.limbNodesLT, distal_offset=self.fingertip))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.waistFeature, limb_sequence=self.limbNodesRT, distal_offset=self.fingertip))

        #manual control target
        #self.iiwas[0].iiwa_frame_controller = IiwaLimbTraversalController(env=self, skel=self.human_skel, iiwa=self.iiwas[0], limb=self.limbNodesL, ef_offset=self.fingertip, offset_dists=[0.1, 0.1, 0.1, 0.15, 0.18, 0.18])
        #self.iiwas[0].iiwa_frame_controller = IiwaApproachHoverProceedAvoidController(self, self.iiwas[0], dressingTargets=self.dressing_targets, target_node=8, node_offset=np.array([0.21, 0.1, 0]), distance=0.4, noise=0.0, control_fraction=0.3, slack=(0.1, 0.075), hold_time=1.0, avoidDist=0.1)
        #self.iiwas[1].iiwa_frame_controller = IiwaApproachHoverProceedAvoidController(self, self.iiwas[1], dressingTargets=self.dressing_targets, target_node=3, node_offset=np.array([-0.21, 0.1, 0]), distance=0.4, noise=0.0, control_fraction=0.3, slack=(0.1, 0.075), hold_time=1.0, avoidDist=0.1)
        #TODO: setup the manual track for iiwa
        rp1 = [-0.33655809,  0.16736516, -0.32865326]
        rq1 = [0.562261291382924, 0.6261955898307, 0.537049389555349, 0.05761316400976]

        #setup handle nodes
        self.iiwas[0].addClothHandle(verts=[1251, 1724, 1402, 1853, 1629, 2111, 1683, 2185, 1562, 1979, 1919, 1249, 1854, 2000, 1250, 1399, 1917, 1438, 1716, 1281, 1639, 1715, 1561, 2113, 1785, 1418, 1851], offset=np.array([0, 0, 0.05]))
        self.iiwas[1].addClothHandle(verts=[66, 67, 68, 69, 2057, 1274, 1723, 2040, 1243, 1863, 1617, 2066, 2039, 1744, 1627, 70, 2055, 1690, 1742, 1415, 2170, 1242, 1936, 1784, 1461, 1305, 1864, 1846, 1272, 1883], offset=np.array([0, 0, 0.05]))

        #setup human obs
        self.human_obs_manager.addObsFeature(feature=ProprioceptionObsFeature(skel=self.human_skel, name="human proprioception"))
        self.human_obs_manager.addObsFeature(feature=HumanHapticObsFeature(self,render=True))
        self.human_obs_manager.addObsFeature(feature=JointPositionObsFeature(self.human_skel, name="human joint positions"))
        self.human_obs_manager.addObsFeature(feature=SPDTargetObsFeature(self))
        #self.human_obs_manager.addObsFeature(feature=DataDrivenJointLimitsObsFeature(self))
        #self.human_obs_manager.addObsFeature(feature=CollisionMPCObsFeature(env=self,is_human=True))
        #self.human_obs_manager.addObsFeature(feature=WeaknessScaleObsFeature(self,self.limbDofs[1],scale_range=(0.1,0.4)))
        #self.human_obs_manager.addObsFeature(feature=WeaknessScaleObsFeature(self,self.limbDofs[0],scale_range=(0.1,0.4)))
        self.human_obs_manager.addObsFeature(feature=OracleObsFeature(env=self,sensor_ix=21,dressing_target=self.dressing_targets[0],sep_mesh=self.separated_meshes[0]))
        self.human_obs_manager.addObsFeature(feature=OracleObsFeature(env=self,sensor_ix=12,dressing_target=self.dressing_targets[1],sep_mesh=self.separated_meshes[1]))
        self.human_obs_manager.addObsFeature(feature=OracleObsFeature(env=self,sensor_ix=3,dressing_target=self.dressing_targets[2],sep_mesh=self.separated_meshes[2]))
        #TODO: head oracle?
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
        #pose: arms outreached above head
        rest_pose = np.array([0.0, 0.0, 0.0, 0.2014567442644234, 0.12976885838990154, 0.07445680418190292, 3.95336417358366, -0.9002739292338819, 0.29925007698275996, 0.4400513472819564, 0.0051886712832222015, 0.2014567442644234, 0.12976885838990154, -0.07445680418190292, 3.95336417358366, 0.9002739292338819, 0.29925007698275996, 0.4400513472819564, 0.0051886712832222015, 0.0, 0.0, 0.0])
        rest_pose_weights = np.ones(self.human_skel.ndofs)
        rest_pose_weights[:2] *= 40 #stable torso
        rest_pose_weights[2] *= 5 #spine
        #rest_pose_weights[3:11] *= 0 #ignore active arm
        #rest_pose_weights[11:19] *= 2 #passive arm
        #rest_pose_weights[3:19] *= 0 #ignore rest pose
        rest_pose_weights[19:] *= 3 #stable head
        self.reward_manager.addTerm(term=RestPoseRewardTerm(self.human_skel, pose=rest_pose, weights=rest_pose_weights))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[0], terminal=True, success_threshold=0.7, weight=20))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[1], terminal=True, success_threshold=0.7, weight=20))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[2], terminal=True, success_threshold=0.7, weight=20))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[3], terminal=True, success_threshold=1.0, weight=20))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[4], terminal=False, success_threshold=1.0, weight=20))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[5], terminal=False, success_threshold=1.0, weight=20))
        self.reward_manager.addTerm(term=GeodesicContactRewardTerm(sensor_index=21, env=self, separated_mesh=self.separated_meshes[0], dressing_target=self.dressing_targets[0], weight=15))
        self.reward_manager.addTerm(term=GeodesicContactRewardTerm(sensor_index=12, env=self, separated_mesh=self.separated_meshes[1], dressing_target=self.dressing_targets[1], weight=15))
        self.reward_manager.addTerm(term=GeodesicContactRewardTerm(sensor_index=3, env=self, separated_mesh=self.separated_meshes[2], dressing_target=self.dressing_targets[2], weight=15))

        self.reward_manager.addTerm(term=ClothDeformationRewardTerm(self, weight=25))
        self.reward_manager.addTerm(term=HumanContactRewardTerm(self, weight=5, tanh_params=(2, 0.15, 10)))

        self.reward_manager.addTerm(term=BodyDistancePenaltyTerm(self, node1=self.iiwas[0].skel.bodynodes[8], offset1=np.zeros(3), node2=self.iiwas[1].skel.bodynodes[8], offset2=np.zeros(3), target_range=(0,0.4), weight=5))


        #set the observation space
        self.obs_dim = self.human_obs_manager.obs_size
        if self.dual_policy:
            self.obs_dim += self.robot_obs_manager.obs_size
        elif not self.is_human:
            self.obs_dim = self.robot_obs_manager.obs_size

        self.observation_space = spaces.Box(np.inf * np.ones(self.obs_dim) * -1.0, np.inf * np.ones(self.obs_dim))
        print(self.observation_space)

        # head bending
        dof_change = 0.4
        dof = 20
        self.human_skel.dof(dof).set_position_lower_limit(
            self.human_skel.dof(dof).position_lower_limit() - dof_change)
        self.human_skel.dof(dof).set_position_upper_limit(
            self.human_skel.dof(dof).position_upper_limit() + dof_change)

    def additionalResets(self):
        # setting the orientation of the pyBulletIiwa, other settings are unnecessary as we give rest poses for IK

        #set manual target to random pose
        if self.manual_human_control:
            self.human_manual_target = self.getValidRandomPose(verbose=False,symmetrical=True)
            self. human_manual_target = np.array([0.0, 0.0, 0.0, 0.2014567442644234, 0.12976885838990154, 0.07445680418190292, 3.95336417358366, -0.9002739292338819, 0.29925007698275996, 0.4400513472819564, 0.0051886712832222015, 0.2014567442644234, 0.12976885838990154, -0.07445680418190292, 3.95336417358366, 0.9002739292338819, 0.29925007698275996, 0.4400513472819564, 0.0051886712832222015, 0.0, 0.0, 0.0])
            #self.human_manual_target = np.array([0.0, 0.0, 0.0, -0.09486478804170062, 0.16919563098552753, -0.4913244737893412, -1.371164742525659, -0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, -0.09486478804170062, 0.16919563098552753, 0.4913244737893412, -1.371164742525659, 0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, 0.48155552859527917, -0.13660824713013747, 0.6881130165905589])
            self.human_skel.set_positions(self.human_manual_target)
            print("chose manual target = " + str(self.human_manual_target.tolist()))
            for iiwa in self.iiwas:
                iiwa.setRestPose()

            #TODO: remove after testing
            #self.human_skel.set_positions([0.0, 0.0, 0.0, -0.21890184289240233, 0.1618533105311784, -0.03417282760690066, 0.670498809614021, -0.16780524349209935, 1.8045016700105585, -0.3012597961534294, 0.4064480138415224, -0.21890184289240233, 0.1618533105311784, 0.03417282760690066, 0.670498809614021, 0.16780524349209935, 1.8045016700105585, -0.3012597961534294, 0.4064480138415224, 0.2530563478930248, -0.5648952906859239, 0.9915228996786887])
            #self.human_skel.set_positions([0.0, 0.0, 0.0, -0.09486478804170062, 0.16919563098552753, -0.4913244737893412, -1.371164742525659, -0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, -0.09486478804170062, 0.16919563098552753, 0.4913244737893412, -1.371164742525659, 0.1465004046206566, 0.3062212857520513, 0.18862771696450964, 0.4970038523987025, 0.48155552859527917, -0.13660824713013747, 0.6881130165905589])
            #self.humanSPDIntperolationTarget = np.array(self.human_skel.q)

        human_pose = np.array(self.human_skel.q)
        human_pose[8] = 2.4
        human_pose[16] = 2.4
        self.human_skel.set_positions(human_pose)
        self.humanSPDIntperolationTarget = human_pose

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

        #initialize the robot poses
        # pick p0 for both robots with rejection sampling
        handleMaxDistance = 0.55
        handleMinXDistance = 0.4
        handleMaxYDistance = 0.1
        handleMaxZDistance = 0.1
        diskRad = 0.7
        good = False
        r_pivotL = self.iiwas[0].skel.bodynodes[3].to_world(np.zeros(3))
        r_pivotR = self.iiwas[1].skel.bodynodes[3].to_world(np.zeros(3))
        p0L = r_pivotL + pyutils.sampleDirections(num=1)[0] * diskRad
        p0R = r_pivotR + pyutils.sampleDirections(num=1)[0] * diskRad
        depth_offset = 0.15
        while (not good):
            good = True
            # xz0 = np.array([p0[0], p0[2]])

            # cut off points too close or behind the robot in x (side)
            if p0L[0] > (r_pivotL[0] - depth_offset):
                good = False

            if p0R[0] < (r_pivotR[0] + depth_offset):
                good = False

            # cut off points too close or behind the robots in z
            if p0L[2] > (r_pivotL[2] - depth_offset) or p0R[2] > (r_pivotR[2] - depth_offset):
                good = False

            if np.linalg.norm(p0L - p0R) > handleMaxDistance:
                good = False

            if abs(p0L[0] - p0R[0]) < handleMinXDistance:
                good = False

            if abs(p0L[1] - p0R[1]) > handleMaxYDistance:
                good = False

            if abs(p0L[2] - p0R[2]) > handleMaxZDistance:
                good = False



            if not good:
                p0L = r_pivotL + pyutils.sampleDirections(num=1)[0] * diskRad
                p0R = r_pivotR + pyutils.sampleDirections(num=1)[0] * diskRad

        #TODO:may not want to start level in this case...
        l45 = np.array([-0.5, -0.5, 0])
        l45 /= np.linalg.norm(l45)
        r45 = np.array([0.5, -0.5, 0])
        r45 /= np.linalg.norm(r45)
        self.iiwas[0].ik_target.setFromDirectionandUp(dir=l45, up=np.array([0, 0, 1.0]), org=p0L)
        self.iiwas[1].ik_target.setFromDirectionandUp(dir=r45, up=np.array([0, 0, 1.0]), org=p0R)
        self.iiwas[0].computeIK(maxIter=300)
        self.iiwas[1].computeIK(maxIter=300)
        self.iiwas[0].skel.set_velocities(np.zeros(len(self.iiwas[0].skel.dq)))
        self.iiwas[0].setIKPose() #frame set in here too
        self.iiwas[1].skel.set_velocities(np.zeros(len(self.iiwas[0].skel.dq)))
        self.iiwas[1].setIKPose() #frame set in here too


        #initialize the garment location
        #self.clothScene.translateCloth(0, np.array([0, 0, 1.0]))
        #self.clothScene.rotateCloth(cid=0, R=pyutils.rotateY(math.pi))
        #self.clothScene.rotateCloth(cid=0, R=pyutils.rotateZ(math.pi - 0.2))

        for iiwa in self.iiwas:
            hn = iiwa.skel.bodynodes[iiwa.handle_bodynode]

            iiwa.handle_node.setOrgToCentroid()
            iiwa.handle_node.setOrientation(R=hn.T[:3, :3])

            iiwa.handle_node.recomputeOffsets()
            iiwa.handle_node.updatePrevConstraintPositions()


        handleCentroid = (self.iiwas[0].handle_node.org + self.iiwas[1].handle_node.org) / 2.0

        # first translate the handleCentroid to the origin
        self.clothScene.translateCloth(0, -1.0 * handleCentroid)
        # then rotate the cloth about that point

        # now translate to desired location
        self.clothScene.translateCloth(0, (p0L + p0R) / 2.0)

        # reset the handle positions
        for iiwa in self.iiwas:
            iiwa.handle_node.setOrgToCentroid()
            iiwa.handle_node.recomputeOffsets()
            iiwa.handle_node.updatePrevConstraintPositions()


        # ensure feature normal directions
        cloth_centroid = self.clothScene.getVertexCentroid(cid=0)
        self.sleeveLSeamFeature.fitPlane()
        from_centroid = self.sleeveLSeamFeature.plane.org - cloth_centroid
        from_centroid = from_centroid / np.linalg.norm(from_centroid)
        self.sleeveLSeamFeature.fitPlane(normhint=from_centroid)

        self.sleeveRSeamFeature.fitPlane()
        from_centroid = self.sleeveRSeamFeature.plane.org - cloth_centroid
        from_centroid = from_centroid / np.linalg.norm(from_centroid)
        self.sleeveRSeamFeature.fitPlane(normhint=from_centroid)

        self.collarFeature.fitPlane()
        from_centroid = self.collarFeature.plane.org - cloth_centroid
        from_centroid = from_centroid / np.linalg.norm(from_centroid)
        self.collarFeature.fitPlane(normhint=from_centroid)

        self.waistFeature.fitPlane()
        from_centroid = self.waistFeature.plane.org - cloth_centroid
        from_centroid = from_centroid / np.linalg.norm(from_centroid)
        self.waistFeature.fitPlane(normhint=-from_centroid)

        #using the expected normal of the plane, so do this here only once
        if self.reset_number == 0:
            for ix,sm in enumerate(self.separated_meshes):
                sm.initSeparatedMeshGraph()
                sm.updateWeights()
                border_skips = self.collar_border_vertices.copy()
                if ix == 0 or ix == 2:
                    border_skips.extend(self.sleeve_R_border_vertices)
                if ix == 1 or ix == 2:
                    border_skips.extend(self.sleeve_L_border_vertices)

                if ix == 3:
                    sm.computeGeodesic(feature=self.cloth_features[ix], oneSided=True, side=0, normalSide=1)
                else:
                    sm.computeGeodesic(feature=self.cloth_features[ix], oneSided=True, side=0, normalSide=0, border_skip_vertices=border_skips)

        # now simulate the cloth while interpolating the handle node positions
        self.clothScene.clearInterpolation()
        simFrames = 50
        hL_init = np.array(self.iiwas[0].handle_node.org)
        hR_init = np.array(self.iiwas[1].handle_node.org)
        for frame in range(simFrames):
            self.iiwas[0].handle_node.org = hL_init + (p0L - hL_init) * (frame / (simFrames - 1))
            self.iiwas[1].handle_node.org = hR_init + (p0R - hR_init) * (frame / (simFrames - 1))
            self.iiwas[0].handle_node.updateHandles()
            self.iiwas[1].handle_node.updateHandles()
            self.clothScene.step()
            #for vix in range(self.clothScene.getNumVertices()):
            #    self.clothScene.setPrevVertexPos(cid=0, vid=vix,pos=self.clothScene.getVertexPos(cid=0,vid=vix))
            #self.clothScene.clearInterpolation()
            #self.text_queue.append("Cloth Settle Step " + str(frame))
            #self.render()
            for feature in self.cloth_features: #ensure these are facing the right way every time
                feature.fitPlane()
        #self.clothScene.saveObjState(filename="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/twoArmTshirtHang")
        #self.clothScene.loadObjState()

    def _getFile(self):
        return __file__