from gym.envs.dart.dart_cloth_iiwa_env import *

class DartClothIiwaTwoarmEnv(DartClothIiwaEnv):
    def __init__(self):

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
        self.iiwa_root_dofs.append(np.array([-1.2, -1.2, -1.2, self.dFromRoot * math.sin(-self.aFrom_invZ), -0.2, -self.dFromRoot * math.cos(-self.aFrom_invZ)]))

        #initialize the base env
        cloth_mesh_file = "fullgown1.obj"
        #cloth_mesh_state_file = "hanginggown.obj"
        cloth_mesh_state_file = "fullgown1.obj"
        DartClothIiwaEnv.__init__(self, robot_root_dofs=self.iiwa_root_dofs, active_compliance=False, cloth_mesh_file=cloth_mesh_file, cloth_mesh_state_file=cloth_mesh_state_file, cloth_scale=1.3)

        #manual control target


        #setup features
        self.sleeveRVerts = [532, 451, 251, 252, 253, 1334, 1320, 1184, 945, 985, 1062, 1607, 1037, 484, 1389, 679, 1230, 736, 1401, 1155, 486, 1410]
        self.sleeveLVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.cloth_features.append(self.sleeveLSeamFeature)
        self.cloth_features.append(self.sleeveRSeamFeature)

        #setup separated meshes
        # setup specific separated meshes and dressing targets for this task
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # left sleeve
        self.separated_meshes.append(meshgraph.MeshGraph(clothscene=self.clothScene))  # right sleeve

        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveLSeamFeature, limb_sequence=self.limbNodesL, distal_offset=self.fingertip))
        self.dressing_targets.append(DressingTarget(env=self, skel=self.human_skel, feature=self.sleeveRSeamFeature, limb_sequence=self.limbNodesR, distal_offset=self.fingertip))

        #setup handle nodes
        self.iiwas[0].addClothHandle(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131], offset=np.array([0, 0, 0.05]))
        self.iiwas[1].addClothHandle(verts=[1502, 808, 554, 1149, 819, 1619, 674, 1918, 1528, 1654, 484, 1590, 1802, 1924], offset=np.array([0, 0, 0.05]))

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
        rest_pose_weights[:2] *= 10 #stable torso
        rest_pose_weights[3] *= 1 #spine
        #rest_pose_weights[3:11] *= 0 #ignore active arm
        #rest_pose_weights[11:19] *= 2 #passive arm
        rest_pose_weights[3:19] *= 0 #ignore rest pose
        rest_pose_weights[19:] *= 2 #stable head
        self.reward_manager.addTerm(term=RestPoseRewardTerm(self.human_skel, pose=np.zeros(self.human_skel.ndofs), weights=rest_pose_weights))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[0], weight=40))
        self.reward_manager.addTerm(term=LimbProgressRewardTerm(dressing_target=self.dressing_targets[1], weight=40))
        self.reward_manager.addTerm(term=ClothDeformationRewardTerm(self, weight=1))
        self.reward_manager.addTerm(term=HumanContactRewardTerm(self, weight=10))

        #set the observation space
        self.obs_dim = self.human_obs_manager.obs_size
        if self.dualPolicy:
            self.obs_dim += self.robot_obs_manager.obs_size
        elif not self.isHuman:
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

        #initialize the robot poses
        # pick p0 for both robots with rejection sampling
        handleMaxDistance = 0.5
        handleMinXDistance = 0.1
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

            if not good:
                p0L = r_pivotL + pyutils.sampleDirections(num=1)[0] * diskRad
                p0R = r_pivotR + pyutils.sampleDirections(num=1)[0] * diskRad

        self.iiwas[0].ik_target.setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=p0L)
        self.iiwas[1].ik_target.setFromDirectionandUp(dir=np.array([0, -1.0, 0]), up=np.array([0, 0, 1.0]), org=p0R)
        self.iiwas[0].computeIK(maxIter=300)
        self.iiwas[1].computeIK(maxIter=300)
        self.iiwas[0].skel.set_velocities(np.zeros(len(self.iiwas[0].skel.dq)))
        self.iiwas[0].setIKPose() #frame set in here too
        self.iiwas[1].skel.set_velocities(np.zeros(len(self.iiwas[0].skel.dq)))
        self.iiwas[1].setIKPose() #frame set in here too

        #initialize the garment location
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
        self.clothScene.rotateCloth(cid=0, R=pyutils.rotateZ(math.pi))
        self.clothScene.rotateCloth(cid=0, R=pyutils.rotateY(math.pi))

        # now translate to desired location
        self.clothScene.translateCloth(0, (p0L + p0R) / 2.0)

        # reset the handle positions
        for iiwa in self.iiwas:
            iiwa.handle_node.setOrgToCentroid()
            iiwa.handle_node.recomputeOffsets()
            iiwa.handle_node.updatePrevConstraintPositions()

        # now simulate the cloth while interpolating the handle node positions
        self.clothScene.clearInterpolation()
        simFrames = 100
        hL_init = np.array(self.iiwas[0].handle_node.org)
        hR_init = np.array(self.iiwas[1].handle_node.org)
        for frame in range(simFrames):
            self.iiwas[0].handle_node.org = hL_init + (p0L - hL_init) * (frame / (simFrames - 1))
            self.iiwas[1].handle_node.org = hR_init + (p0R - hR_init) * (frame / (simFrames - 1))
            self.iiwas[0].handle_node.updateHandles()
            self.iiwas[1].handle_node.updateHandles()
            self.clothScene.step()

        #ensure feature normal directions
        self.sleeveLSeamFeature.fitPlane()
        from_origin = self.sleeveLSeamFeature.plane.org
        from_origin = from_origin / np.linalg.norm(from_origin)
        self.sleeveLSeamFeature.fitPlane(normhint=from_origin)

        self.sleeveRSeamFeature.fitPlane()
        from_origin = self.sleeveRSeamFeature.plane.org
        from_origin = from_origin / np.linalg.norm(from_origin)
        self.sleeveRSeamFeature.fitPlane(normhint=from_origin)

        #using the expected normal of the plane, so do this here only once
        if self.reset_number == 0:
            self.separated_meshes[0].initSeparatedMeshGraph()
            self.separated_meshes[0].updateWeights()
            self.separated_meshes[0].computeGeodesic(feature=self.sleeveLSeamFeature, oneSided=True, side=0, normalSide=1)
            vertex_blacklist = [2055]
            self.separated_meshes[1].initSeparatedMeshGraph()
            self.separated_meshes[1].updateWeights()
            self.separated_meshes[1].computeGeodesic(feature=self.sleeveRSeamFeature, oneSided=True, side=0, normalSide=1, boundary_skip_vertices=vertex_blacklist)

    def _getFile(self):
        return __file__