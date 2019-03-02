from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

register(
    id='KellyCoinflip-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,
)
register(
    id='KellyCoinflipGeneralized-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    max_episode_steps=100,
)

register(
    id='Taxi-v2',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)

register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    max_episode_steps=200,
)

register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    max_episode_steps=200,
)

# Dart
# ----------------------------------------
register(
    id='DartHopper-v1',
    entry_point='gym.envs.dart:DartHopperEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartFullbody-v1',
    entry_point='gym.envs.dart:DartFullbodyEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

'''register(
    id='DartHopperRBF-v1',
    entry_point='gym.envs.dart:DartHopperRBFEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)'''

'''register(
    id='DartHopperCont-v1',
    entry_point='gym.envs.dart:DartHopperEnvCont',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)'''

register(
    id='DartCartPole-v1',
    entry_point='gym.envs.dart:DartCartPoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartDoubleInvertedPendulumEnv-v1',
    entry_point='gym.envs.dart:DartDoubleInvertedPendulumEnv',
    max_episode_steps=1000,
)

register(
    id='DartReacher-v1',
    entry_point='gym.envs.dart:DartReacher2dEnv',
    reward_threshold=-3.75,
    max_episode_steps=50,
)

register(
    id='DartReacher3d-v1',
    entry_point='gym.envs.dart:DartReacherEnv',
    reward_threshold=-200,
    max_episode_steps=500,
)

register(
    id='DartRobotWalk-v1',
    entry_point='gym.envs.dart:DartRobotWalk',
    max_episode_steps=1000,
)

register(
    id='DartDog-v1',
    entry_point='gym.envs.dart:DartDogEnv',
    max_episode_steps=1000,
)

'''register(
    id='DartSwimmer-v1',
    entry_point='gym.envs.dart:DartSwimmerEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)'''

register(
    id='DartCartPole-v0',
    entry_point='gym.envs.dart:DartCartPoleImgEnv',
    reward_threshold=950.0,
    max_episode_steps=2000,
)

register(
    id='DartWalker2d-v1',
    entry_point='gym.envs.dart:DartWalker2dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3d-v1',
    entry_point='gym.envs.dart:DartWalker3dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3dSPD-v1',
    entry_point='gym.envs.dart:DartWalker3dSPDEnv',
    max_episode_steps=1000,
)

'''register(
    id='DartSwing-v0',
    entry_point='gym.envs.dart:DartSwingEnv',
    reward_threshold=950.0,
    max_episode_steps=2000,
)'''

# Dart Cloth
# ----------------------------------------
register(
    id='DartClothSphereTube-v1',
    entry_point='gym.envs.dart:DartClothSphereTubeEnv',
    max_episode_steps=500,
)

register(
    id='DartClothReacher-v1',
    entry_point='gym.envs.dart:DartClothReacherEnv',
    max_episode_steps=500,
)

register(
    id='DartClothReacher-v2',
    entry_point='gym.envs.dart:DartClothReacherEnv2',
    max_episode_steps=500,
)

register(
    id='DartClothReacher-v3',
    entry_point='gym.envs.dart:DartClothReacherEnv3',
    max_episode_steps=500,
)

register(
    id='DartClothPoseReacher-v1',
    entry_point='gym.envs.dart:DartClothPoseReacherEnv',
    max_episode_steps=500,
)

register(
    id='DartClothSleeveReacher-v1',
    entry_point='gym.envs.dart:DartClothSleeveReacherEnv',
    max_episode_steps=500,
)

register(
    id='DartClothShirtReacher-v1',
    entry_point='gym.envs.dart:DartClothShirtReacherEnv',
    max_episode_steps=500,
)

register(
    id='DartClothTestbed-v1',
    entry_point='gym.envs.dart:DartClothTestbedEnv',
    max_episode_steps=10000,
)

register(
    id='DartClothGownDemo-v1',
    entry_point='gym.envs.dart:DartClothGownDemoEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothGrippedTshirt-v1',
    entry_point='gym.envs.dart:DartClothGrippedTshirtEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothGrippedTshirt-v2',
    entry_point='gym.envs.dart:DartClothGrippedTshirtSplineEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothGrippedTshirt-v3',
    entry_point='gym.envs.dart:DartClothGrippedTshirtSpline2ndArmEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothEndEffectorDisplacer-v1',
    entry_point='gym.envs.dart:DartClothEndEffectorDisplacerEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothJointLimitsTest-v1',
    entry_point='gym.envs.dart:DartClothJointLimitsTestEnv',
    #max_episode_steps=250,
)

# Upper Body Data Driven Envs
#---------------------------------------------------

register(
    id='DartClothUpperBodyDataDriven-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenTshirt-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenTshirtEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenTshirt-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothTshirtREnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenTshirt-v3',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothTshirtLEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenTshirtL_HapticHighRes-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothTshirtLHapticHighResEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenReacher-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothReacherEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenDropGrip-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothDropGripEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolate-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolateEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolate-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolate2Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolate-v3',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolate3Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolate-v4',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolate4Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenTshirtMaster-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothTshirtMasterEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenJacket-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothJacketREnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenJacket-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothJacketLEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolateJacket-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolateJacket1Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenPhaseInterpolateJacket-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothPhaseInterpolateJacket2Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenJacketMaster-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothJacketMasterEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenLinearTrack-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothLinearTrack1Env',
    #max_episode_steps=250,
)

register(
    id='DartClothUpperBodyDataDrivenAssistLinearTrack-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothAssistLinearTrack1Env',
    #max_episode_steps=250,
)

# Full Body Data Driven Envs
#---------------------------------------------------

register(
    id='DartClothFullBodyDataDrivenClothTest-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothTestEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothSPDTest-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothSPDTestEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothStand-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothStandEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothOneFootStand-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothOneFootStandEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothOneFootStandCrouch-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothOneFootStandCrouchEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothOneFootStandShorts-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothOneFootStandShortsEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothOneFootStandShorts-v2',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothOneFootStandShorts2Env',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenClothOneFootStandShorts-v3',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenClothOneFootStandShorts3Env',
    #max_episode_steps=250,
)

#locked foot envs
register(
    id='DartClothFullBodyDataDrivenLockedFootClothTest-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenLockedFootClothTestEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenLockedFootClothBalance-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenLockedFootClothBalanceEnv',
    #max_episode_steps=250,
)

register(
    id='DartClothFullBodyDataDrivenLockedFootClothShortsAlign-v1',
    entry_point='gym.envs.dart:DartClothFullBodyDataDrivenLockedFootClothShortsAlignEnv',
    #max_episode_steps=250,
)


#Multi-agent
register(
    id='DartMultiAgent-v1',
    entry_point='gym.envs.dart:DartMultiAgentEnv',
    #max_episode_steps=250,
)

register(
    id='DartFlockAgent-v1',
    entry_point='gym.envs.dart:DartFlockingEnv',
    #max_episode_steps=250,
)

#Sawyer IK Env
register(
    id='DartSawyer-v1',
    entry_point='gym.envs.dart:DartSawyerEnv',
    #max_episode_steps=250,
)

#Sawyer SPD to IK Env
register(
    id='DartSawyer-v2',
    entry_point='gym.envs.dart:DartSawyerSPDTrackingEnv',
    #max_episode_steps=250,
)

#Sawyer SPD to IK Env with rigid cloth approx.
register(
    id='DartSawyer-v3',
    entry_point='gym.envs.dart:DartSawyerSPDTrackingRigidEnv',
    #max_episode_steps=250,
)

#Sawyer Pure Rigid Env
register(
    id='DartSawyerRigid-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothSawyerEnv',
    #max_episode_steps=250,
)

#Sawyer Hanging Rigid Env
register(
    id='DartSawyerRigid-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothHangSawyerEnv',
    #max_episode_steps=250,
)

#Sawyer Hanging Rigid Env (constructed)
register(
    id='DartSawyerRigid-v3',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothConstructSawyerEnv',
    #max_episode_steps=250,
)

#Sawyer Hanging Rigid Env (one file)
register(
    id='DartSawyerRigid-v4',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothOneFileSawyerEnv',
    #max_episode_steps=250,
)

#Iiwa Capability Env
register(
    id='DartIiwaCapability-v1',
    entry_point='gym.envs.dart:DartIiwaSPDCapabilityEnv',
    #max_episode_steps=250,
)

#Iiwa Bezier robot behavior space Env
register(
    id='DartIiwaBezier-v1',
    entry_point='gym.envs.dart:DartIiwaSPDBezierEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Rigid Env (one file)
register(
    id='DartIiwaRigid-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothOneFileIiwaEnv',
    #max_episode_steps=250,
)

#Sawyer Hanging Rigid Env (one file)
register(
    id='DartSawyerRigidAssist-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenRigidClothSawyerAssistEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Env
register(
    id='DartIiwaGown-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Env v2(faster cloth)
register(
    id='DartIiwaGown-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownEnvV2',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Env v3(moved robot and SPD human)
register(
    id='DartIiwaGown-v3',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownEnvV3',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Env v4(moved robot and SPD human with interpolated target...)
register(
    id='DartIiwaGown-v4',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownEnvV4',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Env v5(SPD human with adjusted gains, interpolation, and damping + multiple robot control modes )
register(
    id='DartIiwaGown-v5',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownEnvV5',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Env
register(
    id='DartIiwaGownAssist-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Env 2 (adjusted timestep, etc...)
register(
    id='DartIiwaGownAssist-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistEnvV2',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Env 3 (new output space: local Euler offsets at EF frame)
register(
    id='DartIiwaGownAssist-v3',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistEnvV3',
    #max_episode_steps=250,
)
#Iiwa Hanging Gown Assist Env 4: updated to synchronize with gown env V5
register(
    id='DartIiwaGownAssist-v4',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistEnvV4',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Coopt Env
register(
    id='DartIiwaGownAssistCoopt-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistCooptEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Coopt HELPER Env (duplicate of above with one flag set: HACK)
register(
    id='DartIiwaGownAssistCoopt_h-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistCooptHelperEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Coopt Env V2 (SPD human)
register(
    id='DartIiwaGownAssistCoopt-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistCooptV2Env',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Assist Coopt HELPER Env v2 (SPD human) (duplicate of above with one flag set and name changed: HACK)
register(
    id='DartIiwaGownAssistCoopt_h-v2',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownAssistCooptV2HelperEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Multibot Coopt Env V1 (SPD human)
register(
    id='DartIiwaGownMultibot-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownMultibotEnv',
    #max_episode_steps=250,
)

#Iiwa Hanging Gown Multibot Coopt HELPER Env V1 (SPD human)
register(
    id='DartIiwaGownMultibot_h-v1',
    entry_point='gym.envs.dart:DartClothUpperBodyDataDrivenClothIiwaGownMultibotHelperEnv',
    #max_episode_steps=250,
)

register(
    id='DartIiwaOnearmGown-v1',
    entry_point='gym.envs.dart:DartClothIiwaOnearmEnv',
    #max_episode_steps=250,
)

register(
    id='DartIiwaTwoarmGown-v1',
    entry_point='gym.envs.dart:DartClothIiwaTwoarmEnv',
    #max_episode_steps=250,
)

register(
    id='DartTrialTest-v1',
    entry_point='gym.envs.dart:DartClothExperimentTestingEnv',
    #max_episode_steps=250,
)

#Updated Iiwa envs:


# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Pusher-v0',
    entry_point='gym.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v0',
    entry_point='gym.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v0',
    entry_point='gym.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v1',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v1',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco:Walker2dEnv',
)

register(
    id='Ant-v1',
    entry_point='gym.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v1',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v1',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

# Board games
# ----------------------------------------

register(
    id='Go9x9-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
    # The pachi player seems not to be determistic given a fixed seed.
    # (Reproduce by running 'import gym; h = gym.make('Go9x9-v0'); h.seed(1); h.reset(); h.step(15); h.step(16); h.step(17)' a few times.)
    #
    # This is probably due to a computation time limit.
    nondeterministic=True,
)

register(
    id='Go19x19-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Hex9x9-v0',
    entry_point='gym.envs.board_game:HexEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
)

# Debugging
# ----------------------------------------

register(
    id='OneRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='OneRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundNondeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundNondeterministicRewardEnv',
    local_only=True,
)

# Parameter tuning
# ----------------------------------------
register(
    id='ConvergenceControl-v0',
    entry_point='gym.envs.parameter_tuning:ConvergenceControl',
)

register(
    id='CNNClassifierTraining-v0',
    entry_point='gym.envs.parameter_tuning:CNNClassifierTraining',
)

# Safety
# ----------------------------------------

# interpretability envs
register(
    id='PredictActionsCartpole-v0',
    entry_point='gym.envs.safety:PredictActionsCartpoleEnv',
    max_episode_steps=200,
)

register(
    id='PredictObsCartpole-v0',
    entry_point='gym.envs.safety:PredictObsCartpoleEnv',
    max_episode_steps=200,
)

# semi_supervised envs
    # probably the easiest:
register(
    id='SemisuperPendulumNoise-v0',
    entry_point='gym.envs.safety:SemisuperPendulumNoiseEnv',
    max_episode_steps=200,
)
    # somewhat harder because of higher variance:
register(
    id='SemisuperPendulumRandom-v0',
    entry_point='gym.envs.safety:SemisuperPendulumRandomEnv',
    max_episode_steps=200,
)
    # probably the hardest because you only get a constant number of rewards in total:
register(
    id='SemisuperPendulumDecay-v0',
    entry_point='gym.envs.safety:SemisuperPendulumDecayEnv',
    max_episode_steps=200,
)

# off_switch envs
register(
    id='OffSwitchCartpole-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleEnv',
    max_episode_steps=200,
)

register(
    id='OffSwitchCartpoleProb-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleProbEnv',
    max_episode_steps=200,
)
