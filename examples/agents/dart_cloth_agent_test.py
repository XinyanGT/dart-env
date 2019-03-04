__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib

import pyPhysX.pyutils as pyutils
import os
import sys
import shutil, errno

from rllab import spaces
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from SplitGMLP import SplitGaussianMLPPolicy
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import lasagne.layers as L

def renderGraph(filename, targetField="AverageDiscountedReturn"):
    data = []
    f = open(filename, 'r')
    col = 0
    for ix, line in enumerate(f):
        words = line.split(",")
        if ix == 0:
            for ix2, i in enumerate(words):
                if i.strip() == targetField: #strip to remove trailing newlines at end of csv file
                    col = ix2
        else:
            data.append(float(words[col]))

    f.close()
    graph = pyutils.LineGrapher(title=targetField)
    graph.plotData(data)
    graph.plotData(np.zeros(len(graph.xdata)),color=(0,0,0))
    graph.update()
    filenameParts = filename.split("/")
    dirName = filename[:-len(filenameParts[-1])]
    print("saving graph to " + dirName)
    graph.save(filename=dirName+targetField)

    return data

def main():

    filename = None
    filename2 = None

    this_prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(this_prefix, '../../../rllab/data/local/experiment/')
    env_prefix = os.path.join(this_prefix, '../../gym/envs/dart/')

    trial = None

    # --- new multiarm
    #trial = "experiment_2019_03_02_twoarm_nocompliance_terminal" #TODO
    #trial = "experiment_2019_03_02_twoarm_nocompliance" #TODO
    #trial = "experiment_2019_03_02_twoarm_standard" #TODO
    # ---

    # -- new env onearm
    #trial = "experiment_2019_03_02_onearm_weak_nocompliance_strconpen_terminal"
    #trial = "experiment_2019_03_02_onearm_nocompliance_strconpen_terminal"
    #trial = "experiment_2019_03_02_onearm_warm_nocompliance_4xstrconpen_terminal_MPC"
    #trial = "experiment_2019_03_01_onearm_weaker_strconpen_128"
    #trial = "experiment_2019_03_01_onearm_weaker_strconpen_noactiverest"
    #trial = "experiment_2019_03_01_onearm_weaker_strconpen"
    #trial = "experiment_2019_03_01_onearm_standard_strconpen_datadrivenobs"
    #trial = "experiment_2019_03_01_onearm_standard_strconpen_nonactiverest"
    #trial = "experiment_2019_03_01_onearm_standard_noactivecompliance_strconpen_MPCobs"
    #trial = "experiment_2019_02_27_onearm_weak_collpen_MPC"
    #trial = "experiment_2019_02_27_onearm_weak_128"
    #trial = "experiment_2019_02_27_onearm_weak"
    #trial = "experiment_2019_02_27_onearm_standard"
    #trial = "experiment_2019_02_27_onearm_collpen_MPC"
    #trial = "experiment_2019_02_27_onearm_standard_noactivecompliance"
    # ---

    # --- Split Network: one arm
    #trial = "experiment_2019_02_20_split_coopt_onearm_noconpen_cap_nodef"
    #trial = "experiment_2019_02_19_split_architecture_coopt_noconpen_humreward"
    # ---

    # --- Multibot trials
    #trial = "experiment_2019_02_19_split_architecture_coopt_multibot_noconpen_humreward" #dual network, human reward
    #trial = "experiment_2019_02_19_split_architecture_coopt_multibot_noconpen" #dual network, no contact penalty
    #trial = "experiment_2019_02_18_split_architecture_coopt_multibot" #dual network
    #trial = "experiment_2019_02_15_robo_multibot_cap"
    #trial = "experiment_2019_02_15_robo_multibot"
    #trial = "experiment_2019_02_14_coopt_multibot_cont2" #Not optimized correctly (each fixed at previous end...)
    #trial = "experiment_2019_02_14_coopt_multibot_cont" #Not optimized correctly (each fixed at previous end...)
    #trial = "experiment_2019_02_14_coopt_multibot" #TODO #NOTE: not complete...
    #trial = "experiment_2019_02_13_human_multibot_weak"
    #trial = "experiment_2019_02_13_human_multibot"
    # ---

    # --- Robot trained on tuned SPD human
    #trial = "experiment_2019_02_20_robo_weakerrange_noconpen_cap_weakobs_tar_nodef" #does NOT have weakness obs (robot)
    #trial = "experiment_2019_02_19_robo_weakerrange_noconpen_cap_weakobs_tar"
    #trial = "experiment_2019_02_15_robo_SPD_human_norest_weakerrange_cap_weakobs"
    #trial = "experiment_2019_02_15_robo_SPD_human_norest_weakerrange_obs"
    #trial = "experiment_2019_02_15_robo_SPD_human_norest_weakerrange_cap"
    #trial = "experiment_2019_02_14_robo_SPD_human_norest_weakerrange"
    #trial = "experiment_2019_02_13_robo_SPD_human_norest_weakrange"
    #trial = "experiment_2019_02_11_robo_SPD_human_weaker_newcapsensor"
    #trial = "experiment_2019_02_10_robo_SPD_human_noisytarget_newcapsensor"
    #trial = "experiment_2019_02_10_robo_SPD_human_norest_newcapsensor"
    #trial = "experiment_2019_02_10_robo_SPD_human_hoverproceed_norest"
    #trial = "experiment_2019_02_10_robo_SPD_human_hoverproceed_norest_noisyhuman"
    # ---

    # --- tuned SPD human (fixed)
    #trial = "experiment_2019_02_11_SPD_human_weak_restpose" #restpose of one arm is randomized and included as input
    #trial = "experiment_2019_02_11_SPD_human_restpose" #restpose of one arm is randomized and included as input
    #trial = "experiment_2019_02_10_SPD_human_norest_weaker"
    #trial = "experiment_2019_02_10_SPD_human_hoverproceedavoidbot_weak_conpen_norest"
    #trial = "experiment_2019_02_10_SPD_human_hoverproceedavoidbot_conpen_norest"
    #trial = "experiment_2019_02_09_SPD_human_hoverproceedbot_conpen"
    #trial = "experiment_2019_02_09_SPD_human_hoverproceedbot_weak_conpen"
    #trial = "experiment_2019_02_09_SPD_human_hoverproceedbot_conpen_norest"
    #trial = "experiment_2019_02_09_SPD_human_hoverproceedbot"
    #trial = "experiment_2019_02_09_coopt_warmhuman_noconpen"
    #trial = "experiment_2019_02_09_coopt_warmhuman"
    # --- tuned SPD human (buggy)
    #trial = "experiment_2019_02_08_coopt_test"
    #trial = "experiment_2019_02_08_SPD_human_hoverbot_weakness"
    #trial = "experiment_2019_02_08_SPD_human_hoverbot"
    # ---

    # --- Trial block with moved arm and SPD human
    #trial = "experiment_2019_02_06_SPD_human_elbowlimits_weakness"
    #trial = "experiment_2019_02_06_SPD_human_elbowlimits"
    #trial = "experiment_2019_02_06_SPD_human_weakness"
    #trial = "experiment_2019_02_06_SPD_human_capable"
    #trial = "experiment_2019_02_06_SPD_human_apen_lowexp"
    #trial = "experiment_2019_02_06_SPD_human_lowexp"
    #trial = "experiment_2019_02_05_SPD_human"
    #trial = "experiment_2019_02_04_SPD_human"
    # ---

    #--- Trial block with moved arm
    #trial = "experiment_2019_02_06_spec_hard_nodef"
    #trial = "experiment_2019_02_05_robo_weakness_easyelbow_plane"
    #trial = "experiment_2019_02_05_robo_spec_hard_longHorizon_plane"   #unrendered
    #trial = "experiment_2019_02_05_robo_spec_hard_lowfriction_plane"
    #trial = "experiment_2019_02_05_robo_weakness_easyelbow" #unrendered
    #trial = "experiment_2019_02_05_robo_spec_hard_lowfriction"     #unrendered
    #trial = "experiment_2019_02_05_robo_spec_hard_longHorizon"     #unrendered
    #trial = "experiment_2019_02_03_robo_2D_w_e_u_einterp_lim_conPen_spec_hard_plane_cap2_75"
    #trial = "experiment_2019_02_03_robo_2D_w_e_u_einterp_lim_conPen_spec_hard_plane_cap2_75_prog"
    #trial = "experiment_2019_02_02_robo_2D_w_e_u_einterp_lim_conPen_full_variation_plane_cap"
    #trial = "experiment_2019_02_02_robo_2D_w_e_u_einterp_lim_conPen_spec_hard_plane_cap"
    #trial = "experiment_2019_02_01_robo_2D_w_e_u_einterp_lim_conPen_full_variation"
    #trial = "experiment_2019_02_01_robo_2D_w_e_u_einterp_lim_conPen_spec_hard"
    #trial = "experiment_2019_01_31_robo_2D_w_e_u_einterp_lim_conPen_spec_capobs"
    #trial = "experiment_2019_01_31_robo_2D_w_e_u_einterp_lim_conPen_spec"
    #---

    #trial = "experiment_2019_01_29_coopt_test"
    #trial = "experiment_2019_01_28_robo_2D_w_e_u_einterp_lim_conPen_conplane_spec_capobs_expprog_limbdir"

    #trial = "experiment_2019_01_27_robo_2D_weakness_elbow_universal_eulerinterp_lim_conPen_conplane_spec_capobs" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Contact penalty. straight-ish elbow, full capability, capacitive sensor, shoulder plane penalty
    #trial = "experiment_2019_01_26_robo_2D_weakness_elbow_universal_eulerinterp_lim_conPen_spec_capobs" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Contact penalty. straight-ish elbow, full capability, capacitive sensor
    #trial = "experiment_2019_01_26_robo_2D_weakness_elbow_universal_eulerinterp_lim_conPen_conplane_specHard" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Contact penalty. Bent elbow, full capability. plane penalty
    #trial = "experiment_2019_01_24_robo_2D_weakness_elbow_universal_eulerinterp_lim_conPen_spec" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Contact penalty. straight-ish elbow, full capability
    #trial = "experiment_2019_01_24_robo_2D_weakness_elbow_universal_eulerinterp_lim_conPen_specHard_cont" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Contact penalty. Bent elbow, full capability

    #trial = "experiment_2019_01_23_robo_2D_weakness_elbow_universal_eulerinterp_lim_splane_conPen" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Plane constraint. Contact penalty.
    #trial = "experiment_2019_01_02_robo_2D_weakness_elbow_universal_eulerinterp_lim_splane" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach. Plane constraint.
    #trial = "experiment_2018_12_23_robo_2D_weakness_elbow_universal_eulerinterp_lim" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control. Frame limited to near robot reach.
    #trial = "experiment_2018_12_19_robo_2D_weakness_elbow_universal_eulerinterp" #2D variation weakness and elbow constraint; V3; robot with full interpolation frame control
    #trial = "experiment_2018_12_13_robo_2D_weakness_elbow_universal_euler" #2D variation weakness and elbow constraint; V3; robot with full frame control
    #trial = "experiment_2018_12_10_robo_2D_weakness_elbow_universal" #2D variation weakness and elbow constraint; V2; robot
    #trial = "experiment_2018_11_27_weakness_and_elbow_universal_cont" #Continued: 2D variation weakness and elbow constraint; V2;
    #trial = "experiment_2018_11_27_weakness_and_elbow_universal" #2D variation weakness and elbow constraint; V2; p1(to be continued)
    #trial = "experiment_2018_11_20_robo_elbow_constraint13" #robo trained for "experiment_2018_11_13_elbow_constraint", fixed weakness 1.0: v2
    #trial = "experiment_2018_11_13_elbow_constraint" #fixed weakness 1.0: v2
    #trial = "experiment_2018_11_04_ablegown_cont" #fixed weakness 1.0: v2 (continued at ~1500 iterations)
    #trial = "experiment_2018_11_04_ablegown" #fixed weakness 1.0: v2
    #trial = "experiment_2018_10_25_robo_weakgown18" #robot iiwa policy for 18_weakgown human
    #trial = "experiment_2018_10_18_weakgown" #fixed oracle normal after limb progress > 0 (took ~2500 iterations)
    #trial = "experiment_2018_10_17_weakgown_warm" #warm start of 'experiment_2018_10_15_weakgown' with oracle correction
    #trial = "experiment_2018_10_15_weakgown" #action scale x5: first iiwa cloth gown

    #trial = "experiment_2018_10_02_weak" #action scale x2.5 with strong torso (kinda unstable)
    #trial = "experiment_2018_10_01_weak_x2d5" #action scale x2.5
    #trial = "experiment_2018_09_27_robo__20rhang_weak_simple_lowvar" #trained weakness = [0.05, 0.15]
    #trial = "experiment_2018_09_25_robo__20rhang_weakvar_simple" #(100k samples (not so good))
    #trial = "experiment_2018_09_23_robo__20rhang_weakvar_simple" #better
    #trial = "experiment_2018_09_21_robo__20rhang_weakvar_simple"
    #trial = "experiment_2018_10_03_weak" #x2.5 with strong torso
    #trial = "experiment_2018_09_20_rhang_weakvar_simple" #could be improved
    #trial = "experiment_2018_09_18_rhang_weakvar_simple"
    #trial = "experiment_2018_09_17_rhang_weakvar_simple"
    #trial = "experiment_2018_09_14_weakVar"
    #trial = "experiment_2018_09_13_weakVar" #interpolated behavior, average dominant
    #trial = "experiment_2018_09_010_sawyer_rigid_hang"
    #trial = "experiment_2018_09_08_sawyer_rigid_hang"

    #trial = "experiment_2018_09_06_sawyer_rigid_hang"
    #trial = "experiment_2018_09_03_sawyer_rigid_hang" #works well
    #trial = "experiment_2018_09_01_sawyer_rigid_hang_1" #bug: diagonal initial location


    #Sawyer Trials


    #trial = "experiment_2018_05_27_tshirtR_ablationhaptics"
    #trial = "experiment_2018_05_27_tshirtR_ablationgeo"
    #trial = "experiment_2018_05_26_tshirtR_ablaton_oracle"
    #trial = "experiment_2018_05_25_tshirtR_ablationbaseline_cont"
    #trial = "experiment_2018_05_25_tshirtR_ablationbaseline"

    #trial = "experiment_2018_05_25_assistivelinear1" #good

    #trial = "experiment_2018_05_24_assistivelineartrack" #bugged
    #trial = "experiment_2018_05_24_tshirt_final" #good

    #trial = "experiment_2018_05_23_lsleeve_warm"
    #trial = "experiment_2018_05_23_lineartrack" #good linear

    #trial = "experiment_2018_05_22_jacketL_warm_rest"
    #trial = "experiment_2018_05_23_ltuck_warm"         #good seq 3
    #trial = "experiment_2018_05_22_match_warm_rest"    #good seq 3

    #trial = "experiment_2018_05_22_tuckL"
    #trial = "experiment_2018_05_22_jacketL_warm"

    #trial = "experiment_2018_05_21_match_warm" #good enough? ~75%
    #trial = "experiment_2018_05_21_jacketL"
    #trial = "experiment_2018_05_21_rfootdown"

    # trial = "experiment_2018_05_21_jacketL_restpose"
    # trial = "experiment_2018_05_21_match_setup"
    # trial = "experiment_2018_05_21_jackettransition"

    #trial = "experiment_2018_05_20_matchgrip_pose2"
    #trial = "experiment_2018_05_20_rfootdown2"

    #trial = "experiment_2018_05_20_matchgrip_pose"
    #trial = "experiment_2018_05_20_jacketr" #good to go
    #trial = "experiment_2018_05_20_rfootdown"

    #trial = "experiment_2018_05_19_rfootdown2"
    #trial = "experiment_2018_05_19_matchgrip2"

    #trial = "experiment_2018_05_19_rfootdown"
    #trial = "experiment_2018_05_19_matchgrip"

    #trial = "experiment_2018_05_18_rfootdown2"
    #trial = "experiment_2018_05_18_matchgrip2"

    #trial = "experiment_2018_05_18_matchgrip"
    #trial = "experiment_2018_05_18_rfootdown"

    #trial = "experiment_2018_05_17_matchgrip"
    #trial = "experiment_2018_05_17_lockedL_shortsrlegdown2_cont"

    #trial = "experiment_2018_05_17_tuckL_elbow"
    #trial = "experiment_2018_05_17_lockedL_shortsrlegdown2"
    #trial = "experiment_2018_05_17_lockedL_shortsrlegdown"
    #trial = "experiment_2018_05_15_lockedL_shortsrleg_cont" #best yet
    #trial = "experiment_2018_05_16_lockedL_shortsrleg"

    #trial = "experiment_2018_05_15_lockedL_shortsrleg2"
    #trial = "experiment_2018_05_15_lockedL_shortsrleg" #alright
    #trial = "experiment_2018_05_15_lockedL_shortsalign"
    #trial = "experiment_2018_05_15_lockedL_shortsrleg"

    #trial = "experiment_2018_05_14_lockedL_shortsalign2"
    #trial = "experiment_2018_05_14_lockedL_shortsalign" #this does pretty well
    #trial = "experiment_2018_05_13_lockedL_shortsalign"
    #trial = "experiment_2018_05_12_lockedL_shortsalign3"
    #trial = "experiment_2018_05_12_lockedL_shortsalign2"
    #trial = "experiment_2018_05_12_lockedL_shortsalign"
    #trial = "experiment_2018_05_11_lockedL_balance"

    #trial = "experiment_2018_05_10_stand_SPD_prevtau"
    #trial = "experiment_2018_05_10_stand_SPD_lowvel"
    #trial = "experiment_2018_05_09_lsleeve2_wide_warmhighdef" #high def correction of "experiment_2018_05_06_lsleeve2_wide"
    #trial = "experiment_2018_05_09_stand_SPD"  # first SPD trial (local features)

    #trial = "experiment_2018_05_09_stand_lowbias" # Reduced bias TRPO (local features)
    #trial = "experiment_2018_05_09_stand"  # typical TRPO (local features)

    #trial = "experiment_2018_05_06_lsleeve2_warm"
    #trial = "experiment_2018_05_06_lsleeve2_wide"

    #trial = "experiment_2018_05_04_ltuck_403"
    #trial = "experiment_2018_05_04_onefoot_shorts_align"

    #trial = "experiment_2018_05_02_onefootstandcrouch_simple"
    #trial = "experiment_2018_05_02_sleevel_targeted_cont"
    #trial = "experiment_2018_05_03_onefootstandcrouch_stable"
    #trial = "experiment_2018_05_01_sleevel_widewarm"

    #trial = "experiment_2018_05_02_onefootstandcrouch"
    #trial = "experiment_2018_05_02_sleevel_targeted"

    #trial = "experiment_2018_05_01_sleevel_narrow"
    #trial = "experiment_2018_05_01_onefootstand_crouch"

    #trial = "experiment_2018_04_30_sleeveL_wide_highdef"
    #trial = "experiment_2018_04_30_onefootstand_crouch"

    #trial = "experiment_2018_04_26_lsleeve_wide_warm_cont"
    #trial = "experiment_2018_04_27_sleeveL_narrow_haptichighres"
    #trial = "experiment_2018_04_27_onefootstand_warm"

    #trial = "experiment_2018_04_26_onefootstand"
    #trial = "experiment_2018_04_26_stand"
    #trial = "experiment_2018_04_26_lsleeve_wide_warm"

    #trial = "experiment_2018_04_25_lsleeve_narrow2_warm_cont"

    #trial = "experiment_2018_04_24_lsleeve_narrow"
    #trial = "experiment_2018_04_25_lsleeve_narrow2_warm"
    #trial = "experiment_2018_04_25_stand"

    #trial = "experiment_2018_04_24_lsleeve_narrow"

    #neither entering sleeve
    #trial = "experiment_2018_04_23_lsleeve"
    #trial = "experiment_2018_04_23_lsleeve_warm"

    #trial = "experiment_2018_04_20_ltuck"
    #trial = "experiment_2018_04_21_ltuck"

    #trial = "experiment_2018_04_20_matchgrip_warm"

    #trial = "experiment_2018_04_19_rsleeve_warm"
    #trial = "experiment_2018_04_19_rsleeve"

    #trial = "experiment_2018_04_18_rtuck" #***
    #trial = "experiment_2018_04_18_dropgrip_stablehead" #***

    #new sequence trials above

    #trial = "experiment_2018_04_13_lsleeve_seq_velwarm"
    #trial = "experiment_2018_04_15_lsleeve_seq_velwarm_cont"

    #trial = "experiment_2018_04_11_ltuck_seq_velwarm"

    #trial = "experiment_2018_04_10_match_seq_veltask"
    #trial = "experiment_2018_04_10_rsleeve_seq_highdef_velwarm"

    #trial = "experiment_2018_04_09_match_seqwarm"
    #trial = "experiment_2018_04_09_match_seq"

    #trial = "experiment_2018_04_03_rsleeve_seq_highdef"
    #trial = "experiment_2018_04_03_rsleeve_seq_highdefwarm"

    #trial = "experiment_2018_03_27_lsleeve_widewarm_highdef"
    #trial = "experiment_2018_03_29_rsleeve_seq"

    #trial = "experiment_2018_03_27_lsleeve_narrow_new_cont"
    #trial = "experiment_2018_03_27_lsleeve_narrow_new"
    #trial = "experiment_2018_03_26_lsleeve_narrow" #works but wrong distribution
    #trial = "experiment_2018_03_27_lsleeve_wide"

    #trial = "experiment_2018_03_26_ltuck_wide_warm"
    #trial = "experiment_2018_03_26_ltuck_wide" #***

    #trial = "experiment_2018_03_23_matchgrip_reducedwide" #***
    #trial = "experiment_2018_03_22_matchgrip_narrow2wide"
    #trial = "experiment_2018_03_22_ltuck_narrow_geo"

    #trial = "experiment_2018_03_21_ltuck_narrow" #does not find the inside of the garment

    #trial = "experiment_2018_03_19_sleeveR_wide_lowdef_trpo_cont"
    #trial = "experiment_2018_03_19_sleeveR_wide_lowdef_trpo"
    #trial = "experiment_2018_03_19_matchgrip_narrow" #***
    #trial = "experiment_2018_03_19_sleeveR_narrow2wide_lowdef_trpo" #***

    #trial = "experiment_2018_03_14_sleeveR_wide_lowdef_trpo_cont"
    #trial = "experiment_2018_03_14_sleeveR_narrow0_lowdef_trpo_cont" #***
    #trial = "experiment_2018_03_12_sleeveR_narrow0_lowdef_for_trpo_cont2"

    #trial = "experiment_2018_03_14_sleeveR_narrow0_lowdef_trpo"
    #trial = "experiment_2018_03_14_sleeveR_wide_lowdef_trpo"

    #trial = "experiment_2018_03_12_sleeveR_narrow0_lowdef_for"
    #trial = "experiment_2018_03_12_sleeveR_narrow0_lowdef_for_trpo"
    #trial = "experiment_2018_03_12_sleeveR_narrow0_lowdef_for_trpo_cont"
    #trial = "experiment_2018_03_12_sleeveR_narrow0_lowdef_for_warm"

    #trial = "experiment_2018_03_05_tuckR_triangle_forward" #***
    #trial = "experiment_2018_03_09_sleeveR_narrow0"
    #trial = "experiment_2018_03_09_sleeveR_narrow0forward"
    #trial = "experiment_2018_03_09_sleeveR_narrow0dynamic"
    #trial = "experiment_2018_03_07_sleeveR_wide" #brings the sleeve forward but stays tucked

    #trial = "experiment_2018_03_05_tuckR_triangle_forward"

    #trial = "experiment_2018_03_02_tuckR_triangle_noalign"
    #trial = "experiment_2018_03_02_tuckR_triangle"
    #trial = "experiment_2018_03_02_tuckR_triangle_forsleeve"

    #trial = "experiment_2018_02_28_tuckR_triangle_align"
    #trial = "experiment_2018_02_28_tuckR_triangle"
    #trial = "experiment_2018_02_28_1stsleeveforward_narrow7"

    #trial = "experiment_2018_02_26_tuckR_openness"
    #trial = "experiment_2018_02_26_tuckR_containment"
    #trial = "experiment_2018_02_27_1stsleeve_narrow7"

    #trial = "experiment_2018_02_23_1stsleeve"
    #trial = "experiment_2018_02_21_1stsleeve"

    #trial = "experiment_2018_02_19_tuckR_noCID" #***this actually has CID (go figure)
    #trial = "experiment_2018_02_19_tuckR_noCIDreal" #this actually does not have CID (hm)

    #trial = "experiment_2018_02_16_dropgrip_alignspecific" #***2
    #trial = "experiment_2018_02_16_dropgrip_alignspecific_warm"

    #trial = "experiment_2018_02_11_dropgrip_noalign" #no orientation target
    #trial = "experiment_2018_02_08_dropgrip_align2"
    #trial = "experiment_2018_02_07_dropgrip_align"

    #trial = "experiment_2018_01_30_2armreacher_unlocked_upright_grav_ppo"

    #trial = "experiment_2018_01_28_2armreacher_unlocked_upright"

    #trial = "experiment_2018_01_26_2armreacher_unlocked_precise"
    #trial = "experiment_2018_01_26_2armreacher_locked_precise"
    #trial = "experiment_2018_01_26_2armreacher_locked_superlinear"

    #trial = "experiment_2018_01_25_reacher_locked"
    #trial = "experiment_2018_01_25_reacher_locked_R"

    #trial = "DartClothUpperBodyDataDrivenReacher-v1"
    #trial = "experiment_2018_01_24_SPD_test"
    #trial = "experiment_2018_01_22_timingtest"

    #trial = "experiment_2018_01_16_Ltuck_warm_dist"

    #trial = "experiment_2018_01_15_matchgrip_tiering"
    #trial = "experiment_2018_01_15_matchgrip_dist_xlowpose"
    #trial = "experiment_2018_01_14_matchgrip_dist_lowpose" #***

    #trial = "experiment_2018_01_13_phaseinterpolatejacket_clothplace_warm"
    #trial = "experiment_2018_01_13_jacketL_dist_warm_curriculum"
    #trial = "experiment_2018_01_14_matchgrip_dist_warm3"

    #trial = "experiment_2018_01_12_jacketL_dist_warm"
    #trial = "experiment_2018_01_12_matchgrip_dist_warm"
    #trial = "experiment_2018_01_11_tshirtSleeveRNoOracle"
    #trial = "experiment_2018_01_11_tshirtSleeveRNoHaptics"
    #trial = "experiment_2018_01_09_tshirtR_dist_warm"

    #trial = "experiment_2018_01_11_phaseinterpolatejacket" #***
    #trial = "experiment_2018_01_10_phaseinterpolatejacket" #jacket drops

    #trial = "experiment_2018_01_09_jacketL" #single pose

    #trial = "experiment_2018_01_08_distribution_rightTuck_warm" #***

    #trial = "experiment_2018_01_06_jacketR2"                       #***
    #trial = "experiment_2018_01_04_phaseinterpolate_toR3_cont"      #***
    #trial = "experiment_2018_01_04_phaseinterpolate_matchgrip3_cont" #***

    #trial = "experiment_2018_01_04_jacketR_cont"
    #trial = "experiment_2018_01_04_phaseinterpolate_toL_cont"      #***

    #trial = "experiment_2018_01_04_jacketR"
    #trial = "experiment_2018_01_04_phaseinterpolate_toL"
    #trial = "experiment_2018_01_04_phaseinterpolate_toR3"
    #trial = "experiment_2018_01_04_phaseinterpolate_matchgrip3"

    #trial = "experiment_2018_01_03_phaseinterpolate_toR2"
    #trial = "experiment_2018_01_03_phaseinterpolate_matchgrip"

    #trial = "experiment_2018_01_02_phaseinterpolate_toR"
    #trial = "experiment_2018_01_02_dropgrip2"                      #***
    #trial = "experiment_2018_01_01_dropgrip"

    #trial = "experiment_2017_12_12_halfplane_reacher_cont3"
    #trial = "experiment_2017_12_12_1sdSleeve_progressfocus_cont2"  #***


    #trial = "experiment_2017_12_12_1sdSleeve_progressfocus_cont"
    #trial = "experiment_2017_12_12_1sdSleeve_progressfocus"
    #trial = "experiment_2017_12_12_halfplane_reacher_cont"
    #trial = "experiment_2017_12_08_2ndSleeve_cont"                 #***

    #trial = "experiment_2017_12_08_2ndSleeve"
    #trial = "experiment_2017_12_08_sleeveFeatureOverlapWarmstart"
    #trial = "experiment_2017_12_07_new2reacher2"

    #trial = "experiment_2017_12_07_new2reacher"
    #trial = "experiment_2017_12_05_tshirt_newlimits"

    #trial = "experiment_2017_11_29_tshirt_nodeformationterm2"
    #trial = "experiment_2017_11_16_tshirt_nodeformationterm"
    #trial = "experiment_2017_11_16_tshirt_nodeformationterm_grav"

    #trial = "experiment_2017_09_10_mode7_gripcover"
    #trial = "experiment_2017_09_06_lineargownclose"
    #trial = "experiment_2017_09_12_linearside_warmstart"
    #trial = "experiment_2017_09_11_mode7_nooraclebaseline"
    #trial = "experiment_2017_09_11_mode7_nohapticsbaseline"

    loadSave = False #now done automatically if policy file not found...
    graphOnly = False #if true, exit() after graphing
    demoSave = False #if ture, create an exp named demo folder and render into it

    if loadSave is True:
        import tensorflow as tf
        if trial is not None:
            #load the params.pkl file and save a policy.pkl file
            with tf.Session() as sess:
                print("trying to load the params.pkl file")
                data = joblib.load(prefix+trial+"/params.pkl")
                print("loaded the pkl file")
                policy = data['policy']
                pickle.dump(policy, open(prefix+trial+"/policy.pkl", "wb"))
                print("saved the policy")
                exit()

    print("about to make")

    #construct env
    #env = gym.make('DartClothSphereTube-v1')
    #env = gym.make('DartReacher-v1')
    #env = gym.make('DartClothReacher-v2') #one arm reacher
    #env = gym.make('DartClothReacher-v3') #one arm reacher with target spline
    #env = gym.make('DartClothPoseReacher-v1')  #pose reacher
    #env = gym.make('DartClothSleeveReacher-v1')
    #env = gym.make('DartClothShirtReacher-v1')
    #env = gym.make('DartMultiAgent-v1')
    #env = gym.make('DartClothTestbed-v1')
    #env = gym.make('DartClothGrippedTshirt-v1') #no spline
    #env = gym.make('DartClothGrippedTshirt-v2') #1st arm
    #env = gym.make('DartClothGrippedTshirt-v3') #2nd arm
    #env = gym.make('DartClothEndEffectorDisplacer-v1') #both arms
    #env = gym.make('DartClothJointLimitsTest-v1')
    #env = gym.make('DartClothGownDemo-v1')
    #env = gym.make('DartClothUpperBodyDataDriven-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenTshirt-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenTshirt-v2')
    #env = gym.make('DartClothUpperBodyDataDrivenTshirt-v3')
    #env = gym.make('DartClothUpperBodyDataDrivenTshirtL_HapticHighRes-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenReacher-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenDropGrip-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolate-v1') #dropgrip to tuck right
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolate-v2') #end right sleeve to match grip
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolate-v3') #end match grip to left tuck
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolate-v4') #end match grip to left tuck

    #env = gym.make('DartClothUpperBodyDataDrivenJacket-v1') #jacket right sleeve from grip
    #env = gym.make('DartClothUpperBodyDataDrivenJacket-v2') #jacket left sleeve from grip
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolateJacket-v1') #jacket left sleeve from grip
    #env = gym.make('DartClothUpperBodyDataDrivenPhaseInterpolateJacket-v2') #jacket left sleeve from grip

    #env = gym.make('DartClothUpperBodyDataDrivenLinearTrack-v1') #jacket left sleeve from grip
    #env = gym.make('DartClothUpperBodyDataDrivenAssistLinearTrack-v1') #jacket left sleeve from grip

    #Full Body Data Driven Envs
    #env = gym.make('DartClothFullBodyDataDrivenClothTest-v1') #testing the full body data driven cloth base env setup
    #env = gym.make('DartClothFullBodyDataDrivenClothSPDTest-v1') #testing the full body data driven cloth base env setup with SPD
    #env = gym.make('DartClothFullBodyDataDrivenClothStand-v1')
    #env = gym.make('DartClothFullBodyDataDrivenClothOneFootStand-v1')
    #env = gym.make('DartClothFullBodyDataDrivenClothOneFootStandCrouch-v1')
    #env = gym.make('DartClothFullBodyDataDrivenClothOneFootStandShorts-v1')
    #env = gym.make('DartClothFullBodyDataDrivenClothOneFootStandShorts-v2')
    #env = gym.make('DartClothFullBodyDataDrivenClothOneFootStandShorts-v3')

    #locked foot envs
    #env = gym.make('DartClothFullBodyDataDrivenLockedFootClothTest-v1')
    #env = gym.make('DartClothFullBodyDataDrivenLockedFootClothBalance-v1')
    #env = gym.make('DartClothFullBodyDataDrivenLockedFootClothShortsAlign-v1')

    #Sawyer Env
    #env = gym.make('DartSawyer-v2')
    #env = gym.make('DartSawyerRigid-v4')
    #env = gym.make('DartIiwaRigid-v1')
    #env = gym.make('DartSawyerRigidAssist-v1')
    #env = gym.make('DartClothUpperBodyDataDrivenLinearTrack-v1')

    #envName = 'DartSawyerRigid-v4'
    #envName = 'DartIiwaRigid-v1'
    #envName = 'DartIiwaGown-v1'
    #envName = 'DartIiwaGown-v2'
    #envName = 'DartIiwaGown-v3'
    #envName = 'DartIiwaGown-v4'
    #envName = 'DartIiwaGown-v5'
    #envName = 'DartIiwaGownAssist-v3'
    #envName = 'DartIiwaGownAssist-v4'
    #envName = 'DartIiwaGownAssistCoopt-v2'
    #envName = 'DartIiwaGownAssistCoopt_h-v2'
    #envName = 'DartIiwaGownMultibot-v1'
    #envName = 'DartIiwaOnearmGown-v1'
    envName = 'DartIiwaTwoarmGown-v1'

    if len(sys.argv) > 1:
        #print(sys.argv[1])
        if sys.argv[1] == 'clone' and trial is not None:
            print("Attempting to clone the trial: " + str(trial))
            shutil.copy2(prefix+trial+"/trial_file.py", os.path.join(env_prefix, "experiment_testing_copy_env.py"))
            #now search for the primary class name and change it
            new_env_file = open(os.path.join(env_prefix, "experiment_testing_copy_env_almost.py"), mode='w')
            cur_env_file = open(os.path.join(env_prefix, "experiment_testing_copy_env.py"), mode='r')
            for ix, line in enumerate(cur_env_file):
                if line[:15] == "class DartCloth":
                    #we found the class line
                    #get index of paren:
                    paren_ix = 0
                    for ix,char in enumerate(line):
                        if char == "(":
                            paren_ix = ix
                            break
                    #rename the class
                    new_env_file.write(line[:6] + "DartClothExperimentTestingEnv" + line[paren_ix:])
                else:
                    new_env_file.write(line)
            cur_env_file.close()
            new_env_file.close()
            shutil.copy2(os.path.join(env_prefix, "experiment_testing_copy_env_almost.py"), os.path.join(env_prefix, "experiment_testing_copy_env.py"))
            os.remove(os.path.join(env_prefix, "experiment_testing_copy_env_almost.py"))
            return
        elif sys.argv[1] == 'test':
            envName = "DartTrialTest-v1"

    env = gym.make(envName)

    reloaderTest = False

    if reloaderTest:
        print("reloader test")
        #env.close()
        trials = 0
        failures = 0
        failureRate = 0
        numReloaderSamples = 500
        for i in range(numReloaderSamples):
            trials += 1
            try:
                print("entering try")
                envc = gym.make('DartSawyerRigid-v4')
                #envc = gym.make('DartSawyer-v3')
                #envc = gym.make('DartClothUpperBodyDataDrivenDropGrip-v1')
                #print("reseting")
                #envc.reset()
                #print("stepping")
                #envc.step(action=np.zeros(envc.act_dim))
                #envc.close()
                print("exiting try")
            except:
                failures += 1
            print("----------------------------------------------")
            print("----------------------------------------------")
            print("Number of failures detected: " + str(failures))
            print("Failure rate detected: " + str(failures/trials))
            print("----------------------------------------------")
            print("----------------------------------------------")
        print("done reloader test:")
        print("Number of trials run: " + str(trials))
        print("Number of failures detected: " + str(failures))
        print("Failure rate detected: " + str(failures / trials))
        exit(0)

    useMeanPolicy = True
    showDiscountedReturnGraph = True

    #print("policy time")
    policy = None
    if trial is not None and policy is None:
        try:
            policy = pickle.load(open(prefix+trial+"/policy.pkl", "rb"))
            print(policy)
            useMeanPolicy = True #always use mean if we loaded the policy
            renderGraph(filename=prefix+trial+"/progress.csv")
            if graphOnly:
                exit(1)
        except:
            print("No PICKLE policy file found. Trying joblib...")
            try:
                policy = joblib.load(prefix+trial+"/policy.pkl")
                print(policy)
                useMeanPolicy = True
            except:
                print("FOUND NO POLICY FILE...")
                import tensorflow as tf
                with tf.Session() as sess:
                    print("trying to load the params.pkl file")
                    data = joblib.load(prefix+trial+"/params.pkl")
                    print("loaded the pkl file")
                    policy = data['policy']
                    pickle.dump(policy, open(prefix+trial+"/policy.pkl", "wb"))
                    print("saved the policy")
                    exit()

    #initialize an empty test policy
    if True and policy is None:
        env2 = normalize(GymEnv(envName, record_log=False, record_video=False))
        #env2 = normalize(GymEnv('DartSawyerRigidAssist-v1', record_log=False, record_video=False))
        if False:
            human_obs_size = 251 #single arm coopt
            human_obs_size = env2.wrapped_env.env.human_obs_manager.obs_size
            policy = SplitGaussianMLPPolicy(
                env_spec=env2.spec,
                # The neural network policy should have two hidden layers, each with 32 hidden units.
                hidden_sizes=(64, 64),
                #split_index=281, #multibot
                split_index=human_obs_size,
                merge_index=22,
                #hidden_sizes=(128, 64),
                #init_std=0.2 #exploration scaling
                #init_std=0.15 #exploration scaling #human
                #init_std=0.1 #exploration scaling #SPD human
                init_std=0.15 #robot
            )
            for out_layer in policy._mean_network.split_output_layers:
                all_param_values = L.get_all_param_values(out_layer)
                all_param_values[4] *= 0.01
                L.set_all_param_values(out_layer, all_param_values)

        else:
            policy = GaussianMLPPolicy(
                env_spec=env2.spec,
                # The neural network policy should have two hidden layers, each with 32 hidden units.
                hidden_sizes=(128, 64),
                #hidden_sizes=(128, 64),
                #init_std=0.2 #exploration scaling
                #init_std=0.15 #exploration scaling #human
                #init_std=0.1 #exploration scaling #SPD human
                init_std=0.15 #robot
            )
            all_param_values = L.get_all_param_values(policy._mean_network.output_layer)
            #output bias scaling
            all_param_values[4] *= 0.01 #human
            #all_param_values[4] *= 0.002 #robot
            L.set_all_param_values(policy._mean_network.output_layer, all_param_values)
        env2._wrapped_env.env._render(close=True)
        useMeanPolicy = False #don't use the mean when we want to test a fresh policy initialization
    #print(policy.output_layer)

    if demoSave:
        #first create the folder
        demo_directory = "/home/alexander/Documents/frame_capture_output/demos/"
        demo_directory += trial
        if not os.path.exists(demo_directory):
            os.makedirs(demo_directory)
        env._get_viewer().capturing = True
        env._get_viewer().captureDirectory = demo_directory

    print("about to run")
    paused = False
    #useMeanPolicy = True $set mean policy usage
    time.sleep(0.5)
    cumulativeFPS = 0
    completedRollouts = 0 #counts rollouts which were not terminated early
    successfulTrials = 0
    failureTrials = 0
    env.render()
    #time.sleep(30.0) #window setup time for recording
    #o = env.reset()
    #useMeanPolicy = False

    for i in range(16):
        print("-----------------------------------")
        print("  Starting rollout " + str(i))
        print("-----------------------------------")
        o = env.reset()
        #if i == 7:
        #    env.renderSPDGhost = False
        #envFilename = env.getFile()
        #print(envFilename)
        env.render()
        #time.sleep(0.5)
        rolloutHorizon = 10000
        rolloutHorizon = 600
        #rolloutHorizon = 1200
        #rolloutHorizon = 200
        #rolloutHorizon = 900
        #rolloutHorizon = 10000
        if paused is True:
            rolloutHorizon = 10000
        startTime = time.time()
        #for j in range(rolloutHorizon):
        #start_pose = np.array(env.robot_skeleton.q[6:])
        while(env.numSteps < rolloutHorizon):
            #if j%(rolloutHorizon/10) == 0:
            #    print("------- Checkpoint: " + str(j/(rolloutHorizon/10)) + "/10 --------")
            a = np.zeros(env.act_dim) #22 dof upper body, ?? dof full body

            #SPD target
            #start_pose[31-6] += 0.01
            #a = np.array(start_pose)

            #a = -np.ones(len(a))
            #a += np.random.uniform(-1,1,len(a))
            #a[:11] = np.zeros(11)
            #a += np.random.randint(3, size=len(a))-np.ones(len(a))
            '''
            if j==0:
                a = np.array([-1,-0.5935519015,-0.6243126472,-1,-0.3540411152,-0.8545956428,-0.1052807823,-0.6650868959,1,-1,-0.4370771514,-1,-0.2656309561,-0.7392283111,1,-0.4849024561,-0.4222881197,-1,-1,-0.1260703,1,0.3853144958,])
            elif j==1:
                a = np.array([1,0.08203909,-0.4428489711,0.3709899779,-0.1139084987,0.8878356518,-0.3833406323,0.9175109866,-1,-0.7288833698,-0.3778503588,-0.0617086992,-1,0.5471811498,1,-1,-0.4266441964,-1,0.2783551927,0.0862617301,0.5444295707,0.7144071905])
            elif j==2:
                a = np.array([-1,1,-0.6846910321,0.5774784709,-0.7145496691,-0.7416754164,-1,0.9724756555,-1,1,-1,-0.9628565439,-1,1,-0.9544127885,1,0.5642344238,-0.1455457015,-0.3926989475,-1,1,-0.0842431477])
            elif j==3:
                a = np.array([1,0.6312312283,-0.6876604936,0.5467897784,-0.9867554189,-1,-1,0.314068975,0.2136389088,-1,1,-1,-0.1857029911,0.933112181,-1,-0.9219502237,0.7421179613,1,-1,0.0583067668,1,-0.3022806922])
            '''
            #print(a)
            if policy is not None:
                action, a_info = policy.get_action(o)
                #print(a_info['mean'])
                a = action
                if useMeanPolicy:
                    a = a_info['mean']
                as_ub = np.ones(env.action_space.shape)
                action_space = spaces.Box(-1 * as_ub, as_ub)
                lb, ub = action_space.bounds
                scaled_action = lb + (a + 1.) * 0.5 * (ub - lb)
                scaled_action = np.clip(scaled_action, lb, ub)
                a=scaled_action
            done = False
            if not paused or env.numSteps == 0:# or j==0:
                s_info = env.step(a)
                o = s_info[0]
                done = s_info[2]
                #print(s_info)
                #print(o)
            env.render()

            j = env.numSteps
            if done is True:
                print("killed at step " + str(j))
                cumulativeFPS += (j+1)/(time.time()-startTime)
                print("framerate = " + str((j+1) / (time.time() - startTime)))
                print("average FPS: " + str(cumulativeFPS / (i + 1)))
                #print("episode reward = " + str(env.rewardsData.cumulativeReward))
                #if
                time.sleep(0.5)
                break
            if j == rolloutHorizon-1:
                #print("startTime = " + str(startTime))
                #print("endTime = " + str(time.time()))
                #print("totalTime = " + str(time.time()-startTime))
                cumulativeFPS += rolloutHorizon/(time.time()-startTime)
                print("framerate = " + str(rolloutHorizon/(time.time()-startTime)))
                print("average FPS: " + str(cumulativeFPS/(i+1)))
                print("total rollout time: " + str(time.time()-startTime))
            #    print("Time terminate")
            #paused = True
    env.render(close=True)
    

if __name__ == '__main__':
    main()