#!/usr/bin/env python3
"""
Test Go-Generated Calibration Data

Quick test of calibration data from Go script.
"""

import numpy as np
import cv2
import json


def invert(R, t):
    Rt = R.T
    return Rt, -Rt @ t

def compose(Ra, ta, Rb, tb):
    return Ra @ Rb, Ra @ tb + ta

def residual_spread(R_bg_list, t_bg_list, R_tc_list, t_tc_list, R_est, t_est):
    # OpenCV returns cam->gripper; convert to gTc
    R_gTc, t_gTc = invert(R_est, t_est)
    # cTt for each sample
    cTt = [invert(R, t) for R,t in zip(R_tc_list, t_tc_list)]
    # bTt_i = bTg_i * gTc * cTt_i
    trans = []
    angles = []
    for (R_bg,t_bg),(R_ct,t_ct) in zip(zip(R_bg_list,t_bg_list), cTt):
        R_bt, t_bt = compose(*compose(R_bg,t_bg, R_gTc,t_gTc), R_ct,t_ct)
        trans.append(t_bt.flatten())
        angles.append(np.linalg.norm(cv2.Rodrigues(R_bt)[0]))
    trans = np.vstack(trans)
    std = trans.std(axis=0)
    return std, np.linalg.norm(std), np.std(angles)

def try_all(R_btg, t_btg, R_ttc, t_ttc):
    methods = [
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ("HORAUD",     cv2.CALIB_HAND_EYE_HORAUD),
        ("PARK",       cv2.CALIB_HAND_EYE_PARK),
        ("TSAI",       cv2.CALIB_HAND_EYE_TSAI),
        ("ANDREFF",    cv2.CALIB_HAND_EYE_ANDREFF),
    ]
    # four input configurations
    configs = {
        "bTg + tTc": (R_btg, t_btg, R_ttc, t_ttc),
        "bTg + cTt": (R_btg, t_btg, [R.T for R in R_ttc], [-R.T@t for R,t in zip(R_ttc,t_ttc)]),
        "gTb + tTc": ([R.T for R in R_btg], [-R.T@t for R,t in zip(R_btg,t_btg)], R_ttc, t_ttc),
        "gTb + cTt": ([R.T for R in R_btg], [-R.T@t for R,t in zip(R_btg,t_btg)],
                      [R.T for R in R_ttc], [-R.T@t for R,t in zip(R_ttc,t_ttc)]),
    }
    for cfg_name, (Rg, tg, Rt, tt) in configs.items():
        print(f"\n=== {cfg_name} ===")
        for name, flag in methods:
            R_est, t_est = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=flag)
            std_vec, std_norm, rot_spread = residual_spread(Rg, tg, Rt, tt, R_est, t_est)
            print(f"{name:10s}  t=[{t_est[0,0]:6.1f},{t_est[1,0]:6.1f},{t_est[2,0]:6.1f}]  "
                  f"bTt std [mm]={std_vec}  ||std||={std_norm:.1f}  rot σ={rot_spread:.3f} rad")

def test_go_calibration():
    """Test calibration with Go data."""
    try:
        with open('./go_testing/go_calibration.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ go_calibration_data.json not found. Run Go script first.")
        return
    
    # Convert data back to matrices
    R_gripper2base_list = []
    for r_flat in data['R_gripper2base']:
        R_matrix = np.array(r_flat).reshape(3, 3)
        R_gripper2base_list.append(R_matrix)
    
    t_gripper2base_list = [np.array(t).reshape(3, 1) for t in data['t_gripper2base']]
    
    R_target2cam_list = []
    for r_flat in data['R_target2cam']:
        R_matrix = np.array(r_flat).reshape(3, 3)
        R_target2cam_list.append(R_matrix)
    
    t_target2cam_list = [np.array(t).reshape(3, 1) for t in data['t_target2cam']]
    
    print(f"Go data: {len(R_gripper2base_list)} poses")
    
    if len(R_gripper2base_list) < 3:
        print("❌ Need at least 3 poses")
        return
    
    return try_all(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list)
    
    # Test OpenCV calibration
    methods = [("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("PARK", cv2.CALIB_HAND_EYE_PARK)]
    expected = np.array([74.66119635, -18.36436101, 48.61454126])
    
    for method_name, method_flag in methods:
        try:
            R_est, t_est = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=method_flag
            )

            # Build cTt_i by inverting your target->cam
            cTt = [invert(R, t) for R, t in zip(R_target2cam_list, t_target2cam_list)]

            # gTc is the calibration result (OpenCV returns cam->gripper)
            R_gTc, t_gTc = invert(R_est, t_est)  # turn cam->gripper into gTc

            bTt_trans = []
            bTt_rodr  = []

            for (R_bg, t_bg), (R_ct, t_ct) in zip(zip(R_gripper2base_list, t_gripper2base_list), cTt):
                # bTt_i = bTg_i * gTc * cTt_i
                R_bt, t_bt = compose(*compose(R_bg, t_bg, R_gTc, t_gTc), R_ct, t_ct)
                bTt_trans.append(t_bt.flatten())
                # rotation mag (Rodrigues angle)
                angle = np.linalg.norm(cv2.Rodrigues(R_bt)[0])
                bTt_rodr.append(angle)

            bTt_trans = np.vstack(bTt_trans)
            spread_mm = bTt_trans.std(axis=0)
            print("Base->Tag translation std [mm]:", spread_mm, " | total std norm:", np.linalg.norm(spread_mm))
            print("Base->Tag rotation spread [rad]:", np.std(bTt_rodr))
            
            translation = t_est.flatten()
            error = np.linalg.norm(translation - expected)
            
            print(f"R_est: {R_est}")
            print(f"{method_name}: [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}] mm, error: {error:.1f} mm")
            
        except Exception as e:
            print(f"{method_name}: Failed - {e}")


if __name__ == "__main__":
    test_go_calibration()