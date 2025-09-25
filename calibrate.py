import asyncio
import cv2
import numpy as np
import json
import subprocess
import os
from typing import List

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image

# Load environment variables from .env file
load_dotenv()


# ---- Config ----
JOINT_POSITIONS_RAD = [
    [3.4791183471679688, -0.910707787876465, 2.211569611226217, -1.6651135883727013, -1.89414626756777, -1.7194626967059536],
    [3.22637677192688, -0.5175885719111939, 1.7706020514117637, -1.1642515522292634, -1.8507912794696253, -1.378885571156637],
    [3.4008841514587402, -0.5378768008998414, 1.9870556036578575, -1.3628363621285935, -1.8508146444903775, -1.6463573614703575],
    [3.400956392288208, -0.34310360372576915, 1.7705948988543911, -1.1642001432231444, -1.850769821797506, -1.3788660208331507],
    [2.33966573874, -0.24109776437792974, 1.7116163412677212, -1.024131791000702, -1.857204262410299, -1.4428246657000943],
    [2.191972255706787, -0.9402409356883544, 2.0781582037555144, -1.3654778760722657, -1.8532460371600552, -1.4670594374286097],
    [2.1857411861419678, -1.1597858232310791, 2.1369345823871058, -1.441519693737366, -1.9163535276996055, -1.6312916914569302],
    [2.1329727172851562, -0.8977993291667482, 2.1480796972857874, -1.4416735929301763, -1.9167941252337855, -1.6313355604754847],
    [2.45, -0.595451847915985, 2.1466482321368616, -1.4863408741405983, -1.9168379942523404, -1.6310394446002403],
    [2.6973185539245605, -1.0562105935863038, 2.321317021046774, -1.6285578213133753, -1.9168060461627405, -1.6310661474811001],
    [2.697420358657837, -0.9689555925181884, 2.238986317311422, -1.4576139015010376, -1.7893617788897915, -1.62169582048525],
    [2.8718435764312744, -0.9688817423633119, 2.0645979086505335, -1.545034484272339, -1.876694981251852, -2.583079640065329]
]

# Convert radians to degrees
JOINT_POSITIONS = []
for joint_set in JOINT_POSITIONS_RAD:
    degrees_set = [np.degrees(joint) for joint in joint_set]
    JOINT_POSITIONS.append(degrees_set)

# Chessboard pattern configuration
PATTERN_SIZE = (9, 6)  # (width, height) in corners
SQUARE_SIZE_M = 0.021 # 21mm squares

MIN_SAMPLES = 5
# -----------------


async def get_camera_intrinsics(camera: Camera) -> tuple:
    """Get camera intrinsic parameters"""
    properties = await camera.get_properties()
    intrinsics = properties.intrinsic_parameters
    
    K = np.array([
        [intrinsics.focal_x_px, 0, intrinsics.center_x_px],
        [0, intrinsics.focal_y_px, intrinsics.center_y_px],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # These are values from viam's do command
    dist = np.array([
        0.11473497003316879,    # k1 - radial distortion
        -0.31621694564819336,  # k2 - radial distortion  
        0.00024490756914019585,    # p1 - tangential distortion
        -0.0002616790879983455,    # p2 - tangential distortion
        0.2385278344154358     # k3 - radial distortion
    ], dtype=np.float32)
    
    print(f"Camera intrinsics: K shape={K.shape}, dist shape={dist.shape}")
    print(f"Distortion coefficients: {dist}")
    
    return K, dist

def call_go_ov2mat(ox: float, oy: float, oz: float, theta: float) -> np.ndarray:
    """
    Call Go script to convert Viam orientation vector to rotation matrix
    
    Args:
        ox, oy, oz, theta: Viam orientation vector components
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    try:
        # Call Go script with orientation vector parameters
        result = subprocess.run([
            'go', 'run', 'main.go',
            str(ox), str(oy), str(oz), str(theta)
        ], capture_output=True, text=True, cwd='./orientation_converter')
        
        if result.returncode != 0:
            print(f"Go script error: {result.stderr}")
            return None
            
        # Parse the 9 values returned by Go script (3x3 matrix flattened)
        values = [float(x) for x in result.stdout.strip().split()]
        if len(values) != 9:
            print(f"Expected 9 values from Go script, got {len(values)}")
            return None
            
        # Reshape to 3x3 matrix
        return np.array(values).reshape(3, 3)
        
    except Exception as e:
        print(f"Failed to call Go orientation converter: {e}")
        return None
    
def call_go_mat2ov(R: np.ndarray) -> tuple:
    """
    Call Go script to convert rotation matrix to Viam orientation vector
    
    Args:
        R: 3x3 rotation matrix as numpy array
        
    Returns:
        (ox, oy, oz, theta): Orientation vector components, or None if failed
    """
    try:
        # Flatten the rotation matrix to get 9 elements
        flat_matrix = R.flatten()
        matrix_args = [str(val) for val in flat_matrix]
        
        # Call Go script with mat2ov command and 9 matrix elements
        result = subprocess.run([
            'go', 'run', 'main.go', 'mat2ov'
        ] + matrix_args, capture_output=True, text=True, cwd='./orientation_converter')
        
        if result.returncode != 0:
            print(f"Go script error: {result.stderr}")
            return None
            
        # Parse the 4 values returned by Go script (ox, oy, oz, theta)
        values = [float(x) for x in result.stdout.strip().split()]
        if len(values) != 4:
            print(f"Expected 4 values from Go script, got {len(values)}")
            return None
            
        # Return orientation vector components
        return values[0], values[1], values[2], values[3]
        
    except Exception as e:
        print(f"Failed to call Go mat2ov converter: {e}")
        return None

def detect_chessboard_pose(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple:
    """
    Detect chessboard pattern in image and compute pose
    
    Args:
        image: Grayscale image
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        
    Returns:
        (R, t) where R is 3x3 rotation matrix, t is 3x1 translation vector
        Returns (None, None) if detection fails
    """
    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(
        image, 
        PATTERN_SIZE, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if not found:
        print("Could not find chessboard pattern in image")
        return None, None
    
    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    
    # Generate 3D object points for the chessboard (Z=0 plane)
    objp = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    
    # Solve PnP to get pose
    success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    
    if not success:
        print("Could not solve PnP for chessboard")
        return None, None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Convert tvec to column vector (3x1)
    t = tvec.reshape(3, 1) * 1000  # Convert to mm
    
    return R, t

def save_calibration_data(R_gripper2base: List[np.ndarray], t_gripper2base: List[np.ndarray], 
                         R_target2cam: List[np.ndarray], t_target2cam: List[np.ndarray]):
    """Save calibration data to JSON file"""
    data = {
        'R_gripper2base': [R.flatten().tolist() for R in R_gripper2base],  # Flatten 3x3 to 9 elements
        't_gripper2base': [t.flatten().tolist() for t in t_gripper2base],  # 3x1 to 3 elements
        'R_target2cam': [R.flatten().tolist() for R in R_target2cam],
        't_target2cam': [t.flatten().tolist() for t in t_target2cam],
    }
    
    with open('calibration_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibration data with {len(R_gripper2base)} samples to calibration_data.json")

def perform_hand_eye_calibration(R_gripper2base: List[np.ndarray], t_gripper2base: List[np.ndarray],
                                R_target2cam: List[np.ndarray], t_target2cam: List[np.ndarray]):
    """
    Perform OpenCV hand-eye calibration
    
    Returns:
        (R_cam2gripper, t_cam2gripper): Camera to gripper transformation
    """
    print(f"\nPerforming hand-eye calibration with {len(R_gripper2base)} pose pairs...")
    
    try:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        print("Hand-eye calibration successful!")
        print(f"Camera to gripper translation: {t_cam2gripper.flatten()} mm")
        
        return R_cam2gripper, t_cam2gripper
        
    except Exception as e:
        print(f"Hand-eye calibration failed: {e}")
        return None, None

async def connect():
    """Connect to Viam robot"""
    api_key = os.getenv('VIAM_API_KEY')
    api_key_id = os.getenv('VIAM_API_KEY_ID')
    robot_address = os.getenv('VIAM_ROBOT_ADDRESS')
    
    if not api_key or not api_key_id or not robot_address:
        raise ValueError("Missing required environment variables: VIAM_API_KEY, VIAM_API_KEY_ID, VIAM_ROBOT_ADDRESS")
    
    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id
    )
    return await RobotClient.at_address(robot_address, opts)

async def main():
    machine = await connect()
    
    try:
        arm = Arm.from_robot(machine, "ur20-modular")
        camera = Camera.from_robot(machine, "sensing-camera")
        
        # Get camera intrinsics
        K, dist = await get_camera_intrinsics(camera)
        print(f"Camera intrinsics loaded: K shape {K.shape}")
        
        # Initialize data collection lists
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        
        print(f"Starting calibration data collection for {len(JOINT_POSITIONS)} poses...")
        
        for i, joints in enumerate(JOINT_POSITIONS):
            print(f"\n--- Moving to pose {i+1}/{len(JOINT_POSITIONS)} ---")
            
            # Move arm to joint position
            jp = JointPositions(values=joints)
            await arm.move_to_joint_positions(jp)
            
            # Wait for movement to complete
            while await arm.is_moving():
                await asyncio.sleep(0.05)
            
            # Wait for pose to stabilize
            print("Waiting for pose to stabilize...")
            await asyncio.sleep(1.0)
            
            # Get arm end-effector pose
            arm_pose = await arm.get_end_position()
            print(f"Arm position: ({arm_pose.x:.3f}, {arm_pose.y:.3f}, {arm_pose.z:.3f}) mm")
            print(f"Arm orientation: ({arm_pose.o_x:.3f}, {arm_pose.o_y:.3f}, {arm_pose.o_z:.3f}, {arm_pose.theta:.3f})")
            
            # Convert orientation vector to rotation matrix using Go script
            R_g2b = call_go_ov2mat(arm_pose.o_x, arm_pose.o_y, arm_pose.o_z, arm_pose.theta)
            if R_g2b is None:
                print("Failed to convert arm orientation, skipping pose")
                continue
                
            t_g2b = np.array([[arm_pose.x], [arm_pose.y], [arm_pose.z]], dtype=np.float64)
            
            # Capture camera image
            viam_image = await camera.get_image()
            pil_image = viam_to_pil_image(viam_image)
            image = np.array(pil_image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect chessboard and compute target-to-camera pose
            R_t2c, t_t2c = detect_chessboard_pose(image, K, dist)
            if R_t2c is None:
                print("Failed to detect chessboard pattern, skipping pose")
                continue
            
            print(f"Target position in camera frame: {t_t2c.flatten()} mm")
            
            # Store the pose pair
            print(f"Arm R: {R_g2b}")
            print(f"Arm t: {t_g2b}")
            print(f"Tag R: {R_t2c}")
            print(f"Tag t: {t_t2c}")
            R_gripper2base_list.append(R_g2b)
            t_gripper2base_list.append(t_g2b)
            R_target2cam_list.append(R_t2c)
            t_target2cam_list.append(t_t2c)
            
            print(f"Successfully collected pose pair {len(R_gripper2base_list)}")
        
        # Check if we have enough samples
        n_samples = len(R_gripper2base_list)
        print(f"\nCollected {n_samples} valid pose pairs")
        
        if n_samples < MIN_SAMPLES:
            print(f"Need at least {MIN_SAMPLES} samples for calibration, only got {n_samples}")
            return
        
        # Save calibration data
        save_calibration_data(R_gripper2base_list, t_gripper2base_list, 
                            R_target2cam_list, t_target2cam_list)
        
        # Perform hand-eye calibration
        R_cam2gripper, t_cam2gripper = perform_hand_eye_calibration(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list
        )
        
        if R_cam2gripper is not None:
            print(f"Camera to gripper rotation matrix:\n{R_cam2gripper}")
            print(f"Camera to gripper translation: {t_cam2gripper.flatten()} mm")
            
            # Convert final rotation matrix back to orientation vector
            final_ov = call_go_mat2ov(R_cam2gripper)
            if final_ov is not None:
                ox, oy, oz, theta = final_ov
                print(f"Camera to gripper orientation vector: ({ox:.6f}, {oy:.6f}, {oz:.6f}, {theta:.6f})")
                print(f"Theta in degrees: {np.degrees(theta):.3f}°")
            else:
                print("Failed to convert rotation matrix to orientation vector")
            
            # Save final calibration result
            result = {
                'R_cam2gripper': R_cam2gripper.tolist(),
                't_cam2gripper': t_cam2gripper.tolist(),
                'n_samples': n_samples
            }
            
            with open('hand_eye_calibration_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("Calibration result saved to hand_eye_calibration_result.json")
        
    finally:
        await machine.close()

if __name__ == "__main__":
    asyncio.run(main())