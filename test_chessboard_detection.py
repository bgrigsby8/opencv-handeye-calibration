#!/usr/bin/env python3
"""
Test Chessboard Pose Detection

Test script to detect chessboard pose from a static image file.
"""

import cv2
import numpy as np
import argparse
import sys


# Chessboard pattern configuration (matching your calibrate.py)
PATTERN_SIZE = (9, 6)  # (width, height) in corners
SQUARE_SIZE_M = 0.025  # 25mm squares


def get_default_camera_matrix():
    """
    Return a default camera matrix for testing.
    Replace with your actual camera intrinsics for accurate results.
    """
    # Default camera matrix (you should replace with your actual camera parameters)
    K = np.array([
        [800.0, 0, 320.0],
        [0, 800.0, 240.0],
        [0, 0, 1.0]
    ], dtype=np.float32)
    
    # Assuming minimal distortion
    dist = np.zeros((4, 1), dtype=np.float32)
    
    print("Using default camera matrix - replace with actual camera intrinsics for accurate results")
    return K, dist


def detect_chessboard_pose(image: np.ndarray, K: np.ndarray, dist: np.ndarray, visualize: bool = False) -> tuple:
    """
    Detect chessboard pattern in image and compute pose
    
    Args:
        image: Input image (color or grayscale)
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        visualize: Whether to show detection results
        
    Returns:
        (R, t, corners) where R is 3x3 rotation matrix, t is 3x1 translation vector, corners are detected corners
        Returns (None, None, None) if detection fails
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"Image shape: {image.shape}")
    print(f"Looking for chessboard pattern: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} corners")
    
    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(
        gray, 
        PATTERN_SIZE, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if not found:
        print("❌ Could not find chessboard pattern in image")
        return None, None, None
    
    print(f"✅ Found chessboard pattern with {len(corners)} corners")
    
    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Generate 3D object points for the chessboard (Z=0 plane)
    objp = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    
    # Solve PnP to get pose
    success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    
    if not success:
        print("❌ Could not solve PnP for chessboard")
        return None, None, None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Convert tvec to column vector (3x1)
    t = tvec.reshape(3, 1)
    
    print(f"✅ Pose detection successful!")
    print(f"Translation (meters): [{t[0,0]:.4f}, {t[1,0]:.4f}, {t[2,0]:.4f}]")
    print(f"Translation (mm): [{t[0,0]*1000:.1f}, {t[1,0]*1000:.1f}, {t[2,0]*1000:.1f}]")
    
    # Convert rotation matrix to Euler angles for easier interpretation
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True)
    print(f"Rotation (Euler XYZ degrees): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
    
    if visualize:
        # Draw the corners and coordinate system
        img_vis = image.copy()
        
        # Draw corners
        cv2.drawChessboardCorners(img_vis, PATTERN_SIZE, corners, found)
        
        # Draw coordinate system (3D axes)
        axis_length = SQUARE_SIZE_M * 3  # 3 squares length
        axis_points = np.array([
            [0, 0, 0],           # Origin
            [axis_length, 0, 0], # X-axis (red)
            [0, axis_length, 0], # Y-axis (green)
            [0, 0, -axis_length] # Z-axis (blue, pointing up from board)
        ], dtype=np.float32)
        
        # Project 3D points to image
        projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
        projected_points = projected_points.reshape(-1, 2).astype(int)
        
        origin = tuple(projected_points[0])
        x_axis = tuple(projected_points[1])
        y_axis = tuple(projected_points[2])
        z_axis = tuple(projected_points[3])
        
        # Draw axes
        cv2.arrowedLine(img_vis, origin, x_axis, (0, 0, 255), 5)  # X-axis (red)
        cv2.arrowedLine(img_vis, origin, y_axis, (0, 255, 0), 5)  # Y-axis (green)
        cv2.arrowedLine(img_vis, origin, z_axis, (255, 0, 0), 5)  # Z-axis (blue)
        
        # Add labels
        cv2.putText(img_vis, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_vis, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_vis, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show result
        cv2.imshow('Chessboard Pose Detection', img_vis)
        print("Press any key to close visualization window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return R, t, corners


def main():
    parser = argparse.ArgumentParser(description='Test chessboard pose detection from image file')
    parser.add_argument('image_path', help='Path to chessboard image file')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show detection results')
    parser.add_argument('--camera-matrix', help='Path to camera intrinsics JSON file (optional)')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"❌ Could not load image from: {args.image_path}")
        sys.exit(1)
    
    # Load camera matrix if provided
    if args.camera_matrix:
        try:
            import json
            with open(args.camera_matrix, 'r') as f:
                cam_data = json.load(f)
            K = np.array(cam_data['camera_matrix'], dtype=np.float32)
            dist = np.array(cam_data['distortion_coefficients'], dtype=np.float32)
            print(f"Loaded camera matrix from: {args.camera_matrix}")
        except Exception as e:
            print(f"❌ Could not load camera matrix: {e}")
            print("Using default camera matrix instead")
            K, dist = get_default_camera_matrix()
    else:
        K, dist = get_default_camera_matrix()
    
    # Detect chessboard pose
    R, t, corners = detect_chessboard_pose(image, K, dist, args.visualize)
    
    if R is not None:
        print(f"\n📐 Rotation Matrix:")
        print(R)
        print(f"\n📍 Translation Vector (meters):")
        print(t)
        print(f"\n✅ Pose detection completed successfully!")
    else:
        print(f"\n❌ Pose detection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()