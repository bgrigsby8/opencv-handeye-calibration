from ctypes import POINTER, Structure, c_double, cdll
import math
from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from viam.components.arm import Pose

import os
import sys
import ctypes
from viam.proto.common import Orientation
import importlib.resources as importlib_resources


class EulerAngles(Structure):
    pass

class OrientationVector(Structure):
    pass


class Quaternion(Structure):
    pass


ov_array = c_double * 4


def load_shared_library(package_name, lib_path, lib_name):
    """
    Extract and load a shared library using importlib_resources backport
    """
    # Determine correct file extension based on platform
    if sys.platform.startswith("win"):
        ext = ".dll"
    elif sys.platform == "darwin":
        ext = ".dylib"
    else:  # Linux/Unix
        ext = ".so"

    if not lib_name.endswith(ext):
        lib_name += ext

    package_path = importlib_resources.files(package_name)
    lib_file = package_path.joinpath(lib_path, lib_name)
    return ctypes.CDLL(os.fspath(lib_file))


class SpatialMath:
    def __init__(self) -> None:
        self.viam_rust_utils = load_shared_library("viam", "rpc", "libviam_rust_utils")
        self.viam_rust_utils.free_orientation_vector_memory.argtypes = (
            POINTER(OrientationVector),
        )
        self.viam_rust_utils.free_quaternion_memory.argtypes = (POINTER(Quaternion),)
        self.viam_rust_utils.free_euler_angles_memory.argtypes = (POINTER(EulerAngles),)
        self.viam_rust_utils.orientation_vector_from_quaternion.argtypes = (
            POINTER(Quaternion),
        )
        self.viam_rust_utils.orientation_vector_from_quaternion.restype = POINTER(
            OrientationVector
        )
        self.viam_rust_utils.orientation_vector_get_components.argtypes = (
            POINTER(OrientationVector),
        )
        self.viam_rust_utils.orientation_vector_get_components.restype = POINTER(
            ov_array
        )
        self.viam_rust_utils.quaternion_get_components.argtypes = (POINTER(Quaternion),)
        self.viam_rust_utils.quaternion_get_components.restype = POINTER(ov_array)
        self.viam_rust_utils.new_quaternion.argtypes = (
            c_double,
            c_double,
            c_double,
            c_double,
        )
        self.viam_rust_utils.new_quaternion.restype = POINTER(Quaternion)
        self.viam_rust_utils.new_orientation_vector.argtypes = (
            c_double,
            c_double,
            c_double,
            c_double,
        )
        self.viam_rust_utils.new_orientation_vector.restype = POINTER(OrientationVector)
        self.viam_rust_utils.new_euler_angles.argtypes = (c_double, c_double, c_double)
        self.viam_rust_utils.new_euler_angles.restype = POINTER(EulerAngles)
        self.viam_rust_utils.quaternion_from_euler_angles.argtypes = (
            c_double,
            c_double,
            c_double,
        )
        self.viam_rust_utils.quaternion_from_euler_angles.restype = POINTER(Quaternion)
        self.viam_rust_utils.quaternion_from_orientation_vector.argtypes = (
            POINTER(OrientationVector),
        )
        self.viam_rust_utils.quaternion_from_orientation_vector.restype = POINTER(
            Quaternion
        )
        self.viam_rust_utils.euler_angles_from_quaternion.argtypes = (
            POINTER(Quaternion),
        )
        self.viam_rust_utils.euler_angles_from_quaternion.restype = POINTER(EulerAngles)
        self.viam_rust_utils.orientation_vector_from_quaternion.argtypes = (
            POINTER(Quaternion),
        )
        self.viam_rust_utils.orientation_vector_from_quaternion.restype = POINTER(
            OrientationVector
        )

    def create_orientation_vector(
        self, o_x: float, o_y: float, o_z: float, theta: float
    ) -> OrientationVector:
        return self.viam_rust_utils.new_orientation_vector(o_x, o_y, o_z, theta)

    def create_quaternion(
        self, real: float, i: float, j: float, k: float
    ) -> Quaternion:
        return self.viam_rust_utils.new_quaternion(real, i, j, k)

    def create_euler_angles(self, roll: float, pitch: float, yaw: float) -> EulerAngles:
        return self.viam_rust_utils.new_euler_angles(roll, pitch, yaw)

    def quaternion_from_euler_angles(
        self, roll: float, pitch: float, yaw: float
    ) -> Quaternion:
        return self.viam_rust_utils.quaternion_from_euler_angles(roll, pitch, yaw)

    def quaternion_from_orientation_vector(
        self, o_vec: OrientationVector
    ) -> Quaternion:
        return self.viam_rust_utils.quaternion_from_orientation_vector(o_vec)

    def euler_angles_from_quaternion(self, quat: Quaternion) -> EulerAngles:
        return self.viam_rust_utils.euler_angles_from_quaternion(quat)

    def orientation_vector_from_quaternion(self, quat: Quaternion) -> OrientationVector:
        return self.viam_rust_utils.orientation_vector_from_quaternion(quat)

    def orientation_vector_get_components(
        self, o_vec: OrientationVector
    ) -> Tuple[float, float, float, float]:
        return self.viam_rust_utils.orientation_vector_get_components(o_vec).contents

    def quaternion_get_components(
        self, quat: Quaternion
    ) -> Tuple[float, float, float, float]:
        return self.viam_rust_utils.quaternion_get_components(quat).contents

    def free_quaternion_memory(self, quat: Quaternion):
        self.viam_rust_utils.free_quaternion_memory(quat)

    def free_orientation_vector_memory(self, o_vec: OrientationVector):
        self.viam_rust_utils.free_orientation_vector_memory(o_vec)

    def free_euler_angles_memory(self, euler_angles: EulerAngles):
        self.viam_rust_utils.free_euler_angles_memory(euler_angles)


def invert_transform(T):
    """Inverts a 4x4 transformation matrix."""
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.identity(4)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    return T_inv

def build_transform(rot, t):
    """Builds a 4x4 transform from 3x3 rotation matrix and 3x1 translation vector."""
    T = np.identity(4)
    T[0:3, 0:3] = np.array(rot)
    T[0:3, 3] = np.array(t).flatten()
    return T

def T_to_R_t(T):
    """Decomposes a 4x4 transform into rotation matrix R and translation vector t."""
    R_mat = T[0:3, 0:3]
    t_vec = T[0:3, 3]
    return R_mat, t_vec

def R_to_rvec(R_mat):
    """Converts a 3x3 rotation matrix to a 3x1 rotation vector (Rodrigues)."""
    rvec, _ = cv2.Rodrigues(R_mat)
    return rvec.flatten()

def apriltag_detection_to_matrix(detection) -> np.ndarray:
    """Converts a pupil_apriltags Detection object's pose to a 4x4 NumPy matrix."""
    # detection.pose_R is the 3x3 rotation matrix (Tag frame -> Camera frame)
    # detection.pose_t is the 3x1 translation vector (Tag origin in Camera frame, meters)
    return build_transform(detection.pose_R, detection.pose_t)

def invert_pose(R, t):
    """Inverts a transformation represented by rotation matrix R and translation t."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv

def rotation_vector_to_matrix(rvec):
    """Converts a rotation vector (axis-angle) to a rotation matrix."""
    # Note: OpenCV's Rodrigues function handles this
    R, _ = cv2.Rodrigues(rvec)
    return R

def make_homogeneous_matrix(R, t):
    """Creates a 4x4 homogeneous matrix from R (3x3) and t (3x1)."""
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T
