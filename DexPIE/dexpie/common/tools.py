#!/usr/bin/env python3
import math
import numpy as np
import cv2

class  MATHTOOLS:
    
    # Create a transformation matrix from xyz+rpy.
    def xyzrpy2Mat(self,x, y, z, roll, pitch, yaw):
        transformation_matrix = np.eye(4)
        A = np.cos(yaw)
        B = np.sin(yaw)
        C = np.cos(pitch)
        D = np.sin(pitch)
        E = np.cos(roll)
        F = np.sin(roll)
        DE = D * E
        DF = D * F
        transformation_matrix[0, 0] = A * C
        transformation_matrix[0, 1] = A * DF - B * E
        transformation_matrix[0, 2] = B * F + A * DE
        transformation_matrix[0, 3] = x
        transformation_matrix[1, 0] = B * C
        transformation_matrix[1, 1] = A * E + B * DF
        transformation_matrix[1, 2] = B * DE - A * F
        transformation_matrix[1, 3] = y
        transformation_matrix[2, 0] = -D
        transformation_matrix[2, 1] = C * F
        transformation_matrix[2, 2] = C * E
        transformation_matrix[2, 3] = z
        transformation_matrix[3, 0] = 0
        transformation_matrix[3, 1] = 0
        transformation_matrix[3, 2] = 0
        transformation_matrix[3, 3] = 1
        return transformation_matrix
    
    
    def mat2xyzrpy(self,matrix):
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.asin(-matrix[2, 0])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        return [x, y, z, roll, pitch, yaw]
    
    def xyzQuaternion2matrix(x, y, z, qx, qy, qz, qw):
        # Convert quaternion components into a 3x3 rotation matrix.
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
        
        # Build a homogeneous transform with translation [x, y, z].
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T

    def rpy_to_rotvec(self,roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert RPY angles around X/Y/Z axes to a rotation vector.
        
        Args:
            roll: Rotation around X axis in radians.
            pitch: Rotation around Y axis in radians.
            yaw: Rotation around Z axis in radians.
        
        Returns:
            Rotation vector [rx, ry, rz]; direction is the axis and norm is the angle in radians.
        """
        # Build rotation matrices around each axis.
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Compose rotation matrix in Z-Y-X order.
        R = np.dot(R_z, np.dot(R_y, R_x))
        
        # Compute rotation vector from rotation matrix.
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if np.abs(theta) < 1e-10:  # Zero rotation.
            return np.array([0, 0, 0])
        else:
            # Rotation axis.
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * np.sin(theta))
            
            # Rotation vector = rotation axis * rotation angle.
            return axis * theta


    def rotvec_to_rpy(self,rotvec: np.ndarray) -> tuple:
        """
        Convert a rotation vector to RPY angles around X/Y/Z axes.
        
        Args:
            rotvec: Rotation vector [rx, ry, rz].
        
        Returns:
            (roll, pitch, yaw): Rotations around X/Y/Z axes in radians.
        """
        theta = np.linalg.norm(rotvec)
        
        if np.abs(theta) < 1e-10:  # Zero rotation.
            return (0, 0, 0)
        
        # Rotation axis.
        axis = rotvec / theta
        
        # Build rotation matrix with Rodrigues' formula.
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
        # Extract RPY from rotation matrix using Z-Y-X order.
        # Reference: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return (roll, pitch, yaw)
    
    def rotvec2mat(self,rotvec):
        # Convert rotation vector to column-convention rotation matrix.
        theta = np.linalg.norm(rotvec)
        eps = 1e-6
        if theta < eps:
            return np.eye(3)
        
        axis = rotvec / theta
        K = np.array([
            [0,    -axis[2],  axis[1]],
            [axis[2], 0,      -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    def xyz_rotvec_to_mat(self,xyz_rotvec):
        pos=xyz_rotvec[:3]
        rotvec=xyz_rotvec[3:]
        T=np.eye(4)
        T[:3,3]=pos
        R=self.rotvec2mat(rotvec)
        T[:3,:3]=R
        return T
    def mat2xyz_rotvec(self,matrix, eps=1e-6):
        """
        Extract xyz + rotation vector from a column-convention 4x4 homogeneous matrix.
        
        Args:
            matrix: 4x4 homogeneous transform matrix as a numpy array.
            eps: Singularity detection threshold.
        
        Returns:
            [x, y, z, rx, ry, rz] 
            where [rx, ry, rz] is rotation axis times rotation angle in radians.
        """
        # Extract translation.
        x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        
        # Extract rotation matrix.
        R = matrix[:3, :3]
        
        # Compute rotation angle theta.
        trace = np.trace(R)
        cos_theta = max(min((trace - 1) / 2.0, 1.0), -1.0)  # Numerically safe.
        theta = math.acos(cos_theta)
        
        # Compute rotation vector.
        if theta < eps:
            # Near-zero rotation.
            rx, ry, rz = 0.0, 0.0, 0.0
            
        elif theta > math.pi - eps:
            # theta is near pi and sin(theta) is near zero, so handle it specially.
            # Rotation axis can be any nonzero column of (R + I).
            diag = np.diag(R)
            idx = np.argmax(diag)  # Choose the most stable column.
            
            axis = np.array([R[0, idx], R[1, idx], R[2, idx]])
            axis[idx] += 1  # Column idx of (R + I).
            
            # Normalize.
            axis_norm = np.linalg.norm(axis)
            axis = axis / axis_norm if axis_norm > eps else np.array([1.0, 0.0, 0.0])
            
            rx, ry, rz = axis * theta
            
        else:
            # Regular case: use Rodrigues' formula.
            sin_theta = math.sin(theta)
            factor = theta / (2 * sin_theta)
            
            rx = factor * (R[2, 1] - R[1, 2])
            ry = factor * (R[0, 2] - R[2, 0])
            rz = factor * (R[1, 0] - R[0, 1])
        
        return [x, y, z, rx, ry, rz]
    
    def se3_inverse(self,T):
        """
        Compute the inverse of an SE(3) transformation matrix.
        For T = [R | t; 0 | 1], the inverse is T⁻¹ = [Rᵀ | -Rᵀt; 0 | 1]
        
        Args:
            T: 4x4 SE(3) transformation matrix
            
        Returns:
            4x4 inverse transformation matrix
        """
        R = T[:3, :3]  # 3x3 rotation matrix
        t = T[:3, 3]   # 3x1 translation vector
        
        # Compute R transpose
        R_transpose = R.T
        
        # Compute -Rᵀt
        t_inv = -np.dot(R_transpose, t)
        
        # Build the inverse matrix
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_transpose
        T_inv[:3, 3] = t_inv
        
        return T_inv
  
    def normalize(self,x, axis=-1, eps=1e-8):
        """Apply L2 normalization along the specified axis."""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    def rot6d_to_mat(self,d6:np.ndarray):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = self.normalize(a1)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = self.normalize(b2)
        b3 = np.cross(b1, b2, axis=-1)
        out = np.stack((b1, b2, b3), axis=-2)
        return out
    def xyz_6drot_to_mat(self,xyz_6drot:np.ndarray):
        batch_dims = xyz_6drot.shape[:-1]
        pos=xyz_6drot[...,:3]
        rot_6d=xyz_6drot[...,3:]
        mats = np.tile(np.eye(4), batch_dims + (1, 1))
        rot_mat= self.rot6d_to_mat(rot_6d)
        mats[..., :3, :3] = rot_mat
        mats[..., :3, 3] = pos
        return mats
    def mat_to_rot6d(self,mat:np.ndarray):
        batch_dim = mat.shape[:-2]
        out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
        return out
    def mat2xyz_6drot(self, matrix: np.ndarray, eps=1e-6):
        # Check input dimensions and handle batched transforms.
        if len(matrix.shape) == 3:  # Batched [T, 4, 4].
            batch_size = matrix.shape[0]
            # Extract translation [T, 3].
            pos = matrix[:, :3, 3]
            # Extract rotation matrix [T, 3, 3].
            R = matrix[:, :3, :3]
            # Convert to 6D rotation representation [T, 6].
            rot_6d = self.mat_to_rot6d(R)
            # Concatenate position and rotation [T, 9].
            result = np.concatenate([pos, rot_6d], axis=-1)
            return result
        else:  # Single 4x4 matrix.
            x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
            # Extract rotation matrix.
            R = matrix[:3, :3]
            rot_6d = self.mat_to_rot6d(R)
            return np.array([x, y, z, *rot_6d])
