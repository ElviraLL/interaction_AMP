import os
import os.path as osp
import argparse
import glob
import tqdm
import pickle
import numpy as np
import torch
from smplx import SMPLX
from scipy.spatial.transform import Rotation as sRot
import sys

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

input_folder = "ase/data/motions/chair_cut"
output_folder = "ase/data/motions/chair_corrected"

files = glob.glob(osp.join(input_folder, '*.npy'))
for file in files:
    save_path = osp.join(output_folder, osp.basename(file))
    data = dict(np.load(file, allow_pickle=True).item())
    fps = data['fps']
    skip = int(fps // 30)

    print("load_data")
    root_trans = torch.from_numpy(data['root_translation']['arr'][::skip]) # [1799, 3] -> pick every skip frames
    local_rotation_quat = torch.from_numpy(data['rotation']['arr'][::skip])
    N = local_rotation_quat.shape[0]
    skeleton_tree = SkeletonTree.from_dict(data['skeleton_tree'])

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, local_rotation_quat, root_trans, is_local=True
    )

    pose_quat_global = (
        (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        )
        .as_quat()
        .reshape(N, -1, 4)
    )

    rot_x_90 = sRot.from_euler(
        "x", 90, degrees=True
    )  # rotate around x axis for 90 degree
    pose_quat_global = (
        (rot_x_90 * sRot.from_quat(pose_quat_global.reshape(-1, 4)))
        .as_quat()
        .reshape(N, -1, 4)
    )
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        root_trans,
        is_local=False,
    )
    motion = SkeletonMotion.from_skeleton_state(new_sk_state, 30)
    motion.to_file(save_path)
    break




