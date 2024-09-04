import os
import os.path as osp
import argparse
import glob
import tqdm
import pickle
import numpy as np
import torch
from smplx import SMPLX
from scipy.spatial.transform import Rotation as R


from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

VISUALIZE = False

# extract 20 amp_humanoid_smplx joints from 55 SMPL-X joints
# joints_to_use = np.array(
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# )
selected_joints = np.array(
    [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 13, 16, 18, 20, 14, 17, 19, 21]
)
# joints_to_use = np.array([
#     0, 3, 12, 14, 17, 19, 13, 16, 18, 2, 5, 8, 1, 4, 7
# ])
joints_to_use = np.arange(0, 165).reshape((-1, 3))[selected_joints].reshape(-1)


def dataloader_SAMP(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        fr = data['mocap_framerate']
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        full_trans = torch.tensor(data['pose_est_trans'], dtype=torch.float32)
    return full_poses, full_trans, fr


def dataloader_AMASS(path):
    data = dict(np.load(path, allow_pickle=True))
    fr = data['mocap_frame_rate'].tolist()
    full_poses = torch.tensor(data['poses'], dtype=torch.float32)
    full_trans = torch.tensor(data['trans'], dtype=torch.float32)

    # avoid ground penetration/floating
    model = SMPLX(
        model_path=body_model_dir,
        gender=gender,
        batch_size=10,  # use 10 begin frames
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True
    )
    model_output = model(
        global_orient=full_poses[0:10, 0:3],
        body_pose=full_poses[0:10, 3:66],
        transl=full_trans[0:10, :],
    )
    begin_min_height = torch.min(model_output.vertices[:, :, 2].cpu().detach().reshape(-1))  # assume up-axis is Z
    full_trans[:] -= torch.tensor([0, 0, begin_min_height])

    return full_poses, full_trans, fr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, choices=["SAMP", "AMASS"], default="SAMP")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--body_model_dir", type=str, default="character/data/body_models/smplx")
    parser.add_argument("--gender", type=str, choices=["neutral", "male", "female"], default="male")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    body_model_dir = args.body_model_dir
    gender = args.gender

    motion_dir = "data/motions/chair"
    os.makedirs(motion_dir, exist_ok=True)

    # read all sequences
    if dataset == "SAMP":
        params = [x for x in glob.glob(osp.join(dataset_dir, "*.pkl")) if x.split("/")[-1].startswith("chair")]
        dataloader = dataloader_SAMP
    elif dataset == "AMASS":
        params = glob.glob(osp.join(dataset_dir, dataset, "*/*/*.npz"))
        dataloader = dataloader_AMASS
    else:
        print("Unsupported dataset: {s}\n".format(dataset))
        assert (False)

    print("processing dataset: {} num_params: {}\n".format(dataset, len(params)))

    # load and visualize t-pose files
    # TODO: rewrite a tpose with more bones, need pelvis, spine1, spine2, spine3, neck
    tpose_file = "character/data/amp_humanoid_smplx_tpose_{}.npy".format(gender)
    if not osp.exists(tpose_file):
        print("tpose file not existed! creating from mjcf file...\n")

        # load in XML mjcf file and save zero rotation pose in npy format
        # xml_path = "data/assets/mjcf_smplx/amp_humanoid_smplx_{}.xml".format(gender)
        xml_path = 'ase/data/assets/mjcf/smpl_humanoid_19.xml'
        skeleton = SkeletonTree.from_mjcf(xml_path)
        tpose = SkeletonState.zero_pose(skeleton)
        tpose.to_file(tpose_file)
    else:
        print("reading tpose...\n")
        tpose = SkeletonState.from_file(tpose_file)

    # visualize zero rotation pose
    if VISUALIZE:
        plot_skeleton_state(tpose)

    # generate reference motion
    pbar = tqdm.tqdm(params)
    for path in pbar:
        name = osp.basename(path)
        pbar.set_description(name)
        full_poses, full_trans, fps = dataloader(path)

        # extract useful joints
        full_poses = full_poses[:, joints_to_use].reshape(-1, len(selected_joints), 3)
        
        # angle axis ---> quaternion
        pose_quat_isaac = (
            R.from_rotvec(full_poses.reshape(-1, 3)).as_quat().reshape(-1, len(selected_joints), 4)
        )
        N = pose_quat_isaac.shape[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            tpose.skeleton_tree,
            torch.from_numpy(pose_quat_isaac),
            full_trans,
            is_local=True,
        )
        pose_quat_global = (
            (
                R.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
                * R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            )
            .as_quat()
            .reshape(N, -1, 4)
        )

        # global_translation = new_sk_state.global_transformation[
        #     :, :, 4:
        # ]  # global transformation: [N, 19, 7]
        # min_z = global_translation[:, :, 2].min()  # min for all frames and all joints
        # full_trans[:, 2] -= min_z

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            tpose.skeleton_tree,
            torch.from_numpy(pose_quat_global),
            full_trans,
            is_local=False,
        )
        motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        motion.to_file(osp.join(motion_dir, "{}.npy".format(name[:-4])))


        # visualize motion
        if VISUALIZE:
            plot_skeleton_motion_interactive(motion)

print("done!")