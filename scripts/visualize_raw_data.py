import smplx
import torch
import pickle
import trimesh
import tqdm
input_path = "dataset/pkl/chair_mo_sit2sit_stageII.pkl"
model_path = "character/data/body_models"
gender = "male"

body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)

with open(input_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
    betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
    full_trans = torch.tensor( data['pose_est_trans'], dtype=torch.float32)
    print("Number of frames is {}".format(full_poses.shape[0]))

for i in tqdm.tqdm(range(full_poses.shape[0])):
    global_orient = full_poses[i,0:3].reshape(1,-1)
    body_pose = full_poses[i,3:66].reshape(1,-1)
    transl = full_trans[i,:].reshape(1,-1)
    output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
    m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
    m.show()