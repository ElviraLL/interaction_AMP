
## run visualization
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/chair_cut/chair_mo_sit2sit_stageII.npy
```

## run AMP training
```
python ase/run.py --task HumanoidAMPObject --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/chair_cut 
```

# run AMP testing
```
python ase/run.py --test --task HumanoidAMPObject --num_envs 16 --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_im.yaml --motion_file ase/data/motions/chair_cut/chair_mo_sit2sit_stageII.npy --checkpoint .output/Humanoid_09-09-55-03/nn/Humanoid.pth
```

## Resume training
```
python ase/run.py --task HumanoidAMPObject --cfg_env ase/data/cfg/humanoid_location_scene.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid_im.yaml --motion_file ase/data/motions/chair_cut/chair_mo_sit2sit_stageII.npy  --resume 1 --checkpoint ./output/Humanoid_

```

# tensor board
```
tensorboard --logdir=summaries