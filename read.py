
import pickle
# from click import Path
import numpy as np

# load the .npy file
demo_sample = np.load("/local/home/cormond/SMPL2AddBiomechanics/models/bsm/sample_motion/01/01_01_poses.npz", allow_pickle=True)
my_sample = np.load("/local/home/cormond/SMPL2AddBiomechanics/inputs/smpl_seq/bike_take3/bike_take3.npz", allow_pickle=True)
# with Path("/local/home/cormond/SMPL2AddBiomechanics/models/bsm/sample_motion/01/01_01_poses.npz").open("rb") as f:
#     data = pickle.load(f)

# mesh1=np.load("/media/cormond/hdd/data/pilot_oct10/bike_take3/smpl_fit/mesh-f00001_smpl.pkl", allow_pickle=True)
# mesh2=np.load("/media/cormond/hdd/data/pilot_oct10/bike_take3/smpl_fit/mesh-f00001_smpl.ply", allow_pickle=True)
# mesh3=
# inspect and use

# print("shape:", arr.shape, "dtype:", arr.dtype)
# print("shape2:", arr2.shape, "dtype2:", arr2.dtype)
# example: print first 5 rows / elements

print(demo_sample.files)
print("poses shape: ", demo_sample['poses'].shape)
print("trans shape: ", demo_sample['trans'].shape)
print("betas shape: ", demo_sample['betas'].shape)
print("gender: ", demo_sample['gender'])    
print("mocap_rate: ", demo_sample['mocap_framerate'])

print(my_sample.files)
print("poses shape: ", my_sample['poses'].shape)
print("trans shape: ", my_sample['trans'].shape)
print("betas shape: ", my_sample['betas'].shape)
print("gender: ", my_sample['gender'])    
print("mocap_rate: ", my_sample['mocap_framerate'])
    
# print(mesh1['vertices'].shape)
# print(mesh1['faces'].shape)
# print(mesh2).keys()
# print(mesh2['vertices'].shape)
# print(mesh2['faces'].shape)
