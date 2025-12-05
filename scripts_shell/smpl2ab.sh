# take in a name of the take as argument
# e.g., bike_take3
# outputs to inputs/smpl_seq/<take_name>/<take_name>.npz    

#take in input argument
take_name=$1

python ./scripts/build_smpl_for_smpl2ab.py \
  --path /media/cormond/hdd/data/pilot_oct10/$take_name/smpl_fit \
  --gender male \
  --out inputs/smpl_seq/$take_name/$take_name \
  --fps 30

python ./smpl2ab/smpl2addbio.py -i  inputs/smpl_seq/$take_name