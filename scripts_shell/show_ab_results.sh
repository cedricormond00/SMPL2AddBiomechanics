take_name=$1

python smpl2ab/show_ab_results_c.py \
    --osim_path=/local/home/cormond/SMPL2AddBiomechanics/addbiomechanics/$take_name/Models/match_markers_but_ignore_physics.osim \
    --mot_path=/local/home/cormond/SMPL2AddBiomechanics/addbiomechanics/$take_name/IK/${take_name}_segment_0_ik.mot \
    --smpl_motion_path=/local/home/cormond/SMPL2AddBiomechanics/inputs/smpl_seq/$take_name/$take_name.npz