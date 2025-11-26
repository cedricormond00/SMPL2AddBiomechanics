#!/usr/bin/env python
import os
from pathlib import Path
from collections import defaultdict
import pickle
import numpy as np
import argparse


def build_smpl_sequence(params_dir, gender, fps=30):
    """
    Build a SMPL sequence in the format expected by smpl2ab.utils.load_smpl_seq.

    Required output keys / shapes:
        trans  : (T, 3)
        poses  : (T, 72) - [global_orient(3) | body_pose(69)]
        betas  : (10,)   - single shape vector (padded or truncated to 10)
        gender : str
        mocap_framerate : float (e.g. 120.0)
        dmpls  : (T, 8) or similar (here zeros, for compatibility)
    """
    params_dir = Path(params_dir)
    d = defaultdict(list)

    # Collect per-frame parameters
    pkl_files = sorted(params_dir.glob("*smpl.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *smpl.pkl files found in {params_dir}")

    for fp in pkl_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        # Expected per-frame keys:
        #   body_pose     : (1, 69) or (69,)
        #   global_orient : (1, 3)  or (3,)
        #   betas         : (1, B)  or (B,)
        #   transl        : (1, 3)  or (3,)
        body_pose = np.asarray(data["body_pose"]).reshape(-1, 69)
        global_orient = np.asarray(data["global_orient"]).reshape(-1, 3)
        betas = np.asarray(data["betas"]).reshape(-1, data["betas"].shape[-1])
        transl = np.asarray(data["transl"]).reshape(-1, 3)

        d["body_pose"].append(body_pose[0])
        d["global_orient"].append(global_orient[0])
        d["betas"].append(betas[0])
        d["transl"].append(transl[0])

    # Stack over time: T x ...
    body_pose_seq = np.stack(d["body_pose"], axis=0)        # (T, 69)
    global_orient_seq = np.stack(d["global_orient"], axis=0)  # (T, 3)
    transl_seq = np.stack(d["transl"], axis=0)              # (T, 3)
    betas_seq = np.stack(d["betas"], axis=0)                # (T, B)

    T, B = betas_seq.shape

    # Single shape vector for whole sequence: mean over frames
    betas_mean = betas_seq.mean(axis=0)                     # (B,)

    # Pad or truncate to length 10
    TARGET_BETAS = 10
    if betas_mean.shape[0] < TARGET_BETAS:
        betas_single = np.zeros(TARGET_BETAS, dtype=np.float32)
        betas_single[:betas_mean.shape[0]] = betas_mean.astype(np.float32)
    else:
        betas_single = betas_mean.astype(np.float32)[:TARGET_BETAS]

    # Build poses: (T, 72) = [global_orient(3) | body_pose(69)]
    poses_seq = np.concatenate([global_orient_seq, body_pose_seq], axis=1)  # (T, 72)

    # Optional dmpls: zeros, not used by smpl2ab but matches AMASS-ish structure
    dmpls = np.zeros((T, 8), dtype=np.float32)

    smpl_params = {
        "poses": poses_seq.astype(np.float32),      # (T, 72)
        "trans": transl_seq.astype(np.float32),     # (T, 3)
        "betas": betas_single,                      # (16,)
        "gender": gender,
        "mocap_framerate": float(fps),             # key endswith('rate') -> OK for load_smpl_seq
        "dmpls": dmpls,
    }

    return smpl_params


def save_smpl_params(out_path, smpl_params, save_pkl=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out_path.with_suffix(".npz")
    np.savez(
        npz_path,
        poses=smpl_params["poses"],
        trans=smpl_params["trans"],
        betas=smpl_params["betas"],
        gender=smpl_params["gender"],
        mocap_framerate=smpl_params["mocap_framerate"],
        dmpls=smpl_params["dmpls"],
    )
    print(f"Saved NPZ SMPL params to: {npz_path}")

    if save_pkl:
        pkl_path = out_path.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(smpl_params, f)
        print(f"Saved PKL SMPL params to: {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a SMPL parameter sequence (.npz) in the format expected by SMPL2AddBiomechanics."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to folder containing *smpl.pkl files (one per frame).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=["male", "female"],
        help="SMPL model gender.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="smpl_params",
        help="Output file prefix (without extension).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Framerate to store as 'mocap_framerate'.",
    )
    parser.add_argument(
        "--save-pkl",
        action="store_true",
        help="Also save a .pkl copy of the SMPL params.",
    )
    args = parser.parse_args()

    smpl_params = build_smpl_sequence(args.path, args.gender, args.fps)
    save_smpl_params(args.out, smpl_params, save_pkl=args.save_pkl)
