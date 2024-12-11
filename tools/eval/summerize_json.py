import json
import re
from pathlib import Path

_pwd = Path(__file__).parent.absolute()
_project_root = _pwd.parent.parent
_eval_root = _project_root / "eval"

config_to_json_names = {
    "mini_translator_none": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_None.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_None.json",
    ],
    "mini_translator_nksr_names": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_nksr.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_nksr.json",
    ],
    "mini_translator_nksr_recompute_voxels_force_nearest_sweeps_names": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_nksr_recompute_voxels_force_nearest_sweeps.json",
    ],
    "full_translator_none_names": [
        "da_waymo_nuscenes_trainval_max_sweeps_1_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_2_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_3_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_4_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_5_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_6_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_7_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_8_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_9_translator_None.json",
        "da_waymo_nuscenes_trainval_max_sweeps_10_translator_None.json",
    ],
    "full_translator_none_force_nearest_sweeps_names": [
        "da_waymo_nuscenes_trainval_max_sweeps_1_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_2_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_3_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_4_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_5_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_6_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_7_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_8_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_9_translator_None_force_nearest_sweeps.json",
        "da_waymo_nuscenes_trainval_max_sweeps_10_translator_None_force_nearest_sweeps.json",
    ],
    "mini_translator_pointersect_force_nearest_sweeps_k_50_names": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_50.json",
    ],
    "mini_translator_pointersect_force_nearest_sweeps_k_1_names": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_1.json",
    ],
    "mini_translator_pointersect_force_nearest_sweeps_sweeps_10_ks_names": [
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_2.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_3.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_4.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_5.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_6.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_7.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_8.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_9.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_10.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_20.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_30.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_40.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_50.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_60.json",
    ],
    "mini_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1_names": [
        "da_waymo_nuscenes_mini_max_sweeps_1_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_2_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_3_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_4_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_5_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_6_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_7_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_8_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_9_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_mini_max_sweeps_10_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
    ],
    "translator_pointersect_force_nearest_sweeps_k_1_names": [
        "da_waymo_nuscenes_trainval_max_sweeps_1_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_2_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_3_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_4_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_5_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_6_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_7_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_8_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_9_translator_pointersect_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_10_translator_pointersect_force_nearest_sweeps_k_1.json",
    ],
    "translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1_names": [
        "da_waymo_nuscenes_trainval_max_sweeps_1_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_2_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_3_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_4_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_5_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_6_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_7_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_8_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_9_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
        "da_waymo_nuscenes_trainval_max_sweeps_10_translator_pointersect_recompute_voxels_force_nearest_sweeps_k_1.json",
    ],
}


def main():
    for config_name, json_names in config_to_json_names.items():
        print(f"# {config_name}")
        print("max_sweeps,ap_bev,ap_3d,avg_num_points")
        for json_name in json_names:
            path = _eval_root / json_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            max_sweeps = int(re.findall(r"max_sweeps_(\d+)", json_name)[0])
            ap_bev = data["Car_bev/moderate_R40"]
            ap_3d = data["Car_3d/moderate_R40"]
            avg_num_points = data["avg_num_points"]
            print(f"{max_sweeps},{ap_bev},{ap_3d},{avg_num_points / 1000:.02f}K")


if __name__ == "__main__":
    main()
