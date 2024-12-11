from types import SimpleNamespace

global_configs = SimpleNamespace()

# We only care about the "car" category.
#
# Also see nuscenes_utils.py::map_name_from_general_to_detection
# Type A: treat as foreground, and put back
# Type B: treat as foreground, and remove
# Type C: treat as background, don't care
global_configs.nuscenes_extract_class_names = [
    "car",  # treat as foreground, and put back
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    # "barrier",  # treat as background, don't care
    "motorcycle",
    "bicycle",
    "pedestrian",
    # "traffic_cone",  # treat as background, don't care
    "ignore",
]

global_configs.nuscenes_extract_class_name_to_label = {
    class_name: i + 1
    for i, class_name in enumerate(global_configs.nuscenes_extract_class_names)
}

global_configs.nuscenes_extract_label_to_class_name = {
    label: class_name
    for class_name, label in global_configs.nuscenes_extract_class_name_to_label.items()
}

global_configs.nuscenes_class_names_to_recon = [
    "car",
    # "truck",
    # "construction_vehicle",
    # "bus",
    # "trailer",
]

global_configs.nuscenes_class_labels_to_recon = [
    global_configs.nuscenes_extract_class_name_to_label[class_name]
    for class_name in global_configs.nuscenes_class_names_to_recon
]
