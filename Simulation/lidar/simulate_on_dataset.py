import numpy as np
import os
import argparse
from util import *
import random
from glob import glob
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm
from compound_model import compounding



class PointCloudDataset(Dataset):
    def __init__(self, dataset_root, noise_model):
        """
        Initializes the dataset by automatically retrieving sequence paths,
        point cloud paths, and label paths (if available) in KITTI format.
        Also prepares destination paths for simulated point clouds.

        Parameters:
            dataset_root (str): Root directory of the KITTI dataset.
            noise_model (callable): Function or model for applying noise.
        """

        # Automatically find available sequences in dataset_root/sequences/
        dataset_path = os.path.join(dataset_root, "sequences")
        # dataset_path = os.path.join(dataset_root)
        target_root = dataset_path.replace("sequences", "simulated_sequences")
        # target_root =  "/path/to/local/Desktop/PCL_Visualization/KITTI_Sample/simulated_sequences/"
        seq_names = sorted([seq for seq in os.listdir(dataset_path) if seq.isdigit()])
        seq_list = [os.path.join(dataset_path, str(seq)) for seq in seq_names]
        dest_seq_list = [os.path.join(target_root, str(seq)) for seq in seq_names]

        # Ensure destination structure follows KITTI format
        for seq_path in dest_seq_list:
            os.makedirs(os.path.join(seq_path, "velodyne"), exist_ok=True)
            os.makedirs(os.path.join(seq_path, "labels"), exist_ok=True)

        # Initialize lists for file paths
        self.pc_path_list = []
        self.label_path_list = []
        self.dest_pc_path_list = []
        self.dest_label_path_list = []

        # Populate file lists by iterating over sequences
        for i, seq in enumerate(seq_list):
            velodyne_files = sorted(glob(os.path.join(seq, "velodyne", "*.bin")))
            label_files = sorted(glob(os.path.join(seq, "labels", "*.label")))

            # Ensure label files are aligned with point cloud files
            label_dict = {os.path.basename(f): f for f in label_files}
            aligned_labels = [label_dict.get(os.path.basename(pc), None) for pc in velodyne_files]

            self.pc_path_list.extend(velodyne_files)
            self.label_path_list.extend(aligned_labels)

            pc_filenames = [os.path.basename(pc) for pc in velodyne_files]
            label_filenames = [os.path.basename(label) if label else None for label in aligned_labels]

            self.dest_pc_path_list.extend([os.path.join(dest_seq_list[i], "velodyne", filename) for filename in pc_filenames])
            self.dest_label_path_list.extend([os.path.join(dest_seq_list[i], "labels", filename) if filename else None for filename in label_filenames])

        self.noise_model = noise_model  # Store noise model for future processing

    def __getitem__(self, idx):
        pc_path = self.pc_path_list[idx]
        label_path = self.label_path_list[idx]  # May be None if labels are missing
        pc_target_path = self.dest_pc_path_list[idx]
        label_target_path = self.dest_label_path_list[idx] if self.dest_label_path_list[idx] else None

        return pc_path, label_path, pc_target_path, label_target_path

    def __len__(self):
        return len(self.pc_path_list)
    
def process_sample(idx, dataset):

    pc_path, label_path, pc_target_path, _ = dataset[idx]

    # 1. Read original data
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    
    label = np.fromfile(label_path, dtype=np.int32) if label_path is not None else None

    xyz = pc[:, :3]
    intensity = pc[:, 3]
    if intensity.max() < 1:
        intensity *= dataset.noise_model.lidar.max_intensity
    print(intensity.max())
    # 2. Simulate
    sim_pc, sim_intensity = dataset.noise_model.simulate(xyz, intensity)

    # 3. Write to destination
    sim_pc_intensity = np.concatenate([sim_pc, sim_intensity[:, None]], axis=1).astype(np.float32)
    if label is not None:
        sim_label = sim_label.astype(np.int32)

    sim_pc_intensity.tofile(pc_target_path)
    # sim_label.tofile(label_target_path)

    return idx  # or return some info for logging

###############################################################################################
if __name__ == '__main__': 
     # Parse arguments for selecting the  fog, rain or snow model
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--weather", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--weather_params", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--noise", type=str, default="", help="internal noise such as calibration error")
    parser.add_argument("--noise_params", type=str, default="", help="noise configuration file")
    parser.add_argument("--sensor_cfg", type=str, default="", help="sensor configuration file")
    parser.add_argument("--seed", type=int, default=20, help="random seed")
    parser.add_argument("--brighten", action="store_true", help="brighten the point cloud in visualisation")
    args = parser.parse_args()
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Simulate the weather based on the selected model
    compounding_class = compounding(args.weather, args.weather_params, args.noise, args.noise_params, args.sensor_cfg)
    dataset = PointCloudDataset(args.dataset_root, compounding_class)
    indices = list(range(len(dataset)))
    results = Parallel(n_jobs=35, verbose=10)(
        delayed(process_sample)(i, dataset) 
        for i in tqdm(indices, desc="Simulating Rain")
    )

    print("All samples processed. Number of results:", len(results))


