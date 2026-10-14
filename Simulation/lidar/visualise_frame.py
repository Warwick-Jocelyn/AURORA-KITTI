import numpy as np
import os
import time
import importlib
import argparse
from util import *
import sys
from lidar import Lidar
import random
from compound_model import compounding

###############################################################################################
if __name__ == '__main__': 
     # Parse arguments for selecting the  fog, rain or snow model
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--weather", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--weather_params", type=str, default="", help="Model to simulate: fog, rain, or snow")
    parser.add_argument("--noise", type=str, default="", help="internal noise such as calibration error")
    parser.add_argument("--noise_params", type=str, default="", help="noise configuration file")
    parser.add_argument("--sensor_cfg", type=str, default="", help="sensor configuration file")
    parser.add_argument("--seed", type=int, default=20, help="random seed")
    parser.add_argument("--depth_color", action="store_true", help="depth is used for coloring")
    parser.add_argument("--radar", action="store_true", help="input is radar data")

    args = parser.parse_args()
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load example point clouds from .pcd file 
    file_path = "./Example_PointClouds/CE_Car_15m_Clear.pcd" if args.file_path == "" else args.file_path
    if ".pcd" in file_path:
        clear_pc, clear_intensity = load_pcd_pointcloud(file_path, args.radar)
    elif '.mat' in file_path:
        clear_pc, clear_intensity = load_mat_pointcloud(file_path)
    elif '.npy' in file_path:
        clear_pc, clear_intensity = load_npy_pointcloud(file_path)
    elif '.bin' in file_path:
        clear_pc, clear_intensity, clear_label = load_bin_pointcloud(file_path)
    # Simulate the weather based on the selected model
    compounding_class = compounding(args.weather, args.weather_params, args.noise, args.noise_params, args.sensor_cfg)
    if clear_intensity.max() < 1:
        clear_intensity = clear_intensity * compounding_class.lidar.max_intensity
    pc_sim, intensity_sim = compounding_class.simulate(clear_pc, clear_intensity)
     # We print the dimensions of both original and the simulated point clouds to note the difference
  
    # Plot the point clouds and set the color based on the depth of the points
    non_zero_depth_mask = np.linalg.norm(clear_pc, axis=1) > 0.0
    pc_coords_filtered = clear_pc[non_zero_depth_mask]
    print(f"Dimensions of pc_original: {pc_coords_filtered.shape}")
    print(f"Dimensions of simulated pc: {pc_sim.shape}")
    intensity_filtered = clear_intensity[non_zero_depth_mask]
    depth_color = args.depth_color
    color_intensity_clear = np.linalg.norm(pc_coords_filtered, axis=1) if depth_color else intensity_filtered
    min_color = 1.0 
    max_color = compounding_class.lidar.max_range  if depth_color else compounding_class.lidar.max_intensity
    logarithmic = depth_color
    canvas = plot_pointcloud_vispy(pc_coords_filtered, color_intensity=color_intensity_clear, lidar=compounding_class.lidar,\
                                    title="Clear PC", position=(100, 100),min_color=min_color, max_color=max_color, logarithmic=logarithmic) 
    color_intensity_sim = np.linalg.norm(pc_sim, axis=1) if depth_color else intensity_sim
    plot_title = f"{args.weather}_{args.noise} Sim PC" if args.weather != "" and args.noise != "" else (f"{args.weather} Sim PC" if args.weather != "" else f"{args.noise} Sim PC")   
    plot_pointcloud_vispy(pc_sim, color_intensity=color_intensity_sim, lidar=compounding_class.lidar,\
                           title=plot_title, position=(700, 100),min_color=min_color, max_color=max_color, logarithmic=logarithmic)
    canvas.app.run()

