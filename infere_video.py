# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

from midas.model_loader import default_models, load_model

import torch
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

import open3d as o3d

first_execution = True

def visualize_pointcloud(viewer, image, depth, calib):
    rgb_image = o3d.geometry.Image(np.ascontiguousarray(image.astype(np.uint8)))
    depth_image = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1., depth_trunc=5, convert_rgb_to_intensity=False)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.intrinsic_matrix = calib.copy()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, np.pi, 0))
    pcd.rotate(R, center=(0, 0, 0))
    pcd.translate((0, 0, -3))

    o3d.io.write_point_cloud('PointCloud.pcd', pcd)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

    # viewer.update_geometry(pcd)
    viewer.clear_geometries()
    viewer.add_geometry(voxel_grid)
    viewer.poll_events()
    viewer.update_renderer()

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def test_simple(
        model_type,
        model_path,
        video_path,
        cuda=True
):
    """Function to predict for a single image or folder of images
    """
    assert model_type is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    K = np.eye(3, dtype=np.float32)
    K[0, 0] = 481.2
    K[1, 1] = -480
    K[0, 2] = 319.5
    K[1, 2] = 239.5

    # LOADING PRETRAINED MODEL
    model, transform, feed_width, feed_height = load_model(device, model_path, model_type, False)


    # LOAD VIDEO STREAM
    video_stream = cv2.VideoCapture(video_path)
    # video_stream = cv2.VideoCapture(0)
    success, frame = video_stream.read()
    input_image_pil = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_width, original_height = input_image_pil.size

    K_inv = np.linalg.inv(K)

    # VISUALIZATION
    show_plot = True
    show_pointcloud = False

    if show_plot:
        plt.ioff()
        fig, axs = plt.subplots(2, 2)
        input_plot = None
        input_masked_plot = None
        depth_plot = None

        plt.show(block=False)

    if show_pointcloud:
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.get_render_option().show_coordinate_frame = True    

    meshgrid = torch.meshgrid(torch.arange(0, original_height), torch.arange(0, original_width), indexing='ij')

    pixel_coord = torch.vstack((
        meshgrid[1].unsqueeze(0), 
        meshgrid[0].unsqueeze(0),
        torch.ones((1, original_height, original_width))
        )).to(device)
    
    points_without_depth = torch.tensordot(torch.tensor(K_inv).to(device), pixel_coord, ([1], [0]))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        while success:
            print("-> Predicting on new frame")
            original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
            image = transform({"image": original_image_rgb/255})["image"]

            # PREDICTION
            prediction = process(device, model, model_type, image, (feed_width, feed_height),
                                    original_image_rgb.shape[1::-1], False, True)
            
            pred_depth = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
            pred_depth = 1. / (pred_depth + 1e-5)
            pred_depth = 0.5 * pred_depth
            pred_depth += 0.2
            pred_depth[pred_depth > 15] = 15
            

            if show_plot:
                points_3d = points_without_depth * torch.tensor(pred_depth).to(device).unsqueeze(0).repeat(3, 1, 1)
                mask = (points_3d[1, :, :] > -0.2) * (points_3d[1, :, :] < 1) * (torch.abs(points_3d[2, :, :]) < 3)
                # mask = (points_3d[1, :, :] < 1) * (torch.abs(points_3d[2, :, :]) < 3)
                mask_3c = mask.unsqueeze(-1).repeat(1, 1, 3)

                masked_image = original_image_rgb * mask_3c.cpu().numpy()
                masked_depth = pred_depth * mask.cpu().numpy()

                avg_center = torch.mean(points_3d   [2, :, 220:420])
                print("Average depth value in range 220:320", avg_center.item())

                scatter_map2d = np.zeros((torch.count_nonzero(mask), 2))
                scatter_map2d[:, 0] = points_3d[0, mask].cpu().numpy()
                scatter_map2d[:, 1] = points_3d[2, mask].cpu().numpy()

                if input_plot is None:
                    input_plot = axs[0, 0].imshow(original_image_rgb)
                else:
                    input_plot.set_data(original_image_rgb)

                if input_masked_plot is None:
                    input_masked_plot = axs[0, 1].imshow(masked_image)
                else:
                    input_masked_plot.set_data(masked_image)

                if depth_plot is None:
                    depth_plot = axs[1, 0].imshow(pred_depth)
                else:
                    depth_plot.set_data(pred_depth)

                axs[1, 1].clear()
                axs[1, 1].hist2d(
                    scatter_map2d[:, 0], 
                    scatter_map2d[:, 1], 
                    bins=128, 
                    range=[[-1.5, 1.5], [0, 3]], 
                    density=False,
                    cmin=100,
                    cmax=None,
                    cmap='viridis')
                # axs[2].set_ylim([0, 3])
                # axs[2].set_xlim([-3, 3])

                fig.canvas.flush_events()
                plt.draw()

            if show_pointcloud:
                visualize_pointcloud(viewer, np.asarray(input_image_pil), pred_depth, K)

            success, frame = video_stream.read()
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

    print('-> Done!')

if __name__ == '__main__':
    test_simple('dpt_levit_224', 'weights/dpt_levit_224.pt', 'input/left_1.avi')
    # test_simple('RA-Depth', 'assets/left_1.avi')
    # test_simple('RA-Depth', 'assets/left_2.avi')
    # test_simple('RA-Depth', 'assets/left_3.avi')
    # test_simple('RA-Depth', 'assets/left_4.avi')
    # test_simple('RA-Depth', 'assets/left_5.avi')
    # test_simple('RA-Depth', '/home/davidesarraggiotto/Deer_GroundRobot.mp4')


#CUDA_VISIBLE_DEVICES=0 python test_simple.py --image_path /test/monodepth2-master/assets/test.png --model_name RA-Depth
