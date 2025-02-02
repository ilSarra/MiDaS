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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet50
from segmentation.utils import preprocess_input as seg_preprocess

import torch
from torchvision import transforms, datasets

from scipy.optimize import minimize
from scipy.linalg import svd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import open3d as o3d

import warnings
warnings.filterwarnings('ignore')


first_execution = True

def estimate_scale(points, h):
    x = points[0]
    y = points[1]
    z = points[2]

    def scale_est_cost(a):
        # cost = np.median(np.abs(a[0] * points[0] + a[1] * points[1] + a[2] * points[2] + a[3]))
        cost = a[0] * x + a[1] * z + a[2] - y
        cost = np.abs(cost)
        cost = np.median(cost)
        return cost
        
    initial_a = [0, 0, -h]
    res = minimize(scale_est_cost, initial_a, tol=1e-3)

    scale = -res.x[-1] / h
    # scale = -h / res.x[-1]
    print(res.x)

    return scale


def estimate_scale_svd(points, h):
    xyz = np.moveaxis(points, 0, -1)
    np.random.shuffle(xyz)
    xyz = xyz[:1000]

    centroid = xyz.mean(axis=0)
    # xyzT = np.transpose(xyz)
    xyzR = xyz - centroid
    # xyzRT = np.transpose(xyzR)
    
    u, sigma, v = np.linalg.svd(xyzR)
    normal = v[2]
    normal /= np.linalg.norm(normal)

    b = np.dot(normal, centroid)
    scale = b / h

    show = False

    if show:
        forGraphs = list()
        forGraphs.append(np.array([centroid[0],centroid[1],centroid[2],normal[0],normal[1], normal[2]]))

        xyz = np.transpose(xyz)

        # create x,y for display
        minPlane = int(np.floor(min(min(xyz[0]), min(xyz[1]), min(xyz[2]))))
        maxPlane = int(np.ceil(max(max(xyz[0]), max(xyz[1]), max(xyz[2]))))
        xx, yy = np.meshgrid(range(minPlane,maxPlane), range(minPlane,maxPlane))

        # calculate corresponding z for display
        z = (-normal[0] * xx - normal[1] * yy + b) * 1. /normal[2]

        #matplotlib display code
        forGraphs = np.asarray(forGraphs)
        X, Y, Z, U, V, W = zip(*forGraphs)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, z, alpha=0.2)
        ax.scatter(xyz[0],xyz[1],xyz[2])
        ax.quiver(X, Y, Z, U, V, W)
        ax.set_xlim([min(xyz[0])- 0.1, max(xyz[0]) + 0.1])
        ax.set_ylim([min(xyz[1])- 0.1, max(xyz[1]) + 0.1])
        ax.set_zlim([min(xyz[2])- 0.1, max(xyz[2]) + 0.1])   
        plt.show() 

    return np.abs(scale)

def visualize_pointcloud(viewer, image, depth, calib):
    rgb_image = o3d.geometry.Image(np.ascontiguousarray(image.astype(np.uint8)))
    depth_image = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1., depth_trunc=5, convert_rgb_to_intensity=False)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.intrinsic_matrix = calib.copy()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    # R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, np.pi, 0))
    # pcd.rotate(R, center=(0, 0, 0))
    # pcd.translate((0, 0, -3))

    o3d.io.write_point_cloud('PointCloud.pcd', pcd)
    
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

    # viewer.update_geometry(pcd)
    viewer.clear_geometries()
    viewer.add_geometry(pcd)
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
    depth_model, transform, feed_width, feed_height = load_model(device, model_path, model_type, False)
    seg_model = model = deeplabv3_resnet50(pretrained=True, progress=True)
    seg_model.classifier = DeepLabHead(2048, 2)
    seg_weights = torch.load('weights/deeplabv3_6.pt')
    seg_model.load_state_dict(seg_weights)

    seg_model.eval()

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

    prev_hist2d = None
    scale = -1

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        while success:
            # print("-> Predicting on new frame")
            original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
            image = transform({"image": original_image_rgb/255})["image"]

            # PREDICTION
            disp = process(device, depth_model, model_type, image, (feed_width, feed_height),
                                    original_image_rgb.shape[1::-1], False, True)
            
            seg_input = seg_preprocess(frame, [640, 480]).unsqueeze(0)
            segmentation = seg_model(seg_input)
            grass_mask = torch.argmin(segmentation['out'][0], 0).bool().cuda()
            
            # pred_depth = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
            # pred_depth = 1. / (pred_depth + 1e-5)
            # shifted_disparity = prediction - np.min(prediction)
            # pred_depth = 1/ (shifted_disparity + 1e-5)
            # pred_depth -= np.min(pred_depth)
            # pred_depth = 0.6 * pred_depth
            # pred_depth += 0.3

            # Estimated with equations
            # a = 2000
            # b = 0
            # pred_depth = a * pred_depth + b
            # pred_depth[pred_depth > 15] = 15
            # pred_depth[pred_depth < 0] = torch.inf
            # pred_depth[pred_depth < 0.3] = 0.3

            # pred_depth -= np.min(pred_depth)
            # pred_depth *= 10
            # pred_depth += 0.3

            #########################

            max_d = 1e4
            min_d = 0.0

            # abs_disp = np.abs(prediction)
            # pred_depth = 1 / (abs_disp + 1e-5)
            # pred_depth = (pred_depth - np.min(pred_depth)) / (np.max(pred_depth) - np.min(pred_depth))
            # pred_depth = pred_depth * (max_d - min_d) + min_d

            pred_depth = 1 / np.abs(disp)
            pred_depth -= np.min(pred_depth)
            pred_depth = pred_depth * (max_d - min_d) + min_d

            points_3d_raw = points_without_depth * torch.tensor(pred_depth).to(device).unsqueeze(0).repeat(3, 1, 1)
            
            # grass_mask = (points_3d_raw[1] < -0.1) * (points_3d_raw[2] < 3)
            grass_points = points_3d_raw[:, grass_mask]

            grass_image = grass_mask.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy() * original_image_rgb

            if scale < 0:
                scale = estimate_scale_svd(grass_points.detach().cpu().numpy(), 0.2)

            elif grass_points.shape[-1] > 1e3:
                a = 0.9
                estimated_scale = estimate_scale_svd(grass_points.detach().cpu().numpy(), 0.2)

                if estimated_scale > 0:
                    scale = a * scale + (1 - a) * estimated_scale

                print("estimated scale", estimated_scale, "updated scale", scale)
            

            if scale > 0:
                pred_depth = scale * pred_depth

            pred_depth[pred_depth > 15] = 15
            pred_depth[pred_depth < 0] = 15

            del points_3d_raw
            points_3d = points_without_depth * torch.tensor(pred_depth).to(device).unsqueeze(0).repeat(3, 1, 1)

            if show_plot:
                mask = ~grass_mask * (points_3d[1] < 1) * (points_3d[2] < 3)
                # mask = (points_3d[1, :, :] < 1) * (torch.abs(points_3d[2, :, :]) < 3)
                mask_3c = mask.unsqueeze(-1).repeat(1, 1, 3)

                masked_image = original_image_rgb * mask_3c.cpu().numpy()
                # masked_depth = pred_depth * mask.cpu().numpy()

                # avg_center = torch.mean(points_3d   [2, :, 220:420])
                # print("Average depth value in range 220:320", avg_center.item())

                scatter_map2d = np.zeros((torch.count_nonzero(mask), 2))
                scatter_map2d[:, 0] = points_3d[0, mask].cpu().numpy()
                scatter_map2d[:, 1] = points_3d[2, mask].cpu().numpy()

                if input_plot is None:
                    input_plot = axs[0, 0].imshow(masked_image)
                else:
                    input_plot.set_data(masked_image)

                if input_masked_plot is None:
                    input_masked_plot = axs[0, 1].imshow(grass_image)
                else:
                    input_masked_plot.set_data(grass_image)

                if depth_plot is None:
                    depth_plot = axs[1, 0].imshow(pred_depth)
                else:
                    depth_plot.set_data(pred_depth)

                axs[1, 1].clear()
                # hist2d_out = axs[1, 1].hist2d(
                #     scatter_map2d[:, 0], 
                #     scatter_map2d[:, 1], 
                #     bins=128, 
                #     range=[[-1.5, 1.5], [0, 3]], 
                #     density=True,
                #     cmin=5,
                #     cmax=None,
                #     cmap='viridis')

                hist_2d = np.histogram2d(
                    scatter_map2d[:, 0], 
                    scatter_map2d[:, 1], 
                    bins=128, 
                    range=[[-1.5, 1.5], [0, 3]], 
                    density=True)
                
                hist_2d = np.moveaxis(hist_2d[0], 0, 1)
                hist_2d = np.flip(hist_2d, 0)
                hist_2d[np.isnan(hist_2d)] = 0

                # if prev_hist2d is not None:
                #     # Translate previous hist2d
                #     bin_in_meters = 3 / 128
                #     translation = int(0.3 / bin_in_meters)

                #     prev_hist2d_translated = np.zeros((128, 128))
                #     prev_hist2d_translated[translation:,:] = prev_hist2d[:(128 - translation), :]

                #     hist_2d = 0.9 * hist_2d + 0.1 * prev_hist2d_translated
                #     prev_hist2d = hist_2d

                # else:
                    # prev_hist2d = hist_2d

                hist_2d[hist_2d < 5] = 0

                axs[1, 1].imshow(hist_2d, interpolation=None, extent=[-1.5, 1.5, 0, 3])

                if (hist_2d > 0).any():
                    closest = 0
                    furthest = hist_2d.shape[0] - 1
                    leftmost = 0
                    rightmost = hist_2d.shape[1] - 1

                    while (hist_2d[closest] == 0).all():
                        closest += 1
                    while (hist_2d[furthest] == 0).all():
                        furthest -= 1
                    while (hist_2d[:, leftmost] == 0).all():
                        leftmost += 1
                    while (hist_2d[:, rightmost] == 0).all():
                        rightmost -= 1

                    leftmost = leftmost * 3 / hist_2d.shape[0] - 1.5
                    rightmost = rightmost * 3 / hist_2d.shape[0] - 1.5
                    furthest = 3 - furthest * 3 / hist_2d.shape[0]
                    closest = 3 - closest * 3 / hist_2d.shape[0]

                    rect = patches.Rectangle((leftmost, closest), rightmost - leftmost, furthest - closest, linewidth=1, edgecolor='r', facecolor='none')
                    axs[1, 1].add_patch(rect)
                    # print('Found bounding box is ({}, {}), ({}, {})'.format(leftmost, furthest, rightmost, closest))
                
                # else:
                    # print('No obstacle found')
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
    test_simple('midas_v21_384', 'weights/midas_v21_384.pt', 'input/left_0.avi')
    test_simple('dpt_swin2_tiny_256', 'weights/dpt_swin2_tiny_256.pt', 'input/left_1.avi')
    test_simple('dpt_swin2_tiny_256', 'weights/dpt_swin2_tiny_256.pt', 'input/left_2.avi')
    test_simple('dpt_swin2_tiny_256', 'weights/dpt_swin2_tiny_256.pt', 'input/left_3.avi')
    test_simple('dpt_swin2_tiny_256', 'weights/dpt_swin2_tiny_256.pt', 'input/left_4.avi')
    test_simple('dpt_swin2_tiny_256', 'weights/dpt_swin2_tiny_256.pt', 'input/left_5.avi')


#CUDA_VISIBLE_DEVICES=0 python test_simple.py --image_path /test/monodepth2-master/assets/test.png --model_name RA-Depth
