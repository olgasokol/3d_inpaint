import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
from datetime import datetime
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import read_args, read_depth_from_file, vis_data
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth_estimation, run_depth_estimation_new
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'))
if config['offscreen_rendering'] is True:
    vispy.use(app='egl')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)
sample_list = read_args(config['src_folder'], config['depth_folder'], config, config['specific'])
normal_canvas, all_canvas = None, None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"running on device {device}")

for idx in tqdm(range(len(sample_list))):
    depth = None
    sample = sample_list[idx]
    print("Current Source ==> ", sample['src_pair_name'])
    mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')

    if config['require_midas'] is True:
        print("Running depth extraction at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
        # estimate depth from 1 img 'ref_img_fi' and save it to 'depth_folder'
        # run_depth_estimation([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
        #                          config['MiDaS_model_ckpt'], MonoDepthNet)
        run_depth_estimation_new([sample['ref_img_fi']], config['depth_folder'])

    # rescale image
    if 'npy' in config['depth_format']:
        config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
    elif 'pfm' in config['depth_format']:
        config['output_h'], config['output_w'] = cv2.imread(sample['depth_fi']).shape[:2]
    else:
        config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']

    # load depth map (white is near, black is far)
    depth = read_depth_from_file(sample['depth_fi'], 1.0, config['output_h'], config['output_w'])

    image = imageio.imread(sample['ref_img_fi'])[:, :, :3]  # ignore alpha channel if there is one
    image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
    print(f"im {sample['src_pair_name']}, dim {(config['original_h'], config['original_w'])}")

    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False

    if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
        vis_data(depth, "depth_from_midas")
        depth_n = sparse_bilateral_filtering(depth.copy(), config, num_iter=config['sparse_iter'])
        # vis_data(depth_n-depth, "removed_features")
        vis_data(depth_n, "depth_after_filter")
        depth=depth_n
        # continue
        torch.cuda.empty_cache()
        print("Loading edge model at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
        depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(config['depth_edge_model_ckpt'], map_location=torch.device(device))
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.to(device)
        depth_edge_model.eval()

        print("Loading depth model at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(config['depth_feat_model_ckpt'], map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()

        print("Loading rgb model at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'], map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model = rgb_model.to(device)
        rgb_model.eval()
        graph = None


        print("Writing depth ply at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
        rt_info = write_ply(image,
                            depth,
                            sample['int_mtx'],
                            mesh_fi,
                            config,
                            rgb_model,
                            depth_edge_model,
                            depth_edge_model,
                            depth_feat_model)

        if rt_info is False:
            continue
        rgb_model = None
        color_feat_model = None
        depth_edge_model = None
        depth_feat_model = None
        torch.cuda.empty_cache()
    if config['save_ply'] is True or config['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info


    print("Making video at {}".format(datetime.fromtimestamp(time.time()).strftime(config['time_format'])))
    if (config['inference_video']):
        mean_loc_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
        top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
        left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
        down, right = top + config['output_h'], left + config['output_w']
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas = output_3d_photo(verts.copy(),
                                                    colors.copy(),
                                                    faces.copy(),
                                                    copy.deepcopy(Height),
                                                    copy.deepcopy(Width),
                                                    copy.deepcopy(hFov),
                                                    copy.deepcopy(vFov),
                                                    copy.deepcopy(sample['tgt_pose']),
                                                    sample['video_postfix'],
                                                    copy.deepcopy(sample['ref_pose']),
                                                    copy.deepcopy(config['video_folder']),
                                                    image.copy(),
                                                    copy.deepcopy(sample['int_mtx']),
                                                    config,
                                                    image,
                                                    videos_poses,
                                                    video_basename,
                                                    config.get('original_h'),
                                                    config.get('original_w'),
                                                    border=border,
                                                    depth=depth,
                                                    normal_canvas=normal_canvas,
                                                    all_canvas=all_canvas,
                                                    mean_loc_depth=mean_loc_depth)
