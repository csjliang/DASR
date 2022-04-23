import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels_Info
from basicsr.utils import img2tensor, tensor2img
from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, only_generate_gaussian_noise_pt, only_generate_poisson_noise_pt, add_given_gaussian_noise_pt, add_given_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop_return_indexes, paired_random_crop_by_indexes
from basicsr.models.srgan_dynamic_model import SRGANDynamicModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F

img_path = 'datasets/DIV2K/DIV2K_valid_HR'
output_path = 'datasets/RealSR_TEST/DIV2K_Real_type3'

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path+'_withparams', exist_ok=True)

# img_list = sorted(glob.glob(os.path.join(args.test_path, '*')))
# for img_path in img_list:
#     # read image
#     img_name = os.path.basename(img_path)
#     print(f'Processing {img_name} ...')
#     basename, ext = os.path.splitext(img_name)
#     input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

@torch.no_grad()
def degrade_func(img_dir, output_dir):

    img_list = sorted(glob.glob(os.path.join(img_dir, '*')))
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        # img_path = os.path.join(img_path, img_name)
        print(img_path)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = img2tensor(input_img / 255., bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

        resize_mode_list = ['area', 'bilinear', 'bicubic']
        jpeger = DiffJPEG(differentiable=False).cuda()

        scale_overall = 4

        # the first degradation process
        resize_prob = [0.1, 0.2, 0.7]  # up, down, keep
        resize_range = [0.85, 1.2]
        gaussian_noise_prob = 0.5
        noise_range = [1, 10]
        poisson_scale_range = [0.05, 0.5]
        gray_noise_prob = 0.4
        jpeg_range = [90, 95]

        # blur settings for the first degradation
        blur_kernel_size_minimum = 7
        blur_kernel_size = 21
        kernel_list = ['iso', 'aniso']
        kernel_prob = [0.65, 0.35]
        blur_sigma = [0.2, 0.8]
        betag_range = [0.5, 4]
        betap_range = [1, 2]
        sinc_prob = 0


        kernel_range = [2 * v + 1 for v in range(math.ceil(blur_kernel_size_minimum/2), math.ceil(blur_kernel_size/2))]  # kernel size ranges from 7 to 21
        pulse_tensor = torch.zeros(blur_kernel_size, blur_kernel_size).float()  # convolving with pulse tensor brings no blurry effect
        pulse_tensor[int(blur_kernel_size/2), int(blur_kernel_size/2)] = 1

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma, [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel1 = torch.FloatTensor(kernel).cuda()
        kernel_info['kernel'] = kernel1

        num_degradation_params = 4 * 2 + 2 # kernel
        num_degradation_params += 4 * 2 # resize
        num_degradation_params += 4 * 2 # noise
        num_degradation_params += 3 + 2 + 2 # jpeg
        degradation_params = torch.zeros(num_degradation_params)
        road_map = [0, 10, 10 + 8, 10 + 8 + 8, 10 + 8 + 8 + 7]

        kernel_size_range1 = [blur_kernel_size_minimum, blur_kernel_size]
        rotation_range = [-math.pi, math.pi]
        degradation_params[road_map[0]:road_map[0]+1] = (kernel_info['kernel_size'] - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
        degradation_params[road_map[0]+1:road_map[0]+2] = (kernel_info['sigma_x'] - blur_sigma[0]) / (blur_sigma[1] - blur_sigma[0])
        degradation_params[road_map[0]+2:road_map[0]+3] = (kernel_info['sigma_y'] - blur_sigma[0]) / (blur_sigma[1] - blur_sigma[0])
        degradation_params[road_map[0]+3:road_map[0]+4] = (kernel_info['rotation'] - rotation_range[0]) / (rotation_range[1] - rotation_range[0])

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(resize_mode_list)
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        normalized_scale = (scale - resize_range[0]) / (resize_range[1] - resize_range[0])
        onehot_mode = torch.zeros(len(resize_mode_list))
        for index, mode_current in enumerate(resize_mode_list):
            if mode_current == mode:
                onehot_mode[index] = 1
        degradation_params[road_map[1]:road_map[1] + 1] = torch.tensor(normalized_scale).expand(gt.size(0), 1)
        degradation_params[road_map[1] + 1:road_map[1] + 4] = onehot_mode.expand(gt.size(0), len(resize_mode_list))
        if np.random.uniform() < gaussian_noise_prob:
            sigma, gray_noise, out, noise_g_first = random_add_gaussian_noise_pt(
                out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)

            normalized_sigma = (sigma - noise_range[0]) / (noise_range[1] - noise_range[0])
            degradation_params[road_map[2]:road_map[2] + 1] = normalized_sigma.unsqueeze(1)
            degradation_params[road_map[2] + 1:road_map[2] + 2] = gray_noise.unsqueeze(1)
            degradation_params[road_map[2] + 2:road_map[2] + 4] = torch.tensor([1, 0]).expand(gt.size(0), 2)
        else:
            scale, gray_noise, out, noise_p_first = random_add_poisson_noise_pt(
                out, scale_range=poisson_scale_range, gray_prob=gray_noise_prob, clip=True, rounds=False)
            normalized_scale = (scale - poisson_scale_range[0]) / (poisson_scale_range[1] - poisson_scale_range[0])
            degradation_params[road_map[2]:road_map[2] + 1] = normalized_scale.unsqueeze(1)
            degradation_params[road_map[2] + 1:road_map[2] + 2] = gray_noise.unsqueeze(1)
            degradation_params[road_map[2] + 2:road_map[2] + 4] = torch.tensor([0, 1]).expand(gt.size(0), 2)

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
        normalized_jpeg_p = (jpeg_p - jpeg_range[0]) / (jpeg_range[1] - jpeg_range[0])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        degradation_params[road_map[3]:road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

        # resize back
        mode = random.choice(resize_mode_list)
        onehot_mode = torch.zeros(len(resize_mode_list))
        for index, mode_current in enumerate(resize_mode_list):
            if mode_current == mode:
                onehot_mode[index] = 1
        out = F.interpolate(out, size=(ori_h // scale_overall, ori_w // scale_overall), mode=mode)
        degradation_params[road_map[3] + 4:] = onehot_mode.expand(gt.size(0), len(resize_mode_list))

        print(degradation_params)

        # clamp and round
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        print(out.shape)
        out = tensor2img(out)
        target_name = img_name.split('.')[0] + '__'
        for value in degradation_params:
            target_name += str(value.numpy()) + '_'
        target_name += '_.png'
        print(os.path.join(output_dir, target_name))
        cv2.imwrite(os.path.join(output_dir+'_withparams', target_name), out)
        cv2.imwrite(os.path.join(output_dir, img_name), out)

degrade_func(img_path, output_path)