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

@torch.no_grad()
def degrade_func(img_dir, output_dir):

    img_list = sorted(glob.glob(os.path.join(img_dir, '*')))
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        print(img_path)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = img2tensor(input_img / 255., bgr2rgb=True, float32=True).unsqueeze(0).cuda().contiguous()

        resize_mode_list = ['area', 'bilinear', 'bicubic']
        jpeger = DiffJPEG(differentiable=False).cuda()

        scale_overall = 4

        # the first degradation process
        resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        resize_range = [0.15, 1.5]
        gaussian_noise_prob = 0.5
        noise_range = [1, 30]
        poisson_scale_range = [0.05, 3]
        gray_noise_prob = 0.4
        jpeg_range = [30, 95]

        # the second degradation process
        second_blur_prob = 0.8
        resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        resize_range2 = [0.3, 1.2]
        gaussian_noise_prob2 = 0.5
        noise_range2 = [1, 25]
        poisson_scale_range2 = [0.05, 2.5]
        gray_noise_prob2 = 0.4
        jpeg_range2 = [30, 95]

        # blur settings for the first degradation
        blur_kernel_size_minimum = 7
        blur_kernel_size = 21
        kernel_list = [ 'iso', 'aniso']
        kernel_prob = [ 0.65, 0.35]
        blur_sigma = [0.2, 3]
        betag_range = [0.5, 4]
        betap_range = [1, 2]
        sinc_prob = 0

        # blur settings for the second degradation
        kernel_list2 = [ 'iso', 'aniso' ]
        kernel_prob2 = [ 0.65, 0.35 ]
        blur_sigma2 = [0.2, 1.5]
        betag_range2 = [0.5, 4]
        betap_range2 = [1, 2]
        sinc_prob2 = 0

        # a final sinc filter
        final_sinc_prob = 0.8

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

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2_info = random_mixed_kernels_Info(
                kernel_list2,
                kernel_prob2,
                kernel_size,
                blur_sigma2,
                blur_sigma2, [-math.pi, math.pi],
                betag_range2,
                betap_range2,
                noise_range=None)
            kernel2 = kernel2_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < final_sinc_prob:
            kernel_size = random.choice(kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel).cuda()
            kernel_sinc_info = {'kernel':sinc_kernel, 'kernel_size':kernel_size, 'omega_c':omega_c}
        else:
            sinc_kernel = pulse_tensor.cuda()
            kernel_sinc_info = {'kernel': sinc_kernel, 'kernel_size': 0, 'omega_c': 0}

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel1 = torch.FloatTensor(kernel).cuda()
        kernel2 = torch.FloatTensor(kernel2).cuda()
        kernel_info['kernel'] = kernel1
        kernel2_info['kernel'] = kernel2

        # self.kernel1 = data['kernel1'].to(self.device)
        # self.kernel2 = data['kernel2'].to(self.device)
        # self.sinc_kernel = data['sinc_kernel'].to(self.device)

        # ori_h, ori_w = self.gt.size()[2:4]

        num_degradation_params = 4 * 2 + 2 # kernel
        num_degradation_params += 4 * 2 # resize
        num_degradation_params += 4 * 2 # noise
        num_degradation_params += 3 + 2 + 2 # jpeg
        degradation_params = torch.zeros(num_degradation_params)
        road_map = [0, 10, 10 + 8, 10 + 8 + 8, 10 + 8 + 8 + 7]

        # {'kernel': kernel, 'kernel_size': kernel_size, 'sigma_x': sigma_x, 'sigma_y': sigma_y, 'rotation': rotation}
        kernel_size_range1 = [blur_kernel_size_minimum, blur_kernel_size]
        kernel_size_range2 = [blur_kernel_size_minimum, blur_kernel_size]
        rotation_range = [-math.pi, math.pi]
        omega_c_range = [np.pi / 3, np.pi]
        degradation_params[road_map[0]:road_map[0]+1] = (kernel_info['kernel_size'] - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
        degradation_params[road_map[0]+4:road_map[0]+5] = (kernel2_info['kernel_size'] - kernel_size_range2[0]) / (kernel_size_range2[1] - kernel_size_range2[0])
        degradation_params[road_map[0]+1:road_map[0]+2] = (kernel_info['sigma_x'] - blur_sigma[0]) / (blur_sigma[1] - blur_sigma[0])
        degradation_params[road_map[0]+5:road_map[0]+6] = (kernel2_info['sigma_x'] - blur_sigma2[0]) / (blur_sigma2[1] - blur_sigma2[0])
        degradation_params[road_map[0]+2:road_map[0]+3] = (kernel_info['sigma_y'] - blur_sigma[0]) / (blur_sigma[1] - blur_sigma[0])
        degradation_params[road_map[0]+6:road_map[0]+7] = (kernel2_info['sigma_y'] - blur_sigma2[0]) / (blur_sigma2[1] - blur_sigma2[0])
        degradation_params[road_map[0]+3:road_map[0]+4] = (kernel_info['rotation'] - rotation_range[0]) / (rotation_range[1] - rotation_range[0])
        degradation_params[road_map[0]+7:road_map[0]+8] = (kernel2_info['rotation'] - rotation_range[0]) / (rotation_range[1] - rotation_range[0])
        degradation_params[road_map[0]+8:road_map[0]+9] = (kernel_sinc_info['kernel_size'] - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
        degradation_params[road_map[0]+9:road_map[1]] = (kernel_sinc_info['omega_c'] - omega_c_range[0]) / (omega_c_range[1] - omega_c_range[0])

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
            noise_p_first = only_generate_poisson_noise_pt(out, scale_range=poisson_scale_range, gray_prob=gray_noise_prob)
        else:
            scale, gray_noise, out, noise_p_first = random_add_poisson_noise_pt(
                out, scale_range=poisson_scale_range, gray_prob=gray_noise_prob, clip=True, rounds=False)
            normalized_scale = (scale - poisson_scale_range[0]) / (poisson_scale_range[1] - poisson_scale_range[0])
            degradation_params[road_map[2]:road_map[2] + 1] = normalized_scale.unsqueeze(1)
            degradation_params[road_map[2] + 1:road_map[2] + 2] = gray_noise.unsqueeze(1)
            degradation_params[road_map[2] + 2:road_map[2] + 4] = torch.tensor([0, 1]).expand(gt.size(0), 2)
            noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=noise_range, gray_prob=gray_noise_prob)

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
        normalized_jpeg_p = (jpeg_p - jpeg_range[0]) / (jpeg_range[1] - jpeg_range[0])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        degradation_params[road_map[3]:road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < second_blur_prob:
            out = filter2D(out, kernel2)
            degradation_params[road_map[1] - 1:road_map[1]] = torch.tensor([1]).expand(gt.size(0), 1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(resize_mode_list)
        out = F.interpolate(out, size=(int(ori_h / scale_overall * scale), int(ori_w / scale_overall * scale)), mode=mode)
        normalized_scale = (scale - resize_range2[0]) / (resize_range2[1] - resize_range2[0])
        onehot_mode = torch.zeros(len(resize_mode_list))
        for index, mode_current in enumerate(resize_mode_list):
            if mode_current == mode:
                onehot_mode[index] = 1
        degradation_params[road_map[1] + 4:road_map[1] + 5] = torch.tensor(normalized_scale).expand(gt.size(0), 1)
        degradation_params[road_map[1] + 5:road_map[2]] = onehot_mode.expand(gt.size(0), len(resize_mode_list))
        # noise
        if np.random.uniform() < gaussian_noise_prob2:
            sigma, gray_noise, out, noise_g_second = random_add_gaussian_noise_pt(
                out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob2)
            normalized_sigma = (sigma - noise_range2[0]) / (noise_range2[1] - noise_range2[0])
            degradation_params[road_map[2] + 4:road_map[2] + 5] = normalized_sigma.unsqueeze(1)
            degradation_params[road_map[2] + 5:road_map[2] + 6] = gray_noise.unsqueeze(1)
            degradation_params[road_map[2] + 6:road_map[3]] = torch.tensor([1, 0]).expand(gt.size(0), 2)
            noise_p_second = only_generate_poisson_noise_pt(out, scale_range=poisson_scale_range2, gray_prob=gray_noise_prob2)
        else:
            scale, gray_noise, out, noise_p_second = random_add_poisson_noise_pt(
                out,
                scale_range=poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
            normalized_scale = (scale - poisson_scale_range2[0]) / (poisson_scale_range2[1] - poisson_scale_range2[0])
            degradation_params[road_map[2] + 4:road_map[2] + 5] = normalized_scale.unsqueeze(1)
            degradation_params[road_map[2] + 5:road_map[2] + 6] = gray_noise.unsqueeze(1)
            degradation_params[road_map[2] + 6:road_map[3]] = torch.tensor([0, 1]).expand(gt.size(0), 2)
            noise_g_second = only_generate_gaussian_noise_pt(out, sigma_range=noise_range2, gray_prob=gray_noise_prob2)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(resize_mode_list)
            onehot_mode = torch.zeros(len(resize_mode_list))
            for index, mode_current in enumerate(resize_mode_list):
                if mode_current == mode:
                    onehot_mode[index] = 1
            out = F.interpolate(out, size=(ori_h // scale_overall, ori_w // scale_overall), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
            normalized_jpeg_p = (jpeg_p - jpeg_range2[0]) / (jpeg_range2[1] - jpeg_range2[0])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            degradation_params[road_map[3] + 1:road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
            degradation_params[road_map[3] + 2:road_map[3] + 4] = torch.tensor([1, 0]).expand(gt.size(0), 2)
            degradation_params[road_map[3] + 4:] = onehot_mode.expand(gt.size(0), len(resize_mode_list))
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
            normalized_jpeg_p = (jpeg_p - jpeg_range2[0]) / (jpeg_range2[1] - jpeg_range2[0])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(resize_mode_list)
            onehot_mode = torch.zeros(len(resize_mode_list))
            for index, mode_current in enumerate(resize_mode_list):
                if mode_current == mode:
                    onehot_mode[index] = 1
            out = F.interpolate(out, size=(ori_h // scale_overall, ori_w // scale_overall), mode=mode)
            out = filter2D(out, sinc_kernel)
            degradation_params[road_map[3] + 1:road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
            degradation_params[road_map[3] + 2:road_map[3] + 4] = torch.tensor([0, 1]).expand(gt.size(0), 2)
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
        cv2.imwrite(os.path.join(output_dir + '_withparams', target_name), out)
        cv2.imwrite(os.path.join(output_dir, img_name), out)

degrade_func(img_path, output_path)