import cv2
import os
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():

    log_save_path = 'results/table_logs_all/'

    data_root = 'results/Compare'
    ref_root = 'datasets/'
    ref_dirs = ['DIV2K/DIV2K_valid_HR/']
    datasets = ['DIV2K100']
    methods = ['LDL']

    logoverall_path = log_save_path + 'all_avgs/lpips_all_avgs.txt'

    for index in range(len(ref_dirs)):
        ref_dir = os.path.join(ref_root, ref_dirs[index])
        for method in methods:
            img_dir = os.path.join(data_root, method, datasets[index])

            img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

            os.makedirs(log_save_path, exist_ok=True)

            log_path = log_save_path + 'lpips__' + method + '__' + datasets[index] + '.txt'

            if not os.path.exists(log_path):

                loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
                lpips_all = []

                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
                for i, img_path in enumerate(img_list):
                    file_name = img_path.split('/')[-1]
                    if 'DIV2K100' in img_dir and 'SFTGAN' not in img_dir:
                        gt_path = os.path.join(ref_dir, file_name[:4] + '.png')
                    elif 'Urban100' in img_dir and 'SFTGAN' not in img_dir:
                        gt_path = os.path.join(ref_dir, file_name[:7] + '.png')
                    elif 'SFTGAN' in img_dir:
                        ref_dir_SFTGAN = 'results/Compare/SFTGAN_official/GT'
                        gt_path = os.path.join(ref_dir_SFTGAN, file_name.split('_')[0] + '_gt.png')
                        if 'Urban100' in img_dir:
                            gt_path = os.path.join(ref_dir_SFTGAN, file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_gt.png')
                    else:
                        if '_' in file_name:
                            gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '.png')
                        else:
                            gt_path = os.path.join(ref_dir, file_name)

                    img_restored = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                    img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

                    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
                    # norm to [-1, 1]
                    normalize(img_gt, mean, std, inplace=True)
                    normalize(img_restored, mean, std, inplace=True)

                    # calculate lpips
                    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
                    log = f'{i+1:3d}: {file_name:25}. \tLPIPS: {lpips_val.item():.6f}.'
                    with open(log_path, 'a') as f:
                        f.write(log + '\n')
                    print(log)
                    lpips_all.append(lpips_val.item())

                log = f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}'
                with open(log_path, 'a') as f:
                    f.write(log + '\n')
                log_overall = method + '__' + datasets[index] + '__' + log
                with open(logoverall_path, 'a') as f:
                    f.write(log_overall + '\n')
                print(log_overall)


if __name__ == '__main__':
    main()
