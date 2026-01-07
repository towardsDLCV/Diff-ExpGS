#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import os
import tqdm
import numpy as np
import imageio.v2 as iio
from torchvision.models.vgg import vgg16
import math

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def depth_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.png")

def codebook_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.npy")

def gather_todo_list(depth_root, cameras, force_rerun=False):
    # Gather list of camera to estimate depth
    todo_indices = []
    for i, cam in enumerate(cameras):
        if not os.path.exists(depth_path(depth_root, cam)) or force_rerun:
            todo_indices.append(i)
    return todo_indices

def save_quantize_depth(depth_root, cam, depth):
    # Quantize depth map to 16 bit
    codebook = depth.quantile(torch.linspace(0, 1, 65536).cuda(), interpolation='nearest')
    depth_idx = torch.searchsorted(codebook, depth, side='right').clamp_max_(65535)
    depth_idx[(depth - codebook[depth_idx-1]).abs() < (depth - codebook[depth_idx]).abs()] -= 1
    assert depth_idx.max() <= 65535
    assert depth_idx.min() >= 0

    # Save result
    depth_np = depth_idx.cpu().numpy().astype(np.uint16)
    iio.imwrite(depth_path(depth_root, cam), depth_np)
    np.save(codebook_path(depth_root, cam), codebook.cpu().detach().numpy().astype(np.float32))

def load_depth_to_camera(depth_root, cameras, depth_name):
    for cam in tqdm.tqdm(cameras):
        depth_np = iio.imread(depth_path(depth_root, cam))
        codebook = np.load(codebook_path(depth_root, cam))
        setattr(cam, depth_name, torch.tensor(codebook[depth_np]))

class DepthAnythingv2Loss:

    def __init__(self, iter_from, iter_end, end_mult,
                 source_path, cameras):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

        depth_root = os.path.join(source_path, "mono_priors", "depthanythingv2")
        os.makedirs(depth_root, exist_ok=True)

        todo_indices = gather_todo_list(depth_root, cameras, force_rerun=False)

        if len(todo_indices):
            print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")

            # Load model
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()

        for i in tqdm.tqdm(todo_indices):
            cam = cameras[i]

            # Inference depth
            inputs = image_processor(images=cam.original_image, return_tensors="pt", do_rescale=False)
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
            outputs = model(**inputs)
            depth = outputs['predicted_depth'].squeeze()

            save_quantize_depth(depth_root, cam, depth)

        print("Load the estimated depths to cameras.")
        load_depth_to_camera(depth_root, cameras, 'depthanythingv2')

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0
        # print(render_pkg['depth'].shape)
        invdepth = 1 / render_pkg['depth'].repeat(3, 1, 1).clamp_min(cam.znear)
        alpha = (1 - render_pkg['raw_T'][None, None])
        mono = cam.depthanythingv2.cuda()
        mono = mono[None,None]

        if invdepth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono, size=invdepth.shape[-2:], mode='bilinear')

        X, _, Xref = invdepth.split(1)
        X = X * alpha
        Y = mono

        with torch.no_grad():
            Ymed = Y.median()
            Ys = (Y - Ymed).abs().mean()
            Xmed = Xref.median()
            Xs = (Xref - Xmed).abs().mean()
            target = (Y - Ymed) * (Xs/Ys) + Xmed

        mask = (target > 0.01) & (alpha > 0.5)
        X = X * mask
        target = target * mask
        loss = l2_loss(X, target)

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss



class MapRegularizationLoss:
    def __init__(self, iter=0):
        self.iter = iter
        # self.iter_from = iter_from
        # self.iter_end = iter_end
        # self.end_mult = end_mult

    def sl1_loss(self, input, target):
        return F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)

    def compute_dark_channel_prior(self, image, window_size=15):
        # image: (B, C, H, W)
        padding = window_size // 2
        # 경계 처리를 위한 패딩 (reflect 모드 사용)
        padded_image = F.pad(image, (padding, padding, padding, padding), mode='reflect')

        # 채널별 최솟값 계산 (RGB 중 최솟값)
        min_channel = torch.min(padded_image, dim=1, keepdim=True)[0]  # (B, 1, H+2*padding, W+2*padding)

        # 국부 윈도우 내 최솟값 계산 (min pooling)
        # PyTorch의 max_pool2d를 사용하여 음수 값에 대한 min pooling을 구현
        # padding=0으로 설정하면, 이미 추가된 padding만큼 자동으로 원본 해상도로 맞춰짐
        dark_channel = -F.max_pool2d(-min_channel, kernel_size=window_size, stride=1, padding=0)

        return dark_channel  # (B, 1, H, W)

    # Box Filter for L_Tsmooth (Eqn. 26)
    def box_filter(self, image, kernel_size=3):
        # image: (B, C, H, W)
        # Using average pooling for box filter
        padding = kernel_size // 2
        return F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=padding)

    def l_dcp(self, T_hat, I):
        # L_DCP를 위한 T_DCP는 1x1 윈도우로 계산 (논문에서 명시)
        A_max_for_dcp_loss = I.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        I_normalized_by_A_max = I / (A_max_for_dcp_loss + 1e-6)
        T_DCP_for_loss = 1 - self.compute_dark_channel_prior(I_normalized_by_A_max, window_size=1)

        # T_DCP를 곱하여 밝은 장면에서의 가중치를 낮춤
        return self.sl1_loss((T_hat - T_DCP_for_loss) * T_DCP_for_loss, torch.zeros_like(T_hat))

    # L_Tgray (전송 맵 회색도 손실) - Eqn. 25
    # T_hat: 추정된 전송 맵
    def l_tgray(self, T_hat):
        # 채널별 평균 (회색도로 만들기 위함)
        mean_T_hat_channels = torch.mean(T_hat, dim=1, keepdim=True)
        return self.sl1_loss(T_hat, mean_T_hat_channels)

    def l_dcprec(self, I, J_hat):
        # A_max (I의 각 채널의 최대 픽셀 값)
        A_max = I.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # (B, C, 1, 1)

        # T_DCP 계산 (일반적인 윈도우 크기, 예: 15x15 사용)
        I_normalized_by_A_max = I / (A_max + 1e-6)  # 0으로 나누는 것을 방지하기 위한 엡실론 추가
        # print(I_normalized_by_A_max.shape)
        T_DCP_val = 1 - self.compute_dark_channel_prior(I_normalized_by_A_max, window_size=15)  # (B, 1, H, W)
        # print(J_hat.shape, T_DCP_val.shape, A_max.shape)
        reconstructed_I = J_hat * T_DCP_val + A_max * (1 - T_DCP_val)
        return self.sl1_loss(I, reconstructed_I)

    def l_arec(self, A_A_hat, A_hat, T_A_hat):
        loss_A = self.sl1_loss(A_A_hat, A_hat)
        loss_T = self.sl1_loss(T_A_hat, torch.zeros_like(T_A_hat))  # 전송 맵은 0이어야 함
        return loss_A + loss_T

    # L_Tsmooth (전송 맵 평활화 손실) - Eqn. 26
    # T_hat: 추정된 전송 맵
    def l_tsmooth(self, T_hat):
        # 박스 필터를 이용한 평활화 (커널 크기 3x3)
        smoothed_T_hat = self.box_filter(T_hat, kernel_size=3)
        return self.sl1_loss(T_hat, smoothed_T_hat)

    def __call__(self, L, I):
        L = L.unsqueeze(0)
        I = I.unsqueeze(0)
        loss_dcp = self.l_dcp(L, I)
        loss_tgray = self.l_tgray(L)
        loss_tsmooth = self.l_tsmooth(L)
        loss_dcprec = self.l_dcprec(I, L)

        total_loss = loss_dcp + loss_tgray + loss_tsmooth + loss_dcprec
        return total_loss


class ExposureLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.5):
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        # x는 보정된 이미지 (예: rlist의 마지막 원소)
        mean_patches = self.pool(x)
        return torch.mean((mean_patches - self.mean_val) ** 2)


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, illu):
        Fidelity_Loss = self.l2_loss(illu, input)
        Smooth_Loss = self.smooth_loss(input, illu)
        return 1.5*Fidelity_Loss + Smooth_Loss


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


def soft_histogram(x01, bins=64, sigma=0.02):
    # x01: (B,1,H,W) in [0,1]
    B = x01.shape[0]
    centers = torch.linspace(0, 1, bins, device=x01.device)[None, :, None, None]
    w = torch.exp(-0.5 * ((x01 - centers) / sigma)**2)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    hist = w.sum(dim=(2,3))                                 # (B, bins)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    cdf  = torch.cumsum(hist, dim=1)
    return hist, cdf

def cdf_loss_weighted(out_img, target='uniform', bins=256, sigma=0.02, hi_weight=3.0):
    # out_img: (B,3,H,W) in [0,1]
    Y = (0.2126*out_img[:,0:1] + 0.7152*out_img[:,1:2] + 0.0722*out_img[:,2:3]).clamp(0,1)
    _, cdf_out = soft_histogram(Y, bins=bins, sigma=sigma)
    if target == 'uniform':
        cdf_tgt = torch.linspace(0, 1, bins, device=Y.device)[None, :].expand_as(cdf_out)
    else:
        cdf_tgt = target  # (B,bins) 또는 (1,bins)

    w = torch.linspace(1.0, hi_weight, bins, device=Y.device)
    w = w / w.mean()
    return ((cdf_out - cdf_tgt).abs() * w).mean()


def tail_contrast_loss(img, top_p=0.1, mid_lo=0.4, mid_hi=0.6, margin=0.08):
    """
    img: (B,3,H,W) in [0,1]
    top_p:   상위 퍼센타일 비율 (예: 0.05 → 상위 5%)
    mid_lo, mid_hi: 중간 톤 대역의 분위수 범위
    margin:  상위 평균 - 중간 평균 >= margin 이 되도록 유도 (부족하면 페널티)

    반환: 스칼라 loss
    """
    with torch.no_grad():
        # 휘도 추출
        Y = (0.2126*img[:,0:1] + 0.7152*img[:,1:2] + 0.0722*img[:,2:3]).clamp(0,1)
    B, _, H, W = Y.shape
    y = Y.view(B, -1)                                       # (B, HW)
    k_top = max(1, int(y.shape[1] * top_p))

    # 정렬 인덱스
    vals, idx = torch.sort(y, dim=1)                        # 오름차순
    # 상위 top_p 평균
    top_mean = vals[:, -k_top:].mean(dim=1)

    # 중간 톤 대역 평균(45~55% 구간)
    lo = int(y.shape[1] * mid_lo)
    hi = max(lo+1, int(y.shape[1] * mid_hi))
    mid_mean = vals[:, lo:hi].mean(dim=1)

    contrast = top_mean - mid_mean                          # (B,)
    loss = torch.relu(margin - contrast).mean()             # 부족분만 벌점
    return loss


if __name__ == '__main__':
    loss = tail_contrast_loss
    a  = torch.rand(1, 3, 256, 256)
    b = torch.randn(3, 256, 256)

    loss = loss(a)
    print(loss)




