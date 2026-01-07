import ImageReward as RM
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import torch.fft
import matplotlib.pyplot as plt


def visualize_fft_phase(tensor, shift=True, color_wheel=True):
    """
    tensor: torch.Tensor of shape [b, c, h, w], real-valued
    Visualizes FFT phase with optional fftshift and color wheel.
    """
    # 1. 2D FFT
    fft_complex = torch.fft.fft2(tensor)

    # 2. fftshift (저주파 중앙 배치)
    if shift:
        fft_complex = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    # 3. Phase 계산: [-π, π]
    phase = torch.angle(fft_complex)  # [b, c, h, w]

    # 4. 시각화 준비
    b, c, h, w = phase.shape
    fig, axes = plt.subplots(b, c, figsize=(5 * c, 5 * b))
    if b == 1 and c == 1:
        axes = np.array([[axes]])
    elif b == 1:
        axes = np.array([axes])
    elif c == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)

    # 공통 컬러 범위
    vmin, vmax = -np.pi, np.pi

    for i in range(b):
        for j in range(c):
            phase_np = phase[i, j].cpu().numpy()
            ax = axes[i, j] if b > 1 or c > 1 else axes[0, 0]

            if color_wheel:
                # HSV 컬러휠: H = (phase + π)/(2π), S=1, V=1
                hsv = np.zeros((h, w, 3))
                hsv[..., 0] = (phase_np + np.pi) / (2 * np.pi)  # [0, 1]
                hsv[..., 1] = 1.0
                hsv[..., 2] = 1.0
                rgb = plt.cm.colors.hsv_to_rgb(hsv)
                im = ax.imshow(rgb)
            else:
                # Grayscale: [-π, π] → 회색조
                norm_phase = (phase_np + np.pi) / (2 * np.pi)  # [0, 1]
                im = ax.imshow(norm_phase, cmap='gray', vmin=0, vmax=1)

            title = f'B{i} C{j} - Phase'
            if shift:
                title += ' (shifted)'
            ax.set_title(title)
            ax.axis('off')

            # 컬러바 (grayscale일 때만)
            if not color_wheel:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 0.5, 1])
                cbar.set_ticklabels(['-π', '0', 'π'])

    plt.tight_layout()
    plt.show()


low = '../../../datasets/Deg_Scene/dark/NeRF_360/garden/images1/DSC08063.JPG'
high = '../../../datasets/Deg_Scene/dark/NeRF_360/garden/images/DSC07968.JPG'
# high1 = sorted(os.listdir(high))
# low1 = sorted(os.listdir(low))
# print(high, low)

import torchvision.transforms as transforms
tot = transforms.ToTensor()

# low_eg = tot(Image.open(os.path.join(low, low1[0])).convert('RGB')).unsqueeze(0).cuda()
low_eg = tot(Image.open(low).convert('RGB')).unsqueeze(0).cuda()
high_eg = tot(Image.open(high).convert('RGB')).unsqueeze(0).cuda()
visualize_fft_phase(high_eg)
# high_eg = tot(Image.open(os.path.join(high, low1[0])).convert('RGB')).unsqueeze(0).cuda()
print(low_eg.shape)
plt.subplot(121)
plt.imshow(high_eg[0].permute(1, 2, 0).cpu().detach().numpy())
plt.subplot(122)
plt.imshow(low_eg[0].permute(1, 2, 0).cpu().detach().numpy())
plt.show()
low_freq = torch.fft.fft2(low_eg, norm='backward')
high_freq = torch.fft.fft2(high_eg, norm='backward')
fft_shifted = torch.fft.fftshift(high_eg, dim=(-2, -1))
low_mag = torch.abs(fft_shifted)
low_phase = torch.angle(low_freq)
# print(low_mag.shape)
# print(low_phase.shape)

# log_mag = torch.fft.fftshift(low_mag)
# plt.imshow(torch.log1p(low_phase[0][0]).cpu().numpy())
plt.imshow((low_phase[0][0]).cpu().numpy())
plt.show()
high_mag = torch.abs(high_freq)
high_phase = torch.angle(high_freq)

new_spectrum = high_mag * torch.exp(1j * low_phase)
x_rec = torch.fft.ifft2(new_spectrum, norm='backward').real
# print(x_rec.shape)




