import os
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
import sys
import uuid
import cv2
import random
import lpips
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything

from random import randint

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.dataloader import CameraDataset
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr

from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from render import render_sets
from metrics import evaluate

from network import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianTrainer:

    def __init__(self):
        super(GaussianTrainer, self).__init__()
        self.ref_view_num = 3

        self.sd_locked = True
        self.only_mid_control = False
        self.strength = 1.0

        self.num_samples = 1
        self.scale = 9.0
        self.diffusion_steps = 10
        self.eta = 0.0
        self.guess_mode = False

        sd_model = create_model('./resources/models/cldm_v15.yaml').cpu()
        state_dict = load_state_dict('./resources/models/control_sd15_ini-001.ckpt', location='cpu')

        new_state_dict = {}
        for s in state_dict:
            if "cond_stage_model.transformer" not in s:
                new_state_dict[s] = state_dict[s]
        sd_model.load_state_dict(new_state_dict)
        sd_model.add_new_layers()

        state_dict = load_state_dict('./resources/checkpoints/COCO-final.ckpt', location='cpu')
        new_state_dict = {}
        for sd_name, sd_param in state_dict.items():
            if '_forward_module.control_model' in sd_name:
                new_state_dict[sd_name.replace('_forward_module.control_model.', '')] = sd_param
        sd_model.control_model.load_state_dict(new_state_dict)
        sd_model.change_first_stage('./resources/checkpoints/main-epoch=00-step=7000.ckpt')

        self.sd_model = sd_model.to(device)
        self.diffusion_sampler = DPMSolverSampler(self.sd_model)
        self.to_tensor = transforms.ToTensor()

    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y

    def resize_image(self, input_image, resolution):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def process_ref_imgae(self, imgs):
        ref_list = []
        for img in imgs:
            img = cv2.cvtColor((img * 255.0).permute(1, 2, 0).cpu().detach().clip(0, 255).numpy().astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
            detected_map = self.resize_image(self.HWC3(img), 512)
            control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
            ref_list.append(control)
        ref_list = torch.stack(ref_list, dim=0)
        ref_list = einops.rearrange(ref_list, 'b h w c -> b c h w').clone()
        return ref_list

    def process_image(self, img, ref_img):
        img = cv2.cvtColor((img * 255.0).permute(1, 2, 0).cpu().detach().clip(0, 255).numpy().astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        h1, w1, _ = img.shape
        detected_map = self.resize_image(self.HWC3(img), 512)
        H, W, C = detected_map.shape
        control = torch.from_numpy(detected_map.copy()).cuda() / 255.0

        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if ref_img is not None:
            ref_imgs = self.process_ref_imgae(ref_img)
            ref_z0 = self.sd_model.encode_first_stage(ref_imgs *2 -1)[0]
            ref_z0 = ref_z0.mode()
        else:
            ref_imgs = None
            ref_z0 = None

        ae_hs = self.sd_model.encode_first_stage(control * 2 - 1)[1]
        seed_everything(0)

        cond = {"c_concat": [control],
                "c_crossattn": [self.sd_model.get_unconditional_conditioning(self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control],
                   "c_crossattn": [self.sd_model.get_unconditional_conditioning(self.num_samples)]}
        shape = (4, H // 8, W // 8)

        self.sd_model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else (
                [self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.diffusion_sampler.sample(self.diffusion_steps, self.num_samples,
                                                               shape, cond, verbose=False, eta=self.eta,
                                                               unconditional_guidance_scale=self.scale,
                                                               unconditional_conditioning=un_cond,
                                                               dmp_order=3,
                                                               ref=ref_imgs,
                                                               ref_z0=ref_z0)
        x_samples = self.sd_model.decode_new_first_stage(samples, ae_hs)
        x_samples = F.interpolate(x_samples, size=(h1, w1), mode='bilinear', align_corners=False)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)[0]
        x_samples = self.to_tensor(cv2.cvtColor(x_samples, cv2.COLOR_BGR2RGB))
        return x_samples

    def ext_ref_view(self, dataloader):
        view_num = len(dataloader)
        anchors = [(view_num * i) // self.ref_view_num for i in range(self.ref_view_num)] + [view_num]
        random.seed(13789)
        self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])]
        self.num_ref_views = len(self.ref_indices)

    def __call__(self,
                 dataset,
                 opt,
                 pipe,
                 testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
        if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
            sys.exit(
                f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)

        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians)

        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final,
                                            max_steps=opt.iterations)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))


        train_dataset = CameraDataset(scene=scene)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=1,
                                                       shuffle=False, num_workers=0)

        target_exp = dataset.target_exp
        model = HEC(stage=dataset.stage, exp=target_exp)
        model.apply(weights_init)
        model.to(device)

        from utils.loss_utils import (L_color, L_spa, L_exp,
                                      L_TV, DepthAnythingv2Loss,
                                      cdf_loss_weighted, tail_contrast_loss)
        L_color = L_color()
        L_spa = L_spa()
        L_exp = L_exp(16, target_exp)
        L_TV = L_TV()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=3e-4)
        self_cali_iteration = 0
        max_iteration = dataset.max_iteration
        data_iter = iter(train_dataloader)

        import time
        model.train()
        while self_cali_iteration < max_iteration:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
            image = batch['image'].cuda(non_blocking=True)
            optimizer.zero_grad()

            enhanced_image, A = model(image)
            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, image))

            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            loss_cdf = 1 * cdf_loss_weighted(enhanced_image)
            loss_tail = 10 * tail_contrast_loss(enhanced_image)

            loss = Loss_TV + loss_spa + loss_col + loss_exp + loss_cdf + loss_tail

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            print(f"\rIteration[{self_cali_iteration}/{max_iteration}] Loss: {loss:.4f}", end="")
            self_cali_iteration += 1

        print(f"\n [Save Checkpoint Model: {scene.model_path}]")
        torch.save(model.state_dict(), os.path.join(scene.model_path, 'HEC.pth'))
        model.eval()

        self.ext_ref_view(train_dataloader)
        ref_imgs = []
        with torch.no_grad():
            print(f"\n Extracting reference images {self.ref_indices}")
            for ind in self.ref_indices:
                ref_img = train_dataset[ind]['image'].cuda(non_blocking=True)
                ref_img, _ = model(ref_img.unsqueeze(0))
                ref_img = self.process_image(ref_img[0], None)
                ref_imgs.append(ref_img)
            ref_imgs = torch.stack(ref_imgs, dim=0)

        print(f"\n Diffusion Processing")
        with torch.no_grad():
            model.eval()
            for idx, _ in enumerate(train_dataloader):
                tr_cam = viewpoint_stack[idx]
                deg_image = tr_cam.original_image.unsqueeze(0).cuda()
                Cali_img, _ = model(deg_image)
                pre_img = self.process_image(Cali_img[0], ref_imgs)
                # update dataset
                tr_cam.preprocess_image = pre_img

        from tqdm import tqdm
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        ema_loss_for_log = 0.0
        ema_Ll1depth_for_log = 0.0
        ema_psnr_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                                separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"], render_pkg["viewspace_points"],
                                                                       render_pkg["visibility_filter"], render_pkg["radii"])

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.preprocess_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                # ema_Ll1depth_for_log = loss_depth
                mse = torch.nn.functional.mse_loss(render_pkg['render'], viewpoint_cam.gt_image.cuda())
                psnr = -10 * np.log10(mse.item())

                ema_p = max(0.01, 1 / (iteration - first_iter + 1))
                ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "psnr": f"{ema_psnr_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                     testing_iterations, scene, render,
                                     (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                                     dataset.train_test_exp)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold,
                                                    radii)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        ###### Rendering Results
        print("Rendering")
        render_sets(dataset, -1, pipe, False, False, SPARSE_ADAM_AVAILABLE)
        ###### Calculate Metrics
        print("Evaluation")
        evaluate([scene.model_path])
        return None


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.gt_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    set_seed(0)
    safe_state(args.quiet)

    pipeline = GaussianTrainer()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    _ = pipeline(lp.extract(args),
                 op.extract(args),
                 pp.extract(args),
                 args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
