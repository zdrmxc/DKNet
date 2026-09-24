import glob
import importlib
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from common.h36m_dataset import Human36mDataset
from common.load_data_hm36 import Fusion
from common.opt import opts
from common.utils import *

# Enforce strict FP32 precision (disable TF32 truncation for reproducible results)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


# Dynamic Difficulty-Hardness-Sensitive (DHS) Loss Engine with Cosine Annealing
def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    if not hasattr(step, "w_mpjpe_cached"):
        step.w_mpjpe_cached = torch.tensor(
            [1.0, 1.0, 2.5, 2.5, 1.0, 2.5, 2.5, 1.0, 1.0, 1.0, 1.5, 1.5, 4.0, 4.0, 1.5, 4.0, 4.0],
            dtype=torch.float32
        ).cuda()

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split == 'train':
            output_3D = model(input_2D)
        else:
            input_2D, output_3D = input_augmentation(input_2D, model)

        out_target = gt_3D.clone()
        if out_target.dim() == 4:
            out_target[:, :, 0] = 0
        else:
            out_target[:, 0] = 0

        if split == 'train':
            if output_3D.shape != out_target.shape:
                out_target = out_target.view_as(output_3D)

            joint_diff = torch.norm(output_3D - out_target, dim=-1)
            view_shape = *(1,) * (joint_diff.dim() - 1), -1

            batch_joint_mean = joint_diff.mean(dim=0, keepdim=True)
            ratio = joint_diff / (batch_joint_mean + 1e-6)

            # DHS dynamic annealing control schedule
            if epoch is not None and epoch > 12:
                anneal_factor = 0.5 + 0.5 * np.cos(np.pi * (epoch - 12) / max(opt.nepoch - 12, 1))
                max_clamp = 1.0 + 1.0 * anneal_factor
                min_clamp = 1.0 - 0.5 * anneal_factor
            else:
                max_clamp = 2.0
                min_clamp = 0.5

            dynamic_weight = torch.clamp(ratio, min=min_clamp, max=max_clamp).detach()
            loss = (joint_diff * step.w_mpjpe_cached.view(view_shape) * dynamic_weight).mean()

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            output_3D_eval = output_3D.view(output_3D.shape[0], 1, -1, 3).clone()
            out_target_eval = out_target.view(out_target.shape[0], 1, -1, 3).clone()

            output_3D_eval[:, :, 0, :] = 0
            out_target_eval[:, :, 0, :] = 0

            val_loss = weighted_mpjpe(output_3D_eval, out_target_eval, step.w_mpjpe_cached)
            N = input_2D.size(0)
            loss_all['loss'].update(val_loss.detach().cpu().numpy() * N, N)

            action_error_sum = test_calculation(
                output_3D_eval, out_target_eval, action, action_error_sum, opt.dataset, subject
            )

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return loss_all['loss'].avg, p1, p2


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip = model(input_2D_flip)

    output_3D_flip = output_3D_flip.clone()
    output_3D_flip[..., 0] *= -1

    flipped_output = output_3D_flip.clone()
    if flipped_output.dim() == 4:
        flipped_output[:, :, joints_left, :] = output_3D_flip[:, :, joints_right, :]
        flipped_output[:, :, joints_right, :] = output_3D_flip[:, :, joints_left, :]
    else:
        flipped_output[:, joints_left, :] = output_3D_flip[:, joints_right, :]
        flipped_output[:, joints_right, :] = output_3D_flip[:, joints_left, :]

    output_3D = (output_3D_non_flip + flipped_output) / 2
    return input_2D_non_flip, output_3D


if __name__ == '__main__':
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    manualSeed = opt.seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"Learning Rate: {opt.lr}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"Feature Dimension: {opt.channel}")
    print(f"GPU ID: {opt.gpu}")

    checkpoint_dir = os.path.join('ckpt', opt.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = None
    if opt.train:
        writer = SummaryWriter(log_dir=os.path.join('./runs', opt.model_name))
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=os.path.join(checkpoint_dir, 'train.log'),
            level=logging.INFO
        )

    # Safe fallback if opt.model is an empty string
    model_filename = opt.model if getattr(opt, 'model', '') else 'DKNet'

    # Auto-backup execution scripts for provenance (only in training mode)
    if opt.train:
        try:
            model_src_path = os.path.join("model", f"{model_filename}.py")
            model_dst_path = os.path.join(checkpoint_dir, f"{model_filename}.py")
            if os.path.exists(model_src_path):
                shutil.copy(model_src_path, model_dst_path)

            shutil.copy("main.py", os.path.join(checkpoint_dir, "main.py"))

            block_src_dir = os.path.join("model", "block")
            if os.path.exists(block_src_dir):
                shutil.copytree(block_src_dir, os.path.join(checkpoint_dir, "block"), dirs_exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to back up source files: {e}")

    # Robust dataset path parsing
    root_path = opt.root_path
    dataset_path = os.path.join(root_path, f'data_3d_{opt.dataset}.npz')
    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=opt.batch_size, shuffle=True,
            num_workers=int(opt.workers), pin_memory=True
        )

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.workers), pin_memory=True
    )

    model_module = importlib.import_module(f'model.{model_filename}')
    if hasattr(model_module, 'DKNet'):
        ModelClass = getattr(model_module, 'DKNet')
    else:
        candidates = [
            cls for name, cls in model_module.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, torch.nn.Module) and cls.__module__ == model_module.__name__
        ]
        ModelClass = candidates[-1]

    print(f"INFO: Successfully instantiated model: '{ModelClass.__name__}' from 'model.{model_filename}'")
    model = ModelClass(opt).cuda()

    if opt.reload:
        model_dict = model.state_dict()

        # Support both directory path and direct .pth file path
        if os.path.isfile(opt.previous_dir):
            model_path = opt.previous_dir
        else:
            pth_files = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))
            if not pth_files:
                raise FileNotFoundError(f"No checkpoint (.pth) found in {opt.previous_dir}!")
            model_path = pth_files[0]

        print(f"Reloading weights from: {model_path}")
        pre_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'model' in pre_dict:
            pre_dict = pre_dict['model']
        elif 'state_dict' in pre_dict:
            pre_dict = pre_dict['state_dict']

        load_count = 0
        for name, param in pre_dict.items():
            clean_name = name.replace('module.', '')
            # Compatibility layer: remap legacy norm names if present
            if clean_name.startswith('Temporal_norm.'):
                clean_name = clean_name.replace('Temporal_norm.', 'head_norm.')

            if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
                model_dict[clean_name] = param
                load_count += 1

        model.load_state_dict(model_dict)
        print(f"INFO: Successfully loaded {load_count} / {len(model_dict)} tensors.")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"INFO: Total trainable parameters: {model_params / 1e6:.2f} M")

    all_param = list(model.parameters())
    optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    best_epoch = 0

    # Ensure full execution across all configured epochs
    for epoch in range(1, opt.nepoch + 1):
        train_loss = 0.0
        if opt.train:
            train_loss = train(opt, actions, train_dataloader, model, optimizer, epoch)

        val_loss, p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train and writer is not None:
            writer.add_scalar('loss/train_loss', train_loss, epoch)
            writer.add_scalar('loss/val_loss', val_loss, epoch)
            writer.add_scalar('mpjpe', p1, epoch)
            writer.add_scalar('p2', p2, epoch)

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, checkpoint_dir, epoch, p1, model)
            opt.previous_best_threshold = p1
            best_epoch = epoch

        if not opt.train:
            print(
                f"Validation Loss: {val_loss:.4f} | Protocol #1 (MPJPE): {p1:.2f} mm | Protocol #2 (P-MPJPE): {p2:.2f} mm")
            break
        else:
            log_str = (
                f"Epoch: {epoch:02d} | LR: {optimizer.param_groups[0]['lr']:.7f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"P1: {p1:.2f} mm | P2: {p2:.2f} mm | Best Epoch: {best_epoch:02d} (Best P1: {opt.previous_best_threshold:.2f} mm)"
            )
            logging.info(log_str)
            print(log_str)

        # Learning rate schedule
        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay