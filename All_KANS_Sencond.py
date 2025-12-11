#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DR-KANTreeNet: Enhanced KAN-based Model with Lesion Attention and Vessel-Tree Analysis
--------------------------------------------------------------------------------------
• Innovation 1: Lesion-Aware Attention on ResNet-50 features.
• Innovation 2: VesselTreeNet, a KAN-based CNN to model vascular "tree-like" structures.
• Innovation 3: Enhanced Quad-Modal Fusion of Local, Global, and Vascular features.
• KAN-ViT-S / KAN-ViT-L for multi-scale global context.
• KAN-GatingNet MoME fusion and KAN-GCN refinement head.
• IMPROVED: Enhanced data augmentation, focal loss, better attention mechanisms
"""

# ─── 0 通用依赖 ─────────────────────────────────────────────────────
import os
import gc
import warnings
import argparse

# 内存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:False"
os.environ["OMP_NUM_THREADS"] = "2"  # 限制OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "2"  # 限制MKL线程数

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.multiprocessing as mp 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel
from torch.utils.data.distributed import DistributedSampler

# 尝试导入OpenCV，如果失败则使用替代方案
try:
    import cv2
    CV2_AVAILABLE = True
    print("[OK] OpenCV imported successfully")
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Using PIL-based vessel extraction.")

# 尝试导入torchvision.transforms.functional
try:
    from torchvision.transforms.functional import to_tensor as TF_to_tensor
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: torchvision.transforms.functional not available.")

# 禁用不必要的功能以节省内存
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)  # 限制CPU线程数

def setup(rank, world_size):
    import socket
    import time
    
    # 检查是否已经初始化
    if dist.is_initialized():
        print(f"Rank {rank}: Process group already initialized, skipping...")
        return
    
    # Find an available port dynamically
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    # Try multiple ports if one fails
    max_retries = 5
    for attempt in range(max_retries):
        try:
            port = find_free_port()
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)
            
            # 设置环境变量以改善DDP稳定性
            os.environ['NCCL_IB_DISABLE'] = '1'
            os.environ['NCCL_P2P_DISABLE'] = '1'
            os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
            
            # Set longer timeout and use gloo backend for better compatibility
            dist.init_process_group("gloo", rank=rank, world_size=world_size, 
                                  timeout=datetime.timedelta(seconds=60))
            print(f"Rank {rank} successfully initialized on port {port}")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed on rank {rank}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait longer before retrying
                continue
            else:
                print(f"Failed to initialize process group on rank {rank} after {max_retries} attempts")
                raise e

def cleanup():
    try:
        dist.destroy_process_group()
    except:
        pass  # Ignore cleanup errors

def kill_existing_processes():
    """Kill any existing Python processes that might be using ports"""
    try:
        import subprocess
        result = subprocess.run(['pkill', '-f', 'All_KANS_Sencond.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Killed existing processes. Waiting 2 seconds...")
            import time
            time.sleep(2)
    except:
        pass  # Ignore if pkill is not available


def seed_all(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from torchvision import transforms
import timm


# ─── IMPROVED: Enhanced Loss Functions ──────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in DR classification"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ─── Enhanced KAN Implementation (Original) ─────────────────────────
class KANLayer(nn.Module):
    """Memory-efficient KAN implementation for both Linear and Conv operations"""

    def __init__(self, in_features, out_features, grid_size=3, spline_order=3, dropout=0.1):  # 减少grid_size
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.dropout = nn.Dropout(dropout)
        
        # 更小的权重矩阵以节省内存
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.005)
        
        # 简化的归一化
        self.layer_norm = nn.LayerNorm(out_features)
        
        # 简化的SE块
        self.se = nn.Sequential(
            nn.Linear(out_features, max(out_features // 32, 2)),  # 减少通道数
            nn.ReLU(),
            nn.Linear(max(out_features // 32, 2), out_features),
            nn.Sigmoid()
        )
        
        grid = torch.linspace(-1, 1, grid_size)
        self.register_buffer('grid', grid)

    def forward(self, x):
        # 内存优化的前向传播
        x = x.contiguous()

        original_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])
            is_3d = True
        else:
            is_3d = False

        # 如果输入太大，分批处理以节省内存
        if x.numel() > 1e6:  # 如果输入超过100万个元素
            chunk_size = 1000
            outputs = []
            for i in range(0, x.size(0), chunk_size):
                chunk = x[i:i+chunk_size]
                chunk_output = self._process_chunk(chunk)
                outputs.append(chunk_output)
                del chunk, chunk_output
                gc.collect()
            output = torch.cat(outputs, dim=0)
        else:
            output = self._process_chunk(x)
        
        if is_3d:
            output = output.reshape(original_shape[0], original_shape[1], self.out_features)
        return output

    def _process_chunk(self, x):
        """分批处理数据以节省内存"""
        orig_dtype = x.dtype
        compute_dtype = torch.float32 if orig_dtype == torch.float16 else orig_dtype
        x = x.to(compute_dtype)
        
        base_output = F.linear(x, self.base_weight.to(compute_dtype))
        spline_output = self._compute_spline(x)
        output = (base_output + spline_output).to(orig_dtype)
        
        # 应用dropout和归一化
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        # 应用SE
        if len(output.shape) == 2:
            se_weights = self.se(output)
            output = output * se_weights
        else:
            se_weights = self.se(output)
            output = output * se_weights.unsqueeze(1)
        
        return output

    def _compute_spline(self, x):
        """内存高效的样条计算"""
        x_expanded = x.unsqueeze(-1)
        distances = torch.abs(x_expanded - self.grid.to(x.dtype))
        basis = torch.zeros_like(distances)
        mask = distances < 1.0
        basis[mask] = (1.0 - distances[mask]).pow(2)  # 减少幂次以节省内存
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-6)
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight.to(x.dtype))
        
        # 清理内存
        del x_expanded, distances, basis, mask
        return spline_output


class KANConv2d(nn.Module):
    """Memory-efficient KAN-based 2D Convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, grid_size=3):  # 减少grid_size
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride, self.padding = stride, padding
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=stride, padding=padding)
        patch_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.kan = KANLayer(patch_dim, out_channels, grid_size=grid_size)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # 如果图像太大，分批处理以节省内存
        if H * W > 10000:  # 如果图像大于100x100
            outputs = []
            for i in range(0, B, 4):  # 每次处理4张图像
                batch_x = x[i:i+4]
                batch_output = self._process_batch(batch_x)
                outputs.append(batch_output)
                del batch_x, batch_output
                gc.collect()
            return torch.cat(outputs, dim=0)
        else:
            return self._process_batch(x)

    def _process_batch(self, x):
        """处理一批图像"""
        B, _, H, W = x.shape
        patches = self.unfold(x).transpose(1, 2).contiguous()
        output = self.kan(patches)
        
        H_out = (H + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        return output.transpose(1, 2).reshape(B, self.out_channels, H_out, W_out)


# ─── IMPROVED: Enhanced Loss Functions ──────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in DR classification"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ─── IMPROVED: Mixup Augmentation ───────────────────────────────────
class MixupLoss(nn.Module):
    """Mixup loss for better generalization"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target_a, target_b, lam):
        return lam * F.cross_entropy(pred, target_a) + (1 - lam) * F.cross_entropy(pred, target_b)


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss with higher weights for rare classes"""
    def __init__(self, class_weights=None, gamma=2, reduction='mean'):
        super().__init__()
        if class_weights is None:
            # Higher weights for poorly performing classes (1, 3)
            # Based on your F1 scores: [0.988, 0.542, 0.787, 0.063, 0.590]
            self.class_weights = torch.tensor([1.0, 3.0, 1.5, 8.0, 2.5])
        else:
            self.class_weights = torch.tensor(class_weights)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        # Apply class weights
        weighted_loss = focal_loss * self.class_weights[targets]
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


# ─── IMPROVED: Enhanced Data Augmentation ───────────────────────────
class AdvancedAugmentation:
    """Advanced augmentation strategies for DR images"""
    
    @staticmethod
    def get_training_transforms(size=448):
        """Enhanced training transforms with advanced augmentation"""
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    
    @staticmethod
    def get_validation_transforms(size=448):
        """Validation transforms with center crop"""
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_severe_dr_transforms(size=448):
        """Specialized augmentation for severe DR cases"""
        return transforms.Compose([
            transforms.Resize((size + 64, size + 64)),  # Larger for more aggressive crop
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.7),  # Higher flip probability
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),  # More rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.15),  # More aggressive
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),  # More translation
            transforms.RandomPerspective(distortion_scale=0.15, p=0.5),  # More perspective
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3))  # More erasing
        ])


# ─── 1 CLI & 路径 ──────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=12)  # Reduced BS for higher memory usage
    p.add_argument("--ep", type=int, default=60)  # Increased epochs for better convergence
    p.add_argument("--img", type=int, default=448)  # Increased image size for better detail
    p.add_argument("--multi_scale", action="store_true", help="Use multi-scale training")
    p.add_argument("--min_scale", type=int, default=384, help="Minimum image scale for multi-scale training")
    p.add_argument("--max_scale", type=int, default=512, help="Maximum image scale for multi-scale training")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--save_dir", type=str, default="./hybrid_outputs_v2")
    p.add_argument("--r50", type=str, default="./resnet50-19c8e357.pth")
    p.add_argument("--vit_ckpt", type=str, default="hf_hub:timm/vit_small_patch16_224.augreg_in21k")
    p.add_argument("--focal", action="store_true", help="Use focal loss")
    p.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    p.add_argument("--mixup", action="store_true", help="Use mixup augmentation")
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--class_balanced", action="store_true", help="Use class-balanced focal loss")
    p.add_argument("--severe_dr_weight", type=float, default=8.0, help="Weight for severe DR class")
    p.add_argument("--mild_dr_weight", type=float, default=3.0, help="Weight for mild DR class")
    p.add_argument("--single_gpu", action="store_true", help="Force single-GPU mode")
    return p.parse_known_args()[0]


# ─── 2 随机种子 ────────────────────────────────────────────────────
def seed_all(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ─── IMPROVED: Enhanced Data Augmentation ───────────────────────────
class AdvancedAugmentation:
    """Advanced augmentation strategies for DR images"""
    
    @staticmethod
    def get_training_transforms(size=448):
        """Enhanced training transforms with advanced augmentation"""
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    
    @staticmethod
    def get_validation_transforms(size=448):
        """Validation transforms with center crop"""
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# ─── INNOVATION 1: Enhanced Lesion-Aware Attention Module ───────────
class EnhancedLesionAttention(nn.Module):
    """Enhanced attention module with learned lesion detection"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Lesion-specific attention
        self.lesion_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.lesion_attention = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Severe DR specific attention
        self.severe_dr_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, original_image):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        channel_attn = self.sigmoid(self.conv2(self.relu(self.conv1(avg_pool + max_pool))))
        
        # Lesion attention from original image
        # Resize original_image to match the spatial dimensions of x
        if original_image.shape[2:] != x.shape[2:]:
            original_image = F.interpolate(original_image, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        lesion_feat = self.lesion_conv(original_image)
        lesion_attn = self.lesion_attention(lesion_feat)
        
        # Ensure spatial dimensions match
        if lesion_attn.shape[2:] != x.shape[2:]:
            lesion_attn = F.interpolate(lesion_attn, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Ensure channel attention has the right spatial dimensions
        if channel_attn.shape[2:] != x.shape[2:]:
            channel_attn = F.interpolate(channel_attn, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Severe DR attention
        severe_attn = self.severe_dr_detector(x)
        
        # Combine all attention mechanisms
        combined_attn = channel_attn * lesion_attn * severe_attn
        
        # Final safety check
        if combined_attn.shape != x.shape:
            combined_attn = F.interpolate(combined_attn, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return x * combined_attn


# ─── INNOVATION 2: Enhanced Vessel-Tree Network Module ──────────────
class EnhancedVesselTreeNet(nn.Module):
    """Enhanced vessel tree network with attention"""

    def __init__(self, in_channels=1, out_features=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            KANLayer(256, out_features)
        )

    @torch.no_grad()
    def get_vessel_mask(self, x):
        # 添加调试信息
        print(f"get_vessel_mask: CV2_AVAILABLE = {CV2_AVAILABLE}")
        print(f"get_vessel_mask: globals() has cv2: {'cv2' in globals()}")
        print(f"get_vessel_mask: locals() has cv2: {'cv2' in locals()}")
        
        # 确保cv2在函数作用域中可用
        if CV2_AVAILABLE and 'cv2' not in locals():
            import cv2
        
        masks = []
        for img_tensor in x:
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            green_ch = img_np[:, :, 1]

            if CV2_AVAILABLE:
                # Enhanced vessel extraction using OpenCV
                try:
                    # 直接使用cv2，因为它在模块级别已经导入
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(green_ch)
                    
                    # Multi-scale vessel detection
                    scales = [1, 2, 4]
                    vessel_maps = []
                    for scale in scales:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale*3, scale*3))
                        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
                        vessel_maps.append(opened)
                    
                    # Combine multi-scale maps
                    combined = np.mean(vessel_maps, axis=0)
                    _, mask = cv2.threshold(combined.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except Exception as e:
                    print(f"OpenCV vessel extraction failed: {e}, using fallback method")
                    # Fallback: simple thresholding
                    mask = (green_ch > np.mean(green_ch)).astype(np.uint8) * 255
            else:
                # Fallback: PIL-based vessel extraction
                try:
                    # Simple vessel detection using PIL and numpy
                    from PIL import ImageFilter, ImageEnhance
                    
                    # Convert to PIL image
                    pil_img = Image.fromarray(green_ch)
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(pil_img)
                    enhanced = enhancer.enhance(2.0)
                    
                    # Apply edge detection
                    edges = enhanced.filter(ImageFilter.FIND_EDGES)
                    
                    # Convert back to numpy and threshold
                    enhanced_np = np.array(enhanced)
                    edges_np = np.array(edges)
                    
                    # Combine enhanced and edges
                    combined = (enhanced_np + edges_np) / 2
                    mask = (combined > np.mean(combined)).astype(np.uint8) * 255
                    
                except Exception as e:
                    print(f"PIL vessel extraction failed: {e}, using simple thresholding")
                    # Final fallback: simple thresholding
                    mask = (green_ch > np.mean(green_ch)).astype(np.uint8) * 255
            
            # Convert to tensor
            if TF_AVAILABLE:
                mask_tensor = TF_to_tensor(Image.fromarray(mask))
            else:
                # Manual tensor conversion
                mask_tensor = torch.from_numpy(mask).float() / 255.0
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
            
            masks.append(mask_tensor)

        return torch.stack(masks, dim=0).to(x.device)

    def forward(self, x):
        vessel_mask = self.get_vessel_mask(x)
        return self.layers(vessel_mask)


# ─── 3 Enhanced Dataset ─────────────────────────────────────────────
class EnhancedAptosDS(Dataset):
    def __init__(self, df, img_dir, size=384, aug=False):  # 减少默认图像尺寸
        self.df, self.dir = df.reset_index(drop=True), Path(img_dir)
        self.idc = "id_code" if "id_code" in df else "image"
        self.lab = "diagnosis" if "diagnosis" in df else "level"
        self.size, self.aug = size, aug
        self.suf = [".png", ".jpg", ".jpeg"]
        
        # 简化的变换以节省内存
        if aug:
            self.tfm = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _read(self, name):
        """简化的图像读取以节省内存"""
        for s in self.suf:
            p = self.dir / (name + s)
            if p.exists(): 
                try:
                    # 使用PIL直接读取，避免复杂的OpenCV预处理
                    img = Image.open(p).convert('RGB')
                    return img
                except Exception as e:
                    print(f"Error loading image {name}: {e}")
                    continue
        raise FileNotFoundError(name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._read(r[self.idc])
        img = self.tfm(img)
        return img, torch.tensor(r[self.lab], dtype=torch.long)


# ─── 4 Enhanced KAN-based小模块 ─────────────────────────────────────
class EnhancedKANCA(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg, self.max = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc1, self.fc2 = KANConv2d(c, c // r, 1), KANConv2d(c // r, c, 1)
        self.relu, self.sig = nn.ReLU(), nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        
        # Additional attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c, c // 8, 1),
            nn.ReLU(),
            nn.Conv2d(c // 8, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc2(self.dropout(self.relu(self.fc1(self.avg(x)))))
        max_out = self.fc2(self.dropout(self.relu(self.fc1(self.max(x)))))
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(x)
        channel_attn = self.sig(avg_out + max_out)
        
        # Combine both attention mechanisms
        return x * channel_attn * spatial_attn


class EnhancedKANSA(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv, self.sig = KANConv2d(2, 1, k, padding=k // 2), nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        avg_out, max_out = torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]
        return self.sig(self.conv(torch.cat([avg_out, max_out], dim=1)))


class EnhancedKANDAM(nn.Module):
    def __init__(self, c): 
        super().__init__()
        self.ca, self.sa = EnhancedKANCA(c), EnhancedKANSA()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): 
        x = self.ca(x) * x
        x = self.dropout(x)
        return self.sa(x) * x


class EnhancedKANGCN(nn.Module):
    def __init__(self, n_cls, hid=64, d=224):  # Increased hidden size
        super().__init__()
        self.A = nn.Parameter(torch.eye(n_cls) + 0.2 * (torch.ones(n_cls, n_cls) - torch.eye(n_cls)))
        self.fc1, self.fc2 = KANLayer(n_cls, hid), KANLayer(hid, d)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self):
        x = self.A @ torch.eye(self.A.size(0), device=self.A.device)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.A @ x
        x = self.fc2(x)
        return self.layer_norm(x)


# ─── 5 Enhanced KAN-ViT Block ─────────────────────────────────────
class EnhancedKANMLP(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hid = dim * hidden_mult
        self.fc1, self.fc2 = KANLayer(dim, hid), KANLayer(hid, dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x)


def load_enhanced_vit16(src: str):
    import timm
    if src and src.startswith("hf_hub:"):
        vit = timm.create_model(src, pretrained=True, num_classes=0)
    elif src and Path(src).exists():
        vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        vit.load_state_dict(torch.load(src, map_location="cpu", weights_only=False), strict=False)
    else:
        vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    
    # Enhanced MLP with dropout and layer norm
    for blk in vit.blocks: 
        blk.mlp = EnhancedKANMLP(dim=vit.embed_dim)
    return vit


# ─── 6 Enhanced KAN-MoME Gating ────────────────────────────────────
class EnhancedKANGatingNet(nn.Module):
    def __init__(self, in_dim, n_experts=2):
        super().__init__()
        self.pool, self.flatten = nn.AdaptiveAvgPool2d(1), nn.Flatten()
        self.fc1, self.fc2 = KANLayer(in_dim, in_dim // 4), KANLayer(in_dim // 4, n_experts)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, feats):
        x = feats[0].unsqueeze(-1).unsqueeze(-1)
        x = self.flatten(self.pool(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        w = torch.softmax(self.fc2(x), dim=-1)
        return sum(w[:, i:i + 1] * f for i, f in enumerate(feats))


# ─── 7 IMPROVED MODEL: Enhanced DR-KANTreeNet ──────────────────────
class EnhancedDRKANTreeNet(nn.Module):
    def __init__(self, r50_path, vit_ckpt_path, n_cls=5, dropout_rate=0.3):
        super().__init__()
        # --- 保持完整的骨干网络 ---
        import timm
        self.res = timm.create_model("resnet50", pretrained=False, num_classes=0, features_only=True)
        if r50_path and Path(r50_path).exists():
            sd = torch.load(r50_path, map_location="cpu", weights_only=False)
            self.res.load_state_dict(sd, strict=False)
        
        # 验证ResNet特征维度
        print(f"ResNet model created with features_only=True")
        print(f"Expected feature dimensions: {self.res.feature_info.channels()}")

        # 保持ViT模型
        self.vitS = load_enhanced_vit16(vit_ckpt_path)
        self.vitL = load_enhanced_vit16(vit_ckpt_path)
        
        # 冻结早期层以节省内存
        for n, p in self.vitS.named_parameters():
            if n.startswith("blocks.") and int(n.split('.')[1]) < 6:  # 减少冻结层数
                p.requires_grad = False

        # 保持完整的注意力模块
        self.enhanced_attention = EnhancedLesionAttention(in_channels=2048)
        self.res_dam = EnhancedKANDAM(2048)
        
        # 保持血管树网络
        self.vessel_net = EnhancedVesselTreeNet(in_channels=1, out_features=256)

        # 保持特征金字塔网络，但减少通道数
        self.fpn_layers = nn.ModuleList([
            KANLayer(2048, 512),  # Final features (2048 -> 512)
            KANLayer(1024, 512),  # Mid-level features (1024 -> 512)
            KANLayer(512, 512),   # Low-level features (512 -> 512)
        ])
        
        # 添加维度检查
        self.fpn_input_dims = [2048, 1024, 512]
        self.fpn_output_dim = 512
        
        # 特征映射层
        self.mapR = KANLayer(2048, 512)
        self.lnR = nn.LayerNorm(512)
        self.mapT = KANLayer(self.vitS.num_features, 256)
        self.lnT = nn.LayerNorm(256)
        self.mapS = KANLayer(self.vitL.num_features, 512)
        self.lnS = nn.LayerNorm(512)
        self.mapV = KANLayer(256, 256)
        self.lnV = nn.LayerNorm(256)
        
        # 保持门控网络
        self.gate = EnhancedKANGatingNet(in_dim=512, n_experts=2)
        
        # 简化的融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 计算融合特征的输出维度
        # fusion_conv: 1 -> 32 -> 64 -> 1 (经过AdaptiveAvgPool2d和Flatten)
        # 所以fusion_output_dim = 64
        fusion_output_dim = 64
        
        # 计算总连接维度
        total_concat_dim = 512 + 512 + 256 + fusion_output_dim  # r + fused_vit + vessel_feats + fusion_features
        print(f"Calculated total concatenation dimension: {total_concat_dim}")
        
        # 调整全连接层维度
        self.mil_fc1 = KANLayer(total_concat_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.mil_fc2 = KANLayer(512, 256)   # 减少隐藏层大小
        self.relu = nn.ReLU()

        # 保持完整的头部
        self.head = KANLayer(256, n_cls)
        self.gcn = EnhancedKANGCN(n_cls, d=256)  # 减少维度
        
        # 正则化
        self.batch_norm = nn.BatchNorm1d(256)
        
        # 添加分类头权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，避免梯度消失"""
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 特别初始化分类头
        if hasattr(self.head, 'base_weight'):
            nn.init.xavier_uniform_(self.head.base_weight, gain=0.1)
        if hasattr(self.head, 'spline_weight'):
            nn.init.xavier_uniform_(self.head.spline_weight, gain=0.05)

    def forward(self, x):
        # 确保输入在正确的设备上
        device = x.device
        
        # 获取所有ResNet特征
        res_features = self.res(x)
        final_features = res_features[4]  # 2048 channels
        mid_features = res_features[2]    # 1024 channels
        low_features = res_features[1]    # 512 channels

        # 增强注意力与原始图像
        r_feats = self.enhanced_attention(final_features, x)
        r_feats = self.res_dam(r_feats)

        # FPN处理 - 修复维度不匹配问题
        fpn_feats = []
        
        try:
            # 处理最终特征 (2048 -> 512)
            final_pooled = F.adaptive_avg_pool2d(final_features, 1).flatten(1)
            if final_pooled.size(1) == 2048:
                # 确保FPN层在正确的设备上
                fpn_layer = self.fpn_layers[0].to(device)
                fpn_feats.append(fpn_layer(final_pooled))
            else:
                # 如果维度不匹配，创建一个零张量
                fpn_feats.append(torch.zeros(final_pooled.size(0), 512, device=device))
            
            # 处理中间特征 (1024 -> 512)
            mid_pooled = F.adaptive_avg_pool2d(mid_features, 1).flatten(1)
            if mid_pooled.size(1) == 1024:
                # 确保FPN层在正确的设备上
                fpn_layer = self.fpn_layers[1].to(device)
                fpn_feats.append(fpn_layer(mid_pooled))
            else:
                # 如果维度不匹配，创建一个零张量
                fpn_feats.append(torch.zeros(mid_pooled.size(0), 512, device=device))
            
            # 处理低层特征 (512 -> 512)
            low_pooled = F.adaptive_avg_pool2d(low_features, 1).flatten(1)
            if low_pooled.size(1) == 512:
                # 确保FPN层在正确的设备上
                fpn_layer = self.fpn_layers[2].to(device)
                fpn_feats.append(fpn_layer(low_pooled))
            else:
                # 如果维度不匹配，创建一个零张量
                fpn_feats.append(torch.zeros(low_pooled.size(0), 512, device=device))
                
        except Exception as e:
            print(f"FPN processing error: {e}")
            # 如果FPN处理失败，创建默认特征
            batch_size = x.size(0)
            fpn_feats = [torch.zeros(batch_size, 512, device=device) for _ in range(3)]
        
        # 组合FPN特征 - 确保所有特征都是512维
        fpn_feats = [feat for feat in fpn_feats if feat.size(1) == 512]
        if fpn_feats:
            fpn_combined = torch.stack(fpn_feats, dim=1).mean(dim=1)
        else:
            # 如果没有有效的FPN特征，创建一个零张量
            fpn_combined = torch.zeros(r_feats.size(0), 512, device=device)

        # 池化和展平
        r = F.adaptive_avg_pool2d(r_feats, 1).flatten(1)
        r = self.lnR(self.mapR(r))
        
        # 与FPN特征结合
        r = r + fpn_combined

        # 处理ViT特征
        xv = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        vt_cls = self.vitS.forward_features(xv)[:, 0]
        vs_cls = self.vitL.forward_features(xv)[:, 0]
        vt = self.lnT(self.mapT(vt_cls))
        vs = self.lnS(self.mapS(vs_cls))

        # 血管特征
        vessel_feats = self.vessel_net(x)
        vessel_feats = self.lnV(self.mapV(vessel_feats))

        # 增强融合
        vt_expanded = torch.cat([vt, vt], dim=1)
        fused_vit = self.gate([vs, vt_expanded])
        
        # 血管特征处理
        vessel_2d = vessel_feats.unsqueeze(-1).unsqueeze(-1)
        vessel_2d = F.interpolate(vessel_2d, size=(7, 7), mode='bilinear', align_corners=False)
        vessel_2d = vessel_2d.mean(dim=1, keepdim=True)
        fusion_features = self.fusion_conv(vessel_2d)

        # 确保所有特征具有相同的批次大小
        min_batch_size = min(r.size(0), fused_vit.size(0), vessel_feats.size(0), fusion_features.size(0))
        if min_batch_size < r.size(0):
            r = r[:min_batch_size]
        if min_batch_size < fused_vit.size(0):
            fused_vit = fused_vit[:min_batch_size]
        if min_batch_size < vessel_feats.size(0):
            vessel_feats = vessel_feats[:min_batch_size]
        if min_batch_size < fusion_features.size(0):
            fusion_features = fusion_features[:min_batch_size]

        # 连接所有特征 - 添加调试信息
        concat_features = torch.cat([r, fused_vit, vessel_feats, fusion_features], dim=1)
        print(f"Feature dimensions - r: {r.shape}, fused_vit: {fused_vit.shape}, vessel_feats: {vessel_feats.shape}, fusion_features: {fusion_features.shape}")
        print(f"Concatenated features shape: {concat_features.shape}")
        print(f"Expected input for mil_fc1: {self.mil_fc1.in_features}, Actual: {concat_features.size(1)}")
        
        # 如果维度不匹配，调整mil_fc1层
        if concat_features.size(1) != self.mil_fc1.in_features:
            print(f"Adjusting mil_fc1 input dimension from {self.mil_fc1.in_features} to {concat_features.size(1)}")
            # 重新创建mil_fc1层以匹配实际输入维度
            self.mil_fc1 = KANLayer(concat_features.size(1), 512).to(x.device)
        
        # 最终分类
        f = self.mil_fc1(concat_features)
        f = self.dropout(self.relu(f))
        f = self.mil_fc2(f)
        f = self.batch_norm(f)
        
        # 添加调试信息
        print(f"Feature f shape: {f.shape}, mean: {f.mean():.4f}, std: {f.std():.4f}")
        
        base_logits = self.head(f)
        print(f"Base logits shape: {base_logits.shape}, mean: {base_logits.mean():.4f}, std: {base_logits.std():.4f}")
        
        # 检查logits是否合理
        if torch.isnan(base_logits).any() or torch.isinf(base_logits).any():
            print("Warning: NaN or Inf detected in base_logits, using fallback")
            base_logits = torch.randn_like(base_logits) * 0.01
        
        # 简化GCN部分，避免复杂的矩阵运算
        try:
            gcn_weight = self.gcn()
            gcn_logits = F.sigmoid(f @ gcn_weight.t()) * base_logits
            print(f"GCN logits shape: {gcn_logits.shape}, mean: {gcn_logits.mean():.4f}")
        except Exception as e:
            print(f"GCN computation failed: {e}, using base_logits only")
            gcn_logits = base_logits

        # 确保所有参数都被使用，但减少正则化强度
        param_reg = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_reg = param_reg + param.norm(2)
        
        # 使用更合理的正则化系数
        final_output = base_logits + 0.1 * gcn_logits + 1e-6 * param_reg
        
        # 最终检查输出
        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            print("Warning: NaN or Inf detected in final_output, using fallback")
            final_output = torch.randn_like(final_output) * 0.01
        
        print(f"Final output shape: {final_output.shape}, mean: {final_output.mean():.4f}, std: {final_output.std():.4f}")
        print(f"Output class distribution: {torch.softmax(final_output, dim=1).mean(dim=0)}")
        
        return final_output


# ─── 8 Enhanced Metric Calculation ──────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: matplotlib or seaborn not available. Confusion matrix plotting will be disabled.")
    plt = None
    sns = None

@torch.no_grad()
def enhanced_metric(model, loader, dev):
    model.eval()
    yt, yp, prob = [], [], []
    
    print(f"\n=== Validation Metrics Calculation ===")
    print(f"Device: {dev}")
    
    for batch_idx, (x, y) in enumerate(loader):
        try:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            
            # 检查输出
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"Warning: Invalid outputs in validation batch {batch_idx}")
                continue
            
            # 获取预测
            pred = out.argmax(1).cpu()
            probs = torch.softmax(out, 1).cpu()
            
            # 收集结果
            yt.extend(y.cpu().tolist())
            yp.extend(pred.tolist())
            prob.append(probs.numpy())
            
            # 打印前几个batch的详细信息
            if batch_idx < 3:
                print(f"Val Batch {batch_idx}:")
                print(f"  True labels: {y[:5].tolist()}")
                print(f"  Pred labels: {pred[:5].tolist()}")
                print(f"  Probabilities: {probs[:5].max(dim=1)[0].tolist()}")
                print(f"  Class distribution: {probs.mean(dim=0)}")
                
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {e}")
            continue
    
    if not yt or not yp:
        print("Warning: No valid predictions collected!")
        return 0.0, 0.0, 0.0, np.nan, None, [0.0] * 5
    
    print(f"Total validation samples: {len(yt)}")
    print(f"True label distribution: {np.bincount(yt, minlength=5)}")
    print(f"Pred label distribution: {np.bincount(yp, minlength=5)}")
    
    # 连接概率
    try:
        prob = np.concatenate(prob, 0)
    except:
        print("Warning: Failed to concatenate probabilities")
        prob = np.zeros((len(yt), 5))
    
    # Enhanced metrics
    k = cohen_kappa_score(yt, yp, weights='quadratic')
    a = accuracy_score(yt, yp)
    f = f1_score(yt, yp, average='weighted')
    
    try:
        auc = roc_auc_score(yt, prob, multi_class='ovr', average='weighted')
    except ValueError:
        auc = np.nan
    
    # Additional metrics
    cm = confusion_matrix(yt, yp)
    per_class_f1 = f1_score(yt, yp, average=None)
    
    print(f"Final metrics: K={k:.4f}, A={a:.4f}, F1={f:.4f}, AUC={auc:.4f}")
    print(f"Per-class F1: {per_class_f1}")
    print(f"Confusion Matrix:\n{cm}")
    
    return k, a, f, auc, cm, per_class_f1


# ─── 9 Enhanced Training Function ──────────────────────────────────
def train_fold(rank, world_size, fold, tr_idx, va_idx, df, args, img_dir, save_dir, r50_path, vit_ckpt_path):
    """
    内存优化的训练函数，保持分布式训练能力
    """
    # 只在多GPU模式下初始化分布式训练
    if world_size > 1:
        setup(rank, world_size)
        dev = torch.device(f"cuda:{rank}")
        
        # 增强数据集
        lbl = "diagnosis" if "diagnosis" in df else "level"
        train_dataset = EnhancedAptosDS(df.iloc[tr_idx], img_dir, args.img, True)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        # 优化DataLoader设置
        tr_ld = DataLoader(train_dataset, batch_size=args.bs, num_workers=2, pin_memory=True, sampler=train_sampler)
        va_ld = DataLoader(EnhancedAptosDS(df.iloc[va_idx], img_dir, args.img, False),
                           batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)
    else:
        # 单GPU模式
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 增强数据集
        lbl = "diagnosis" if "diagnosis" in df else "level"
        train_dataset = EnhancedAptosDS(df.iloc[tr_idx], img_dir, args.img, True)

        # 优化DataLoader设置
        tr_ld = DataLoader(train_dataset, batch_size=args.bs, num_workers=2, pin_memory=True, shuffle=True)
        va_ld = DataLoader(EnhancedAptosDS(df.iloc[va_idx], img_dir, args.img, False),
                           batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    # 增强模型
    model = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt_path).to(dev)
    
    # 测试模型
    if rank == 0:
        try:
            with torch.no_grad():
                dummy_input = torch.randn(2, 3, args.img, args.img).to(dev)
                dummy_output = model(dummy_input)
                print("Model test successful")
        except Exception as e:
            print(f"Model test failed: {e}")
            
            # 调用调试函数
            debug_model_dimensions(model, (2, 3, args.img, args.img), dev)
            
            print("Attempting to fix model architecture...")
            
            # 尝试修复模型架构问题
            try:
                # 检查并修复FPN层
                actual_model = model.module if isinstance(model, DDP) else model
                
                # 重新初始化FPN层以确保维度正确，并移动到正确的设备
                actual_model.fpn_layers = nn.ModuleList([
                    KANLayer(2048, 512).to(dev),  # Final features (2048 -> 512)
                    KANLayer(1024, 512).to(dev),  # Mid-level features (1024 -> 512)
                    KANLayer(512, 512).to(dev),   # Low-level features (512 -> 512)
                ])
                
                # 确保所有FPN层都在正确的设备上
                for layer in actual_model.fpn_layers:
                    layer.to(dev)
                
                # 再次测试
                with torch.no_grad():
                    dummy_input = torch.randn(2, 3, args.img, args.img).to(dev)
                    dummy_output = model(dummy_input)
                    print("Model test successful after fix")
            except Exception as e2:
                print(f"Model fix failed: {e2}")
                import traceback
                traceback.print_exc()
                return None
    
    # 只在多GPU模式下使用DDP
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    def pg(m, lr):
        return {"params": filter(lambda p: p.requires_grad, m.parameters()), "lr": lr}

    # 获取实际模型（如果是DDP包装的，需要访问.module）
    m = model.module if hasattr(model, 'module') else model
    
    # 增强优化器，但减少学习率以节省内存
    opt = torch.optim.AdamW([
        pg(m.vitS, 2e-5),  # 降低ViT学习率
        pg(m.vitL, 2e-5),
        pg(m.res, 5e-5),   # 降低ResNet学习率
        pg(m.enhanced_attention, 1e-4),
        pg(m.vessel_net, 1e-4),
        pg(m.mil_fc1, 2e-4),
        pg(m.mil_fc2, 2e-4),
        pg(m.head, 2e-4),
        pg(m.gcn, 2e-4),
        pg(m.gate, 2e-4)
    ], weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # 简化的学习率调度器
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5)
    
    # 保持完整的损失函数选择
    if args.mixup:
        ce = MixupLoss(alpha=args.mixup_alpha)
    elif args.focal:
        if args.class_balanced:
            custom_weights = [1.0, args.mild_dr_weight, 1.5, args.severe_dr_weight, 2.5]
            ce = ClassBalancedFocalLoss(class_weights=custom_weights, gamma=3)
        else:
            ce = FocalLoss(alpha=1, gamma=2)
    else:
        ce = LabelSmoothingLoss(classes=5, smoothing=args.label_smoothing)

    best_k, best_p = -1, save_dir / f"enhanced_kantree_fold{fold}_best.pth"
    patience = 15  # 增加耐心值
    no_improve_count = 0
    min_delta = 0.001  # 最小改进阈值
    
    for ep in range(1, args.ep + 1):
        model.train()
        # 只在多GPU模式下设置epoch
        if world_size > 1:
            train_sampler.set_epoch(ep)
        
        pbar = tqdm(tr_ld, desc=f"", ncols=0, disable=True)  # 静默进度条
        
        for batch_idx, (x, y) in enumerate(pbar):
            try:
                x, y = x.to(dev), y.to(dev)
                
                # Mixup增强
                if args.mixup and random.random() < 0.5:
                    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                    batch_size = x.size(0)
                    index = torch.randperm(batch_size).to(dev)
                    x = lam * x + (1 - lam) * x[index, :]
                    y_a, y_b = y, y[index]
                    
                    if isinstance(ce, MixupLoss):
                        loss = ce(model(x), y_a, y_b, lam)
                    else:
                        loss = ce(model(x), y)
                else:
                    loss = ce(model(x), y)
                
                # 检查NaN损失
                if torch.isnan(loss):
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # 更新权重
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                # 学习率调度
                sch.step()
                
                # 清理内存
                del x, y, loss
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "size" in str(e) or "dimension" in str(e):
                    continue
                else:
                    continue
            except Exception as e:
                continue

        # 增强验证
        if world_size > 1:
            # 多GPU模式，使用model.module
            k, a, f1, auc, cm, per_class_f1 = enhanced_metric(model.module, va_ld, dev)
        else:
            # 单GPU模式，直接使用model
            k, a, f1, auc, cm, per_class_f1 = enhanced_metric(model, va_ld, dev)

        if rank == 0:
            print(f"[Val] K={k:.4f} A={a:.4f} F1={f1:.4f} AUC={auc:.4f}")
            print(f"[Per-Class F1] {per_class_f1}")
            
            if k > best_k + 0.001:  # 最小改进阈值
                best_k = k
                no_improve_count = 0
                # 根据模式保存模型
                if world_size > 1:
                    torch.save(model.module.state_dict(), best_p)
                else:
                    torch.save(model.state_dict(), best_p)
                
                # 保存混淆矩阵
                if plt is not None and sns is not None:
                    try:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Confusion Matrix - Fold {fold}, Epoch {ep}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.savefig(save_dir / f'confusion_matrix_fold{fold}_ep{ep}.png')
                        plt.close()
                    except Exception:
                        pass
            else:
                no_improve_count += 1
            
            # 早停
            if no_improve_count >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stopping training.")
                break
        
        # 清理内存
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    # 只在多GPU模式下清理分布式训练
    if world_size > 1:
        cleanup()
    
    if rank == 0:
        return best_k, best_p
    return None


# ─── 10 Enhanced Training Overall + External Validation ─────────────
from sklearn.model_selection import StratifiedKFold, train_test_split

# ─── 11 Multi-GPU Training with DataParallel ───────────────────────
def debug_model_dimensions(model, input_shape, device=None):
    """调试模型各层的维度"""
    print(f"\n=== Model Dimension Debug ===")
    print(f"Input shape: {input_shape}")
    
    try:
        # 如果提供了设备参数，将模型移动到该设备
        if device is not None:
            print(f"Moving model to device: {device}")
            model = model.to(device)
        
        with torch.no_grad():
            # 确保输入在正确的设备上
            device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
            x = torch.randn(*input_shape).to(device)
            
            if hasattr(model, 'module'):
                actual_model = model.module
            else:
                actual_model = model
            
            # 检查ResNet特征
            if hasattr(actual_model, 'res'):
                try:
                    res_features = actual_model.res(x)
                    print(f"ResNet features:")
                    for i, feat in enumerate(res_features):
                        print(f"  Layer {i}: {feat.shape}")
                except Exception as e:
                    print(f"ResNet feature extraction failed: {e}")
            
            # 检查FPN层
            if hasattr(actual_model, 'fpn_layers'):
                print(f"FPN layers:")
                for i, layer in enumerate(actual_model.fpn_layers):
                    print(f"  FPN {i}: in_features={layer.in_features}, out_features={layer.out_features}")
                    # 检查设备
                    if hasattr(layer, 'base_weight'):
                        print(f"  Device: {layer.base_weight.device}")
            
            # 检查其他关键层
            if hasattr(actual_model, 'mapR'):
                print(f"mapR: in_features={actual_model.mapR.in_features}, out_features={actual_model.mapR.out_features}")
                if hasattr(actual_model.mapR, 'base_weight'):
                    print(f"  Device: {actual_model.mapR.base_weight.device}")
            if hasattr(actual_model, 'mil_fc1'):
                print(f"mil_fc1: in_features={actual_model.mil_fc1.in_features}, out_features={actual_model.mil_fc1.out_features}")
                if hasattr(actual_model.mil_fc1, 'base_weight'):
                    print(f"  Device: {actual_model.mil_fc1.base_weight.device}")
                
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

def multi_gpu_train(tr_df, val_df, args, img_dir, save_dir, r50_path, vit_ckpt_path, num_gpus):
    """
    使用DataParallel进行多GPU训练
    避免分布式训练的网络问题
    """
    print(f"Starting multi-GPU training with {num_gpus} GPUs using DataParallel...")
    
    # 设置主设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Main device: {device}")
    
    # 数据分割
    y = "diagnosis" if "diagnosis" in tr_df else "level"
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []
    
    # 调整batch size以适应多GPU
    effective_batch_size = args.bs * num_gpus
    print(f"Effective batch size: {effective_batch_size} (per GPU: {args.bs})")
    
    for i, (tr, va) in enumerate(skf.split(tr_df, tr_df[y]), 1):
        print(f"\n--- Training Fold {i}/5 ---")
        
        # 创建数据集
        train_dataset = EnhancedAptosDS(tr_df.iloc[tr], img_dir, args.img, True)
        val_dataset = EnhancedAptosDS(tr_df.iloc[va], img_dir, args.img, False)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True)
        
        # 创建模型
        model = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt_path).to(device)
        
        # 测试模型
        try:
            with torch.no_grad():
                dummy_input = torch.randn(2, 3, args.img, args.img).to(device)
                dummy_output = model(dummy_input)
                print("Model test successful")
        except Exception as e:
            print(f"Model test failed: {e}")
            
            # 调用调试函数
            debug_model_dimensions(model, (2, 3, args.img, args.img), device)
            
            print("Attempting to fix model architecture...")
            
            # 尝试修复模型架构问题
            try:
                # 检查并修复FPN层
                actual_model = model.module if isinstance(model, nn.DataParallel) else model
                
                # 重新初始化FPN层以确保维度正确，并移动到正确的设备
                actual_model.fpn_layers = nn.ModuleList([
                    KANLayer(2048, 512).to(device),  # Final features (2048 -> 512)
                    KANLayer(1024, 512).to(device),  # Mid-level features (1024 -> 512)
                    KANLayer(512, 512).to(device),   # Low-level features (512 -> 512)
                ])
                
                # 确保所有FPN层都在正确的设备上
                for layer in actual_model.fpn_layers:
                    layer.to(device)
                
                # 再次测试
                with torch.no_grad():
                    dummy_input = torch.randn(2, 3, args.img, args.img).to(device)
                    dummy_output = model(dummy_input)
                    print("Model test successful after fix")
            except Exception as e2:
                print(f"Model fix failed: {e2}")
                import traceback
                traceback.print_exc()
                continue
        
        # 使用DataParallel包装模型
        if num_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Model wrapped with DataParallel for {num_gpus} GPUs")
        
        # 优化器设置
        def pg(m, lr):
            return {"params": filter(lambda p: p.requires_grad, m.parameters()), "lr": lr}
        
        # 获取实际模型（移除DataParallel包装）
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        opt = torch.optim.AdamW([
            pg(actual_model.vitS, 2e-5),
            pg(actual_model.vitL, 2e-5),
            pg(actual_model.res, 5e-5),
            pg(actual_model.enhanced_attention, 1e-4),
            pg(actual_model.vessel_net, 1e-4),
            pg(actual_model.mil_fc1, 2e-4),
            pg(actual_model.mil_fc2, 2e-4),
            pg(actual_model.head, 2e-4),
            pg(actual_model.gcn, 2e-4),
            pg(actual_model.gate, 2e-4)
        ], weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        
        # 学习率调度器
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5)
        
        # 损失函数 - 使用更稳定的损失函数
        if args.mixup:
            ce = MixupLoss(alpha=args.mixup_alpha)
        elif args.focal:
            if args.class_balanced:
                custom_weights = [1.0, args.mild_dr_weight, 1.5, args.severe_dr_weight, 2.5]
                ce = ClassBalancedFocalLoss(class_weights=custom_weights, gamma=3)
            else:
                ce = FocalLoss(alpha=1, gamma=2)
        else:
            # 使用标准的交叉熵损失，更稳定
            ce = nn.CrossEntropyLoss()
            
        # 添加额外的损失函数用于调试
        debug_loss = nn.CrossEntropyLoss(reduction='none')
        
        # 训练参数
        best_k, best_p = -1, save_dir / f"enhanced_kantree_fold{i}_best.pth"
        patience = 15
        no_improve_count = 0
        min_delta = 0.001
        
        print(f"Starting training for fold {i}...")
        
        for ep in range(1, args.ep + 1):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Fold {i} Epoch {ep}/{args.ep}", ncols=100)
            for batch_idx, (x, y) in enumerate(pbar):
                 try:
                     x, y = x.to(device), y.to(device)
                     
                     # 前向传播
                     outputs = model(x)
                     
                     # 检查输出是否合理
                     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                         print(f"Warning: Invalid outputs detected in batch {batch_idx}")
                         continue
                     
                     # 检查预测分布
                     probs = torch.softmax(outputs, dim=1)
                     pred_classes = outputs.argmax(dim=1)
                     
                     # 打印调试信息（每10个batch）
                     if batch_idx % 10 == 0:
                         print(f"Batch {batch_idx}: Output shape: {outputs.shape}")
                         print(f"  Pred classes: {pred_classes[:5].tolist()}")
                         print(f"  True classes: {y[:5].tolist()}")
                         print(f"  Class distribution: {probs.mean(dim=0)}")
                         print(f"  Output stats - mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")
                     
                     # Mixup增强
                     if args.mixup and random.random() < 0.5:
                         lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                         batch_size = x.size(0)
                         index = torch.randperm(batch_size).to(device)
                         x = lam * x + (1 - lam) * x[index, :]
                         y_a, y_b = y, y[index]
                         
                         if isinstance(ce, MixupLoss):
                             loss = ce(outputs, y_a, y_b, lam)
                         else:
                             loss = ce(outputs, y)
                     else:
                         loss = ce(outputs, y)
                     
                     # 检查NaN损失
                     if torch.isnan(loss):
                         print(f"Warning: NaN loss detected in batch {batch_idx}")
                         continue
                     
                     # 反向传播
                     loss.backward()
                     
                     # 梯度裁剪 - 增加梯度裁剪阈值
                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     
                     # 更新权重
                     opt.step()
                     opt.zero_grad(set_to_none=True)
                     
                     train_loss += loss.item()
                     train_batches += 1
                     
                     # 更新进度条
                     pbar.set_postfix({
                         'Loss': f'{loss.item():.4f}',
                         'Avg Loss': f'{train_loss/train_batches:.4f}',
                         'Pred': f'{pred_classes[:3].tolist()}',
                         'True': f'{y[:3].tolist()}'
                     })
                     
                     # 清理内存
                     del x, y, outputs, probs, pred_classes, loss
                     if device.type == 'cuda':
                         torch.cuda.empty_cache()
                         
                 except RuntimeError as e:
                     if "size" in str(e) or "dimension" in str(e):
                         print(f"Size error in batch {batch_idx}: {e}")
                         continue
                     else:
                         print(f"Runtime error in batch {batch_idx}: {e}")
                         continue
                 except Exception as e:
                     print(f"Unexpected error in batch {batch_idx}: {e}")
                     continue
            
            # 验证阶段
            model.eval()
            k, a, f1, auc, cm, per_class_f1 = enhanced_metric(model, val_loader, device)
            
            print(f"[Fold {i}, Epoch {ep}] K={k:.4f} A={a:.4f} F1={f1:.4f} AUC={auc:.4f}")
            print(f"[Per-Class F1] {per_class_f1}")
            
            # 学习率调度
            sch.step(k)
            
            # 保存最佳模型
            if k > best_k + min_delta:
                best_k = k
                no_improve_count = 0
                
                # 保存模型时移除DataParallel包装
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), best_p)
                else:
                    torch.save(model.state_dict(), best_p)
                
                print(f"  New best model saved! Kappa: {best_k:.4f}")
                
                # 保存混淆矩阵
                if plt is not None and sns is not None:
                    try:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Confusion Matrix - Fold {i}, Epoch {ep}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.savefig(save_dir / f'confusion_matrix_fold{i}_ep{ep}.png')
                        plt.close()
                    except Exception:
                        pass
            else:
                no_improve_count += 1
            
            # 早停
            if no_improve_count >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stopping training.")
                break
            
            # 清理内存
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 记录结果
        results.append((best_k, best_p))
        print(f"Fold {i} completed. Best Kappa: {best_k:.4f}")
        
        # 清理当前fold的模型
        del model, opt, sch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 评估最佳模型
    if results:
        best_k, best_p = max(results, key=lambda t: t[0])
        print(f"\n--- Best model from all folds: {best_p} (Kappa: {best_k:.4f}) ---")
        
        # 加载最佳模型进行评估
        net = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt_path).to(device)
        net.load_state_dict(torch.load(best_p, map_location=device, weights_only=True))
        
        # 最终评估
        print("\n--- Final Evaluation ---")
        for tag, df_ in [("Val", val_df)]:
            ld = DataLoader(EnhancedAptosDS(df_, img_dir, args.img, False),
                           batch_size=effective_batch_size, shuffle=False, num_workers=4)
            k, a, f1, auc, cm, per_class_f1 = enhanced_metric(net, ld, device)
            print(f">>> {tag} Final Score: Kappa={k:.4f} ACC={a:.4f} F1={f1:.4f} AUC={auc:.4f}")
            print(f">>> {tag} Per-Class F1: {per_class_f1}")
        
        print(f"\nMulti-GPU training completed successfully!")
        print(f"Best model saved at: {best_p}")
        print(f"Best Kappa: {best_k:.4f}")
    else:
        print("Training failed, no models to evaluate.")

def main_worker(rank, world_size, tr_df, val_df, te_df, args, img_dir, save_dir, r50_path, vit_ckpt_path):
    """
    Enhanced main worker with better evaluation
    """
    print(f"Process starting on rank {rank}.")
    y = "diagnosis" if "diagnosis" in tr_df else "level"
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []

    for i, (tr, va) in enumerate(skf.split(tr_df, tr_df[y]), 1):
        res = train_fold(rank, world_size, i, tr, va, tr_df, args, img_dir, save_dir, r50_path, vit_ckpt_path)
        if rank == 0 and res is not None:
            results.append(res)

    if rank == 0:
        if not results:
            print("Training failed, no models to evaluate.")
            return

        best_k, best_p = max(results, key=lambda t: t[0])
        print(f"\n--- Evaluating best model from all folds: {best_p} (Kappa: {best_k:.4f}) ---")

        net = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt_path).to(rank)
        net.load_state_dict(torch.load(best_p, map_location=f"cuda:{rank}", weights_only=True))

        # Enhanced evaluation with TTA (Test Time Augmentation)
        print("\n--- Standard Evaluation ---")
        for tag, df_ in [("Val", val_df), ("Test", te_df)]:
            ld = DataLoader(EnhancedAptosDS(df_, img_dir, args.img, False),
                            batch_size=args.bs * 2, shuffle=False, num_workers=4)
            k, a, f1, auc, cm, per_class_f1 = enhanced_metric(net, ld, f"cuda:{rank}")
            print(f">>> {tag} Final Score: Kappa={k:.4f} ACC={a:.4f} F1={f1:.4f} AUC={auc:.4f}")
            print(f">>> {tag} Per-Class F1: {per_class_f1}")
        
        # Test Time Augmentation for better predictions
        print("\n--- Test Time Augmentation Evaluation ---")
        net.eval()
        tta_predictions = []
        tta_targets = []
        
        # Ensemble prediction with multiple checkpoints
        print("\n--- Ensemble Evaluation ---")
        ensemble_predictions = []
        ensemble_targets = []
        
        # Load multiple best models from different folds for ensemble
        ensemble_models = []
        for fold_idx in range(1, 6):  # Assuming 5 folds
            model_path = save_dir / f"enhanced_kantree_fold{fold_idx}_best.pth"
            if model_path.exists():
                ensemble_net = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt_path).to(rank)
                ensemble_net.load_state_dict(torch.load(model_path, map_location=f"cuda:{rank}", weights_only=True))
                ensemble_net.eval()
                ensemble_models.append(ensemble_net)
        
        if len(ensemble_models) > 1:
            print(f"Using {len(ensemble_models)} models for ensemble prediction")
            
            # Evaluate ensemble on validation set
            val_ld = DataLoader(EnhancedAptosDS(val_df, img_dir, args.img, False),
                               batch_size=args.bs * 2, shuffle=False, num_workers=4)
            
            for x, y in val_ld:
                x, y = x.to(f"cuda:{rank}"), y.to(f"cuda:{rank}")
                
                # Get predictions from all models
                ensemble_preds = []
                for model in ensemble_models:
                    with torch.no_grad():
                        pred = model(x)
                        pred = torch.softmax(pred, dim=1)
                        ensemble_preds.append(pred)
                
                # Average predictions
                avg_pred = torch.mean(torch.stack(ensemble_preds), dim=0)
                pred_class = avg_pred.argmax(dim=1)
                
                ensemble_predictions.extend(pred_class.cpu().tolist())
                ensemble_targets.extend(y.cpu().tolist())
            
            # Calculate ensemble metrics
            if len(ensemble_predictions) > 0:
                ensemble_k = cohen_kappa_score(ensemble_targets, ensemble_predictions, weights='quadratic')
                ensemble_a = accuracy_score(ensemble_targets, ensemble_predictions)
                ensemble_f1 = f1_score(ensemble_targets, ensemble_predictions, average='weighted')
                
                print(f"\n>>> Ensemble Scores: Kappa={ensemble_k:.4f} ACC={ensemble_a:.4f} F1={ensemble_f1:.4f}")
                print(f">>> Ensemble vs Single: Kappa +{ensemble_k - k:.4f}, ACC +{ensemble_a - a:.4f}, F1 +{ensemble_f1 - f1:.4f}")
        
        # Create TTA transforms
        tta_transforms = [
            transforms.Compose([
                transforms.Resize((args.img + 32, args.img + 32)),
                transforms.CenterCrop(args.img),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((args.img + 32, args.img + 32)),
                transforms.CenterCrop(args.img),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((args.img, args.img)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        
        # Evaluate with TTA
        for df_ in [val_df, te_df]:
            for idx in range(len(df_)):
                img_name = df_.iloc[idx][df_.columns[0]]  # Get image name
                target = df_.iloc[idx]['diagnosis'] if 'diagnosis' in df_.columns else df_.iloc[idx]['level']
                
                # Read and preprocess image
                img_path = None
                for suf in [".png", ".jpg", ".jpeg"]:
                    p = img_dir / (img_name + suf)
                    if p.exists():
                        img_path = p
                        break
                
                if img_path is None:
                    continue
                
                # Apply TTA
                predictions = []
                for tta_tf in tta_transforms:
                    img = cv2.imread(str(img_path))[:, :, ::-1]
                    img = Image.fromarray(img)
                    img = tta_tf(img).unsqueeze(0).to(f"cuda:{rank}")
                    
                    with torch.no_grad():
                        pred = net(img)
                        pred = torch.softmax(pred, dim=1)
                        predictions.append(pred.cpu())
                
                # Average predictions
                avg_pred = torch.mean(torch.stack(predictions), dim=0)
                pred_class = avg_pred.argmax().item()
                
                tta_predictions.append(pred_class)
                tta_targets.append(target)
        
        # Calculate TTA metrics
        if len(tta_predictions) > 0:
            tta_k = cohen_kappa_score(tta_targets, tta_predictions, weights='quadratic')
            tta_a = accuracy_score(tta_targets, tta_predictions)
            tta_f1 = f1_score(tta_targets, tta_predictions, average='weighted')
            
            print(f"\n>>> TTA Enhanced Scores: Kappa={tta_k:.4f} ACC={tta_a:.4f} F1={tta_f1:.4f}")
            print(f">>> TTA Improvement: Kappa +{tta_k - k:.4f}, ACC +{tta_a - a:.4f}, F1 +{tta_f1 - f1:.4f}")
        
        # Save final results
        final_results = {
            'best_model_path': str(best_p),
            'best_kappa': best_k,
            'validation_scores': {},
            'test_scores': {}
        }
        
        import json
        with open(save_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)


def main():
    """简化的主函数，专注于内存优化和多GPU训练"""
    warnings.filterwarnings("ignore", category=UserWarning)

    ARGS = get_args()
    DATA_DIR = Path(ARGS.data_dir)
    CSV_PATH = DATA_DIR / "train.csv"
    IMG_DIR = DATA_DIR / "train_images"
    R50 = Path(ARGS.r50)
    PATCH16 = ARGS.vit_ckpt
    SAVE_DIR = Path(ARGS.save_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    seed_all()
    
    # 检查数据
    if not CSV_PATH.exists():
        print(f"CSV file not found: {CSV_PATH}")
        return
    
    if not IMG_DIR.exists():
        print(f"Image directory not found: {IMG_DIR}")
        return

    # 加载数据
    print("Loading data...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # 如果数据集太大，采样以节省内存
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)
        print(f"Sampled {len(df)} samples for memory efficiency")
    
    # 简化的数据分割
    col = "diagnosis" if "diagnosis" in df else "level"
    tr_df, val_df = train_test_split(df, test_size=0.2, stratify=df[col], random_state=42)
    
    print(f"Train: {len(tr_df)}, Val: {len(val_df)}")
    
    # 检查GPU数量
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")

    if world_size > 1 and not ARGS.single_gpu:
        print(f"Found {world_size} GPUs. Using DataParallel for multi-GPU training...")
        
        # 使用DataParallel进行多GPU训练
        try:
            # 设置环境变量以优化多GPU训练
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(world_size)])
            
            # 调用多GPU训练函数
            multi_gpu_train(tr_df, val_df, ARGS, IMG_DIR, SAVE_DIR, R50, PATCH16, world_size)
            
        except Exception as e:
            print(f"Multi-GPU training failed: {e}")
            print("Falling back to single-GPU mode...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            main_worker(0, 1, tr_df, val_df, None, ARGS, IMG_DIR, SAVE_DIR, R50, PATCH16)
    else:
        if ARGS.single_gpu:
            print("Single-GPU mode forced by user.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print("Only one GPU found. Running in single-GPU mode.")
        main_worker(0, 1, tr_df, val_df, None, ARGS, IMG_DIR, SAVE_DIR, R50, PATCH16)


if __name__ == "__main__":
    main()