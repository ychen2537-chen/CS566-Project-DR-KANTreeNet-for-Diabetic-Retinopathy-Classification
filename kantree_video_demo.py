import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Directly use the model class defined in All_KANS_Sencond.py
from All_KANS_Sencond import EnhancedDRKANTreeNet  # Ensure the filename matches


# Normalization parameters consistent with the dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 448

preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_image_rgb(path: str):
    """Read image using cv2 and convert BGR->RGB"""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def fig_to_rgb_array(fig):
    """matplotlib Figure -> H×W×3 numpy"""
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image = buf.reshape((height, width, 4))  # RGBA
    image = image[:, :, :3]  # Extract RGB (drop alpha channel)
    plt.close(fig)
    return image


def make_image_frame(image_rgb: np.ndarray, title: str = "", description: str = "", figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_rgb)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16)
    if description:
        fig.text(0.5, 0.02, description, ha="center", fontsize=11)
    frame = fig_to_rgb_array(fig)
    return frame


def make_probability_frame(probs: np.ndarray, class_names, pred_idx: int, figsize=(6, 4)):
    """Create a bar chart showing final classification probabilities"""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(class_names))
    ax.bar(x, probs, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title(f"Prediction: {class_names[pred_idx]}")
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    frame = fig_to_rgb_array(fig)
    return frame


def overlay_heatmap_on_image(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap on image.
    heatmap: 2D numpy array with values in [0, 1], will be resized to match img_rgb size externally
    """
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def normalize_tensor_to_0_1(t: torch.Tensor) -> np.ndarray:
    """Normalize any tensor to [0, 1] range and return as numpy array"""
    t = t.detach().float()
    t = t - t.min()
    maxv = t.max()
    if maxv > 0:
        t = t / maxv
    return t.cpu().numpy()


def extract_intermediates(model: EnhancedDRKANTreeNet, x: torch.Tensor):
    """
    Extract intermediate results from the model.
    
    Args:
        x: Input tensor that has been resized and normalized (1, 3, IMG_SIZE, IMG_SIZE)
    
    Returns:
        A dictionary containing:
          - vessel_mask: (H, W) Vessel mask
          - lesion_attn: (h, w) Lesion attention heatmap (LesionAttention)
          - dam_attn:    (h, w) Feature intensity after DAM enhancement
          - vit_heatmap: (14, 14) ViT-S patch token norm
          - logits:      (1, 5) Final logits (including GCN head)
    """
    model.eval()

    # ------- 1) ResNet multi-scale features -------
    with torch.no_grad():
        res_features = model.res(x)
        final_features = res_features[4]  # (B, 2048, h, w)
        mid_features = res_features[2]
        low_features = res_features[1]

        # Vessel mask from EnhancedVesselTreeNet.get_vessel_mask
        vessel_mask = model.vessel_net.get_vessel_mask(x)[0, 0]  # (H, W)

    # ------- 2) Lesion attention + DAM -------
    lesion_module = model.enhanced_attention
    dam_module = model.res_dam

    with torch.no_grad():
        B, C, H_in, W_in = x.shape
        _, _, Hf, Wf = final_features.shape
        if (H_in, W_in) != (Hf, Wf):
            x_resized = F.interpolate(x, size=(Hf, Wf), mode="bilinear", align_corners=False)
        else:
            x_resized = x

        # Channel attention (avg + max pool)
        avg_pool = F.adaptive_avg_pool2d(final_features, 1)
        max_pool = F.adaptive_max_pool2d(final_features, 1)
        channel_attn = lesion_module.sigmoid(
            lesion_module.conv2(
                lesion_module.relu(
                    lesion_module.conv1(avg_pool + max_pool)
                )
            )
        )

        # Lesion attention from original image
        lesion_feat = lesion_module.lesion_conv(x_resized)
        lesion_attn = lesion_module.lesion_attention(lesion_feat)

        if lesion_attn.shape[2:] != final_features.shape[2:]:
            lesion_attn = F.interpolate(lesion_attn, size=final_features.shape[2:], mode="bilinear", align_corners=False)
        if channel_attn.shape[2:] != final_features.shape[2:]:
            channel_attn = F.interpolate(channel_attn, size=final_features.shape[2:], mode="bilinear", align_corners=False)

        severe_attn = lesion_module.severe_dr_detector(final_features)

        combined_attn = channel_attn * lesion_attn * severe_attn

        # Apply attention and pass through KANDAM
        r_feats = final_features * combined_attn
        r_feats = dam_module(r_feats)

        lesion_attn_map = combined_attn.mean(dim=1, keepdim=True)  # (B, 1, h, w)
        dam_attn_map = r_feats.norm(dim=1, keepdim=True)          # (B, 1, h, w)

        lesion_attn_map = normalize_tensor_to_0_1(lesion_attn_map[0, 0])
        dam_attn_map = normalize_tensor_to_0_1(dam_attn_map[0, 0])

    # ------- 3) ViT-S global context heatmap -------
    with torch.no_grad():
        xv = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        vit_tokens = model.vitS.forward_features(xv)  # (B, 1+N, D), index 0 is CLS token
        if vit_tokens.dim() == 3 and vit_tokens.size(1) > 1:
            patch_tokens = vit_tokens[:, 1:, :]      # Remove CLS token
            B, N, D = patch_tokens.shape
            S = int(N ** 0.5)                        # 14×14 patches
            patch_tokens = patch_tokens.reshape(B, S, S, D)
            vit_map = patch_tokens.norm(dim=-1)      # (B, S, S)
            vit_map = normalize_tensor_to_0_1(vit_map[0])
        else:
            vit_map = np.zeros((14, 14), dtype=np.float32)

    # ------- 4) Final logits (call the original forward) -------
    with torch.no_grad():
        logits = model(x)  # (1, 5)

    return {
        "vessel_mask": normalize_tensor_to_0_1(vessel_mask),
        "lesion_attn": lesion_attn_map,
        "dam_attn": dam_attn_map,
        "vit_heatmap": vit_map,
        "logits": logits
    }


DEFAULT_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def generate_kantree_video(
    image_path: str = "train_images/000c1434d8d7.png",
    output_video_path: str = "kantree_analysis_demo.mp4",
    r50_path: str = "./resnet50-19c8e357.pth",
    vit_ckpt: str = "hf_hub:timm/vit_small_patch16_224.augreg_in21k",
    model_weights: str = None,
    class_names=None,
    fps: int = 2,
    step_duration_sec: float = 2.0,
    device: str = None,
):
    """
    Perform forward inference on a single DR image using EnhancedDRKANTreeNet,
    and create an MP4 video showing key intermediate steps.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 1) Load image & resize to 448×448 (for alignment with model)
    orig_rgb = load_image_rgb(image_path)
    resized_rgb = cv2.resize(orig_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 2) Convert to model input tensor (with normalization)
    pil_img = transforms.ToPILImage()(resized_rgb)
    input_tensor = preprocess_transform(pil_img).unsqueeze(0).to(device)

    # 3) Build model (weights consistent with All_KANS_Sencond.py)
    model = EnhancedDRKANTreeNet(r50_path=r50_path, vit_ckpt_path=vit_ckpt, n_cls=len(class_names))
    model.to(device)
    model.eval()

    # Load trained checkpoint if available
    if model_weights is not None and os.path.isfile(model_weights):
        state = torch.load(model_weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded weights with missing keys:", len(missing), "unexpected:", len(unexpected))

    # 4) Extract intermediate results
    inter = extract_intermediates(model, input_tensor)
    logits = inter["logits"]
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))

    # Resize all heatmaps to 448×448 for overlay
    vessel_mask = inter["vessel_mask"]
    lesion_attn = inter["lesion_attn"]
    dam_attn = inter["dam_attn"]
    vit_map = inter["vit_heatmap"]

    vessel_resized = cv2.resize(vessel_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    lesion_resized = cv2.resize(lesion_attn, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    dam_resized = cv2.resize(dam_attn, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    vit_resized = cv2.resize(vit_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # 5) Build video frames step by step
    frames = []

    frames.append(
        make_image_frame(
            orig_rgb,
            title="Step 1: Original Fundus Image",
            description="Original DR image loaded from disk."
        )
    )

    frames.append(
        make_image_frame(
            resized_rgb,
            title="Step 2: Resized to (448×448)",
            description="Resized to model input size and normalized."
        )
    )

    # Vessel tree branch
    vessel_vis = (vessel_resized * 255).astype(np.uint8)
    vessel_vis_rgb = cv2.cvtColor(vessel_vis, cv2.COLOR_GRAY2RGB)
    frames.append(
        make_image_frame(
            vessel_vis_rgb,
            title="Step 3: Vessel Tree Branch (VesselTreeNet)",
            description="Vessel-like structures extracted from green channel, input to EnhancedVesselTreeNet."
        )
    )

    # Lesion attention
    lesion_overlay = overlay_heatmap_on_image(resized_rgb, lesion_resized, alpha=0.5)
    frames.append(
        make_image_frame(
            lesion_overlay,
            title="Step 4: Lesion Attention",
            description="ResNet high-level features + original image, automatically focusing on suspected lesion regions."
        )
    )

    # DAM-enhanced local structures
    dam_overlay = overlay_heatmap_on_image(resized_rgb, dam_resized, alpha=0.5)
    frames.append(
        make_image_frame(
            dam_overlay,
            title="Step 5: DAM-Enhanced Local Structures",
            description="KANDAM module further strengthens discriminative textures and local structures."
        )
    )

    # ViT global context
    vit_overlay = overlay_heatmap_on_image(resized_rgb, vit_resized, alpha=0.5)
    frames.append(
        make_image_frame(
            vit_overlay,
            title="Step 6: ViT-S Global Context",
            description="Global attention heatmap based on ViT-S patch token norms."
        )
    )

    # Final classification result
    frames.append(
        make_probability_frame(
            probs,
            class_names=class_names,
            pred_idx=pred_idx,
        )
    )

    if not frames:
        raise RuntimeError("No frames generated.")

    # Each step stays for step_duration_sec seconds
    repeat = max(int(step_duration_sec * fps), 1)
    extended_frames = []
    for frame in frames:
        extended_frames.extend([frame] * repeat)

    height, width = frames[0].shape[:2]
    for i in range(len(extended_frames)):
        if extended_frames[i].shape[:2] != (height, width):
            extended_frames[i] = cv2.resize(extended_frames[i], (width, height))

    # Write MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in extended_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"[OK] Saved analysis video to: {output_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EnhancedDRKANTreeNet image analysis -> video demo")
    parser.add_argument("--image_path", type=str, default="train_images/000c1434d8d7.png",
                        help="Path to input fundus image.")
    parser.add_argument("--output_video", type=str, default="kantree_analysis_demo.mp4",
                        help="Output video path.")
    parser.add_argument("--r50", type=str, default="./resnet50-19c8e357.pth",
                        help="Optional ResNet-50 weight path used in All_KANS_Sencond.py.")
    parser.add_argument("--vit_ckpt", type=str,
                        default="hf_hub:timm/vit_small_patch16_224.augreg_in21k",
                        help="ViT checkpoint string/path (same as in All_KANS_Sencond.py).")
    parser.add_argument("--weights", type=str, default=None,
                        help="(Optional) Trained EnhancedDRKANTreeNet checkpoint (.pth).")
    parser.add_argument("--fps", type=int, default=2, help="Video frames per second.")
    parser.add_argument("--step_sec", type=float, default=2.0,
                        help="Seconds for each explanation step.")
    args = parser.parse_args()

    generate_kantree_video(
        image_path=args.image_path,
        output_video_path=args.output_video,
        r50_path=args.r50,
        vit_ckpt=args.vit_ckpt,
        model_weights=args.weights,
        fps=args.fps,
        step_duration_sec=args.step_sec,
    )
