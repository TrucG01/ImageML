"""Run inference on a video sequence using a trained DAVID segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CV2_AVAILABLE = False

from model_backend.data import (
    CLASS_ID_TO_COLOR,
    CLASS_NAMES,
    DEFAULT_MEAN,
    DEFAULT_STD,
    colorize_label_map,
)
from model_backend.model import build_model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load model weights from checkpoint file.
    Args:
        model: Model instance.
        checkpoint_path: Path to checkpoint file.
    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return model

def infer_sequence(
    model: torch.nn.Module,
    device: torch.device,
    image_dir: Path,
    output_dir: Path,
    image_size: tuple = (256, 256)
) -> None:
    """Run inference on a sequence of images and save overlays/video."""
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}. Please check the path and file types.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    def draw_legend_area(image: np.ndarray, class_id_to_color, class_names, box_width=180, box_height=20, font_scale=0.5, font_thickness=1):
        # Creates a new canvas with the legend area to the right of the image
        import cv2
        h, w = image.shape[:2]
        
        # Legend width
        legend_w = 260  
        legend_area = np.zeros((h, legend_w, 3), dtype=np.uint8) + 30  # dark background
        
        x0, y0 = 10, 10
        for idx, (class_id, color) in enumerate(class_id_to_color.items()):
            y = y0 + idx * box_height
            cv2.rectangle(legend_area, (x0, y), (x0 + box_height, y + box_height), color, -1)
            
            # Restore class name text in the legend
            if class_id < len(class_names):
                text = class_names[class_id]
            else:
                text = f"Class {class_id}"
                
            cv2.putText(legend_area, text, (x0 + box_height + 5, y + box_height - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
        
        # Draw border around legend area
        cv2.rectangle(legend_area, (x0-5, y0-5), (legend_w-5, y0+len(class_id_to_color)*box_height+5), (0,0,0), 2)
        
        # Add line thickness key below colors
        key_y = y0 + len(class_id_to_color)*box_height + 20
        
        # Boundary line (1px)
        cv2.line(legend_area, (x0+10, key_y), (x0+70, key_y), (200,200,200), 1)
        cv2.putText(legend_area, 'Boundary (>50%)', (x0+80, key_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
        
        # High Conf line (3px)
        cv2.line(legend_area, (x0+10, key_y+25), (x0+70, key_y+25), (200,200,200), 3)
        cv2.putText(legend_area, 'High Conf (>80%)', (x0+80, key_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
        
        # Explicit Data Type Safety before merging
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Robust Resize check to avoid crash
        img_h = image.shape[0]
        leg_h = legend_area.shape[0]
        if img_h != leg_h:
            legend_area = cv2.resize(legend_area, (legend_area.shape[1], img_h))
            
        combined = np.concatenate([image, legend_area], axis=1)
        return combined

    for idx, img_path in enumerate(image_paths):
        print(f"[INFO] Processing image {idx+1}/{len(image_paths)}: {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.BILINEAR
        else:
            resample = getattr(Image, "BILINEAR", 2)
            
        image = image.resize(image_size, resample)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, DEFAULT_MEAN, DEFAULT_STD)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)["out"]
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred = pred.squeeze().cpu().numpy().astype(np.uint8)
            conf = conf.squeeze().cpu().numpy()
            pred_rgb = colorize_label_map(pred)
        
        # Overlay prediction on original image
        img_np = np.array(image)
        
        if CV2_AVAILABLE:
            import cv2
            overlay = cv2.addWeighted(img_np, 0.6, pred_rgb, 0.4, 0)
            
            # Draw topographical contours for each class (multi-object)
            for class_id, class_color in CLASS_ID_TO_COLOR.items():
                # Convert color to BGR for OpenCV
                bgr_color = tuple(int(c) for c in class_color[::-1])
                img_area_total = pred.shape[0] * pred.shape[1]
                
                # --- BASE LAYER: prob > 0.5 ---
                base_mask = ((probs[0, class_id] > 0.5).cpu().numpy().astype(np.uint8))
                # Slight blur for base to smooth edges
                base_mask_blur = cv2.GaussianBlur(base_mask, (5,5), 0)
                base_contours, _ = cv2.findContours(base_mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Iterate through all significant contours (multi-object)
                for c in base_contours:
                     area = cv2.contourArea(c)
                     # Filter tiny noise
                     if area > 0.005 * img_area_total:
                        peri = cv2.arcLength(c, True)
                        smooth_base = cv2.approxPolyDP(c, 0.001 * peri, True)
                        
                        # Draw base boundary: Thickness 1, slightly transparent logic
                        overlay_base = overlay.copy()
                        cv2.drawContours(overlay_base, [smooth_base], -1, bgr_color, 1, lineType=cv2.LINE_AA)
                        overlay = cv2.addWeighted(overlay, 0.4, overlay_base, 0.6, 0)
                
                # --- CORE LAYER: prob > 0.8 ---
                core_mask = ((probs[0, class_id] > 0.8).cpu().numpy().astype(np.uint8))
                # Use reduced blur (3,3) for sharper high-confidence regions
                core_mask_blur = cv2.GaussianBlur(core_mask, (3,3), 0)
                core_contours, _ = cv2.findContours(core_mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Iterate through all significant contours
                for c in core_contours:
                    area = cv2.contourArea(c)
                    
                    # Threshold: Only show core if it's significant (> 2% of screen)
                    if area > 0.02 * img_area_total:
                        peri = cv2.arcLength(c, True)
                        smooth_core = cv2.approxPolyDP(c, 0.001 * peri, True)
                        
                        # Draw core boundary: Thickness 2 (Solid/Thick)
                        cv2.drawContours(overlay, [smooth_core], -1, bgr_color, 2, lineType=cv2.LINE_AA)
                        
                        # NO TEXT LABELS
            
            # Attach the legend
            overlay = draw_legend_area(overlay, CLASS_ID_TO_COLOR, CLASS_NAMES)
            
        else:
            overlay = (0.6 * img_np + 0.4 * pred_rgb).astype(np.uint8)
            
        frames.append(overlay)
        out_path = output_dir / f"{img_path.stem}_overlay.png"
        Image.fromarray(overlay).save(out_path)
        
    # Save video
    if CV2_AVAILABLE and frames:
        import cv2
        video_path = str(output_dir / "inference_playback.mp4")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved playback video to {video_path}")
    elif CV2_AVAILABLE:
        print("No frames were processed; skipping video creation.")
    else:
        print("OpenCV (cv2) not installed. Skipping video creation. Overlays saved as PNGs.")

def main() -> None:
    """
    Main entrypoint for DAVID video inference script.
    """
    parser = argparse.ArgumentParser(description="DAVID Segmentation Inference on Video Sequence")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (best_model.pth)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of input video frames")
    parser.add_argument("--output_dir", type=str, default="outputs/inference", help="Directory to save overlays and video")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256,256], help="Resize images to (H,W)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=13, pretrained=False)
    model = load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    infer_sequence(model, device, image_dir, output_dir, tuple(args.image_size))

if __name__ == "__main__":
    main()