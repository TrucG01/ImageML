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
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    def draw_legend_area(image: np.ndarray, class_id_to_color, class_names, box_width=180, box_height=20, font_scale=0.5, font_thickness=1):
        # Creates a new canvas with the legend area to the right of the image
        import cv2
        h, w, _ = image.shape
        legend_area = np.zeros((h, box_width, 3), dtype=np.uint8) + 30  # dark background
        x0, y0 = 10, 10
        for idx, (class_id, color) in enumerate(class_id_to_color.items()):
            y = y0 + idx * box_height
            cv2.rectangle(legend_area, (x0, y), (x0 + box_height, y + box_height), color, -1)
            text = class_names[class_id]
            cv2.putText(legend_area, text, (x0 + box_height + 5, y + box_height - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
        # Draw border around legend area
        cv2.rectangle(legend_area, (x0-5, y0-5), (box_width-5, y0+len(class_id_to_color)*box_height+5), (0,0,0), 2)
        # Concatenate image and legend area horizontally
        combined = np.concatenate([image, legend_area], axis=1)
        return combined

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        # Pillow >=9.1 uses Image.Resampling.BILINEAR, older uses Image.BILINEAR
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
            # Softmax for confidence
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
            # --- NEW LOGIC: Only label and draw largest contour per class ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.3, min(0.7, img_np.shape[0] / 512 * 0.5))
            font_thickness = max(2, int(img_np.shape[0] * 0.005))
            contour_thickness = max(3, int(img_np.shape[0] * 0.01))
            for class_id, class_color in CLASS_ID_TO_COLOR.items():
                class_mask = (pred == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Draw high-contrast contour (thick, with white outline)
                    cv2.drawContours(overlay, [largest_contour], -1, (255,255,255), contour_thickness+2)
                    cv2.drawContours(overlay, [largest_contour], -1, tuple(class_color), contour_thickness)
                    # Place label at centroid
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label_text = f"{CLASS_NAMES[class_id]}"
                        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                        box_x1 = max(0, cx - text_w // 2 - 2)
                        box_y1 = max(0, cy - text_h // 2 - 2)
                        box_x2 = min(img_np.shape[1], cx + text_w // 2 + 2)
                        box_y2 = min(img_np.shape[0], cy + text_h // 2 + 2)
                        if box_x2 > box_x1 and box_y2 > box_y1:
                            overlay_box = overlay.copy()
                            cv2.rectangle(overlay_box, (box_x1, box_y1), (box_x2, box_y2), (0,0,0), -1)
                            alpha = 0.4
                            overlay[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(
                                overlay[box_y1:box_y2, box_x1:box_x2], 1-alpha,
                                overlay_box[box_y1:box_y2, box_x1:box_x2], alpha, 0)
                            text_color = (255,255,255)
                            cv2.putText(overlay, label_text, (box_x1+2, box_y2-2), font, font_scale, (0,0,0), font_thickness+2, cv2.LINE_AA)
                            cv2.putText(overlay, label_text, (box_x1+2, box_y2-2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            overlay = draw_legend_area(overlay, CLASS_ID_TO_COLOR, CLASS_NAMES)
        else:
            overlay = (0.6 * img_np + 0.4 * pred_rgb).astype(np.uint8)
            # If cv2 not available, skip legend area and contours
        frames.append(overlay)
        # Optionally save individual frames
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
    # Convert CLI string arguments to Path objects
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    infer_sequence(model, device, image_dir, output_dir, tuple(args.image_size))

if __name__ == "__main__":
    main()
