"""
Run inference on a video sequence using a trained DAVID segmentation model.
Saves a playback video with predicted segmentation overlays.
"""
import argparse
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from PIL import Image
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import numpy as np

from david_backend.model import build_model
from david_backend.data import encode_label_image, colorize_label_map, DEFAULT_MEAN, DEFAULT_STD


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
    """
    Run inference on a sequence of images and save overlays/video.
    Args:
        model: Trained segmentation model.
        device: Target device.
        image_dir: Path to directory of input frames.
        output_dir: Path to directory to save overlays/video.
        image_size: Resize images to this size.
    """
    """
    Run inference on a sequence of images and save overlays/video.
    Args:
        model: Trained segmentation model.
        device: Target device.
        image_dir: Directory of input frames.
        output_dir: Directory to save overlays/video.
        image_size: Resize images to this size.
    """
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    output_dir.mkdir(parents=True, exist_ok=True)
    from david_backend.data import CLASS_ID_TO_COLOR, CLASS_NAMES
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
            pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
            pred_rgb = colorize_label_map(pred)
        # Overlay prediction on original image
        img_np = np.array(image)
        if CV2_AVAILABLE:
            import cv2
            overlay = cv2.addWeighted(img_np, 0.6, pred_rgb, 0.4, 0)
            overlay = draw_legend_area(overlay, CLASS_ID_TO_COLOR, CLASS_NAMES)
        else:
            overlay = (0.6 * img_np + 0.4 * pred_rgb).astype(np.uint8)
            # If cv2 not available, skip legend area
        frames.append(overlay)
        # Optionally save individual frames
        out_path = output_dir / f"{img_path.stem}_overlay.png"
        Image.fromarray(overlay).save(out_path)
    # Save video
    if CV2_AVAILABLE:
        import cv2
        video_path = str(output_dir / "inference_playback.mp4")
        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved playback video to {video_path}")
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
