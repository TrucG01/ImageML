"""
Run inference on a video sequence using a trained DAVID segmentation model.
Saves a playback video with predicted segmentation overlays.
"""
import argparse
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import cv2
import numpy as np

from david_backend.model import build_model
from david_backend.data import encode_label_image, colorize_label_map, DEFAULT_MEAN, DEFAULT_STD


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return model

def infer_sequence(model, device, image_dir, output_dir, image_size=(256,256)):
    image_paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = image.resize(image_size, Image.BILINEAR)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, DEFAULT_MEAN, DEFAULT_STD)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)["out"]
            pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
            pred_rgb = colorize_label_map(pred)
        # Overlay prediction on original image
        img_np = np.array(image)
        overlay = cv2.addWeighted(img_np, 0.6, pred_rgb, 0.4, 0)
        frames.append(overlay)
        # Optionally save individual frames
        out_path = output_dir / f"{img_path.stem}_overlay.png"
        Image.fromarray(overlay).save(out_path)
    # Save video
    video_path = str(output_dir / "inference_playback.mp4")
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved playback video to {video_path}")

def main():
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
    infer_sequence(model, device, args.image_dir, args.output_dir, tuple(args.image_size))

if __name__ == "__main__":
    main()
