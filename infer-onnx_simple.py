# ------------------------------------------------------------------------
# RF-DETR ONNX Inference (Simple)
# - Based on rfdetr/infer-onnx_area.py
# - Image I/O, resize, draw, save: Pillow (no OpenCV)
# - Removes: mask area calculation / txt / excel aggregation
# ------------------------------------------------------------------------

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a RF-DETR ONNX model (simple).")
    parser.add_argument(
        "--model",
        default=r"D:\aotto\budingban\model\yuqi_04-20-576m\export\yunqi-04_21-576m.onnx",
        type=str,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--image_dir",
        default=r"D:\aotto\budingban\data_set\aaa_hebing",
        type=str,
        help="Input image directory (or a single image path)",
    )
    parser.add_argument(
        "--output",
        default=r"D:\aotto\budingban\data_set\emp—out",
        type=str,
        help="Output directory to save visualized images",
    )
    parser.add_argument(
        "--threshold",
        default=0.65,
        type=float,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--low_conf_threshold",
        default=0.9,
        type=float,
        help="If any detection score is below this value, save the visualization into output/low_conf as well.",
    )
    parser.add_argument(
        "--max_number_boxes",
        default=300,
        type=int,
        help="Maximum number of boxes to return",
    )
    parser.add_argument(
        "--model_shape",
        default="576,576",
        type=str,
        help="Model input shape as 'width,height' (e.g. '512,512')",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="Device to use for inference: 'gpu' or 'cpu'",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="If the model outputs masks, also save per-object mask PNGs.",
    )
    return parser.parse_args()


def get_image_files(image_dir: str):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isfile(image_dir):
        return [image_dir]
    files = []
    for p in Path(image_dir).iterdir():
        if p.is_file() and p.suffix.lower() in image_extensions:
            files.append(str(p))
    return sorted(files)


def imread_pil(image_path: str) -> Image.Image | None:
    """Load image as RGB (supports Unicode paths on Windows via Pillow)."""
    try:
        with Image.open(image_path) as im:
            return im.convert("RGB")
    except OSError:
        return None


def get_providers(device: str = "gpu"):
    if device.lower() in ("gpu", "cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preprocess_image(image_rgb: Image.Image, target_size):
    """
    Preprocess image:
    - resize to target_size (w,h)
    - normalize with ImageNet mean/std
    - HWC -> CHW
    - add batch dimension
    """
    w0, h0 = image_rgb.size
    target_w, target_h = target_size
    w_rate = w0 / target_w
    h_rate = h0 / target_h

    resized = image_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    chw = np.transpose(rgb, (2, 0, 1))
    inp = np.expand_dims(chw, axis=0).astype(np.float32)
    return inp, (w_rate, h_rate), (w0, h0), resized


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xywh_cxcywh_to_xyxy_abs(boxes_cxcywh: np.ndarray, target_size):
    """boxes: (N,4) in cxcywh normalized to [0,1] -> xyxy absolute in model input size."""
    target_w, target_h = target_size
    cx, cy, w, h = (
        boxes_cxcywh[..., 0],
        boxes_cxcywh[..., 1],
        boxes_cxcywh[..., 2],
        boxes_cxcywh[..., 3],
    )
    x1 = (cx - w / 2.0) * target_w
    y1 = (cy - h / 2.0) * target_h
    x2 = (cx + w / 2.0) * target_w
    y2 = (cy + h / 2.0) * target_h
    return np.stack([x1, y1, x2, y2], axis=-1)


def postprocess(outputs, target_size, conf_threshold: float, max_number_boxes: int):
    """
    Expected RF-DETR ONNX outputs:
      - outputs[0]: boxes (1, Q, 4)  in cxcywh normalized
      - outputs[1]: logits (1, Q, C) raw logits
      - outputs[2] (optional): masks (1, Q, Hm, Wm)
    Returns filtered (scores, labels, boxes_xyxy_model, masks_raw_or_none)
    """
    boxes = outputs[0].squeeze(0)  # (Q,4)
    logits = outputs[1].squeeze(0)  # (Q,C)

    masks = None
    if len(outputs) >= 3 and outputs[2] is not None:
        masks = outputs[2].squeeze(0)  # (Q,Hm,Wm) maybe

    prob = sigmoid(logits)
    scores = np.max(prob, axis=1)
    labels = np.argmax(prob, axis=1)

    sorted_idx = np.argsort(scores)[::-1][:max_number_boxes]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    boxes = boxes[sorted_idx]
    if masks is not None:
        masks = masks[sorted_idx]

    keep = scores > conf_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    if masks is not None:
        masks = masks[keep]

    boxes_xyxy_model = xywh_cxcywh_to_xyxy_abs(boxes, target_size)
    return scores, labels, boxes_xyxy_model, masks


def _load_font(image_w: int, image_h: int) -> ImageFont.ImageFont:
    size = max(12, int(min(image_w, image_h) / 1000.0 * 24))
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image_rgb: Image.Image,
    boxes_xyxy_orig: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> Image.Image:
    out = image_rgb.copy()
    draw = ImageDraw.Draw(out)
    w_img, h_img = out.size
    font = _load_font(w_img, h_img)
    outline_w = max(1, min(w_img, h_img) // 400)
    green = (0, 255, 0)

    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes_xyxy_orig[i].astype(int).tolist()
        draw.rectangle([x1, y1, x2, y2], outline=green, width=outline_w)
        text = f"{int(labels[i])} {float(scores[i]):.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_text = y1 - 5 - th
        if y_text < 0:
            y_text = y2 + 5
        draw.rectangle([x1, y_text, x1 + tw + 6, y_text + th + 4], fill=green)
        draw.text((x1 + 3, y_text + 2), text, fill=(0, 0, 0), font=font)
    return out


def resize_mask_bilinear(mask_2d: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize float 2D mask to (W,H) using Pillow mode F."""
    w0, h0 = size_wh
    arr = np.ascontiguousarray(mask_2d.astype(np.float32))
    pil_m = Image.fromarray(arr, mode="F")
    pil_m = pil_m.resize((w0, h0), Image.Resampling.BILINEAR)
    return np.asarray(pil_m, dtype=np.float32)


def main():
    args = parse_args()

    try:
        w_s, h_s = args.model_shape.split(",")
        model_shape = (int(w_s.strip()), int(h_s.strip()))
    except Exception as e:
        raise ValueError(f"Invalid --model_shape '{args.model_shape}', expected 'width,height'") from e

    os.makedirs(args.output, exist_ok=True)
    out_vis_dir = os.path.join(args.output, "visualized")
    out_low_conf_dir = os.path.join(args.output, "low_conf")
    out_mask_dir = os.path.join(args.output, "masks")
    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(out_low_conf_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(out_mask_dir, exist_ok=True)

    image_files = get_image_files(args.image_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")

    providers = get_providers(args.device)
    session = onnxruntime.InferenceSession(args.model, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"Using provider: {actual_provider}")

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"Model outputs: {output_names}")

    total_start = time.time()
    ok = 0
    for idx, image_path in enumerate(image_files, 1):
        image = imread_pil(image_path)
        if image is None:
            print(f"[{idx}/{len(image_files)}] skip (read failed): {image_path}")
            continue

        inp, (w_rate, h_rate), (w0, h0), _ = preprocess_image(image, model_shape)

        t0 = time.time()
        outputs = session.run(None, {input_name: inp})
        t_infer = (time.time() - t0) * 1000.0

        scores, labels, boxes_xyxy_model, masks = postprocess(
            outputs, model_shape, args.threshold, args.max_number_boxes
        )

        boxes_xyxy_orig = boxes_xyxy_model.copy()
        boxes_xyxy_orig[:, [0, 2]] *= w_rate
        boxes_xyxy_orig[:, [1, 3]] *= h_rate
        boxes_xyxy_orig[:, [0, 2]] = np.clip(boxes_xyxy_orig[:, [0, 2]], 0, w0 - 1)
        boxes_xyxy_orig[:, [1, 3]] = np.clip(boxes_xyxy_orig[:, [1, 3]], 0, h0 - 1)

        vis = draw_detections(image, boxes_xyxy_orig, labels, scores)
        stem = Path(image_path).stem
        vis_name = f"{stem}_det.jpg"
        vis_path = os.path.join(out_vis_dir, vis_name)
        vis.save(vis_path, quality=95)

        if len(scores) > 0 and np.any(scores < float(args.low_conf_threshold)):
            low_path = os.path.join(out_low_conf_dir, vis_name)
            vis.save(low_path, quality=95)

        if args.save_masks and masks is not None and len(masks) == len(scores):
            for i in range(len(scores)):
                m_resized = resize_mask_bilinear(masks[i], (w0, h0))
                m_bin = (m_resized > 0).astype(np.uint8) * 255
                m_path = os.path.join(out_mask_dir, f"{stem}_mask_{i}_cls{int(labels[i])}_{float(scores[i]):.2f}.png")
                Image.fromarray(m_bin, mode="L").save(m_path)

        ok += 1
        print(f"[{idx}/{len(image_files)}] det={len(scores)} infer={t_infer:.1f}ms -> {vis_path}")

    print(f"Done. success={ok}/{len(image_files)} total_time={time.time()-total_start:.2f}s")
    print(f"Outputs saved in: {args.output}")


if __name__ == "__main__":
    main()
