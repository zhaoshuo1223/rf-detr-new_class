# ------------------------------------------------------------------------
# RF-DETR ONNX Inference (Simple)
# - Based on rfdetr/infer-onnx_area.py
# - Keeps only: inference + visualization output
# - Removes: mask area calculation / txt / excel aggregation
# ------------------------------------------------------------------------

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a RF-DETR ONNX model (simple).")
    parser.add_argument(
        "--model",
        default=r"D:\aotto\budingban\model\hebing\03-04\576m\export\liao_576.onnx",
        type=str,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--image_dir",
        default=r"D:\aotto\budingban\model\test\0323-2\0323-2",
        type=str,
        help="Input image directory (or a single image path)",
    )
    parser.add_argument(
        "--output",
        default=r"D:\aotto\budingban\model\test\simple_out",
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


def imread_unicode(image_path: str):
    """Read image path that may contain Chinese characters (Windows)."""
    try:
        with open(image_path, "rb") as f:
            image_data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_providers(device: str = "gpu"):
    if device.lower() in ("gpu", "cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preprocess_image(image_bgr: np.ndarray, target_size):
    """
    Preprocess image:
    - resize to target_size (w,h)
    - BGR -> RGB
    - normalize with ImageNet mean/std
    - HWC -> CHW
    - add batch dimension
    """
    h0, w0 = image_bgr.shape[:2]
    target_w, target_h = target_size
    w_rate = w0 / target_w
    h_rate = h0 / target_h

    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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

    # sort and cap
    sorted_idx = np.argsort(scores)[::-1][:max_number_boxes]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    boxes = boxes[sorted_idx]
    if masks is not None:
        masks = masks[sorted_idx]

    # threshold
    keep = scores > conf_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    if masks is not None:
        masks = masks[keep]

    boxes_xyxy_model = xywh_cxcywh_to_xyxy_abs(boxes, target_size)
    return scores, labels, boxes_xyxy_model, masks


def draw_detections(image_bgr: np.ndarray, boxes_xyxy_orig: np.ndarray, labels: np.ndarray, scores: np.ndarray):
    out = image_bgr.copy()
    font_scale = max(0.5, min(2.0, min(out.shape[0], out.shape[1]) / 1000.0))
    thickness = max(1, int(font_scale * 2))
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes_xyxy_orig[i].astype(int).tolist()
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{int(labels[i])} {float(scores[i]):.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = y1 - 5
        if y_text - th - bl < 0:
            y_text = y2 + th + bl + 5
        cv2.rectangle(out, (x1, y_text - th - bl), (x1 + tw + 6, y_text + 2), color, -1)
        cv2.putText(out, text, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return out


def main():
    args = parse_args()

    try:
        w_s, h_s = args.model_shape.split(",")
        model_shape = (int(w_s.strip()), int(h_s.strip()))
    except Exception as e:
        raise ValueError(f"Invalid --model_shape '{args.model_shape}', expected 'width,height'") from e

    os.makedirs(args.output, exist_ok=True)
    out_vis_dir = os.path.join(args.output, "visualized")
    out_mask_dir = os.path.join(args.output, "masks")
    os.makedirs(out_vis_dir, exist_ok=True)
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
        image = imread_unicode(image_path)
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

        # map boxes to original image coordinates
        boxes_xyxy_orig = boxes_xyxy_model.copy()
        boxes_xyxy_orig[:, [0, 2]] *= w_rate
        boxes_xyxy_orig[:, [1, 3]] *= h_rate
        boxes_xyxy_orig[:, [0, 2]] = np.clip(boxes_xyxy_orig[:, [0, 2]], 0, w0 - 1)
        boxes_xyxy_orig[:, [1, 3]] = np.clip(boxes_xyxy_orig[:, [1, 3]], 0, h0 - 1)

        vis = draw_detections(image, boxes_xyxy_orig, labels, scores)
        stem = Path(image_path).stem
        vis_path = os.path.join(out_vis_dir, f"{stem}_det.jpg")
        cv2.imwrite(vis_path, vis)

        # optionally save masks (raw -> resized -> binarized)
        if args.save_masks and masks is not None and len(masks) == len(scores):
            for i in range(len(scores)):
                m = masks[i].astype(np.float32)
                m_resized = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
                m_bin = (m_resized > 0).astype(np.uint8) * 255
                m_path = os.path.join(out_mask_dir, f"{stem}_mask_{i}_cls{int(labels[i])}_{float(scores[i]):.2f}.png")
                cv2.imwrite(m_path, m_bin)

        ok += 1
        print(f"[{idx}/{len(image_files)}] det={len(scores)} infer={t_infer:.1f}ms -> {vis_path}")

    print(f"Done. success={ok}/{len(image_files)} total_time={time.time()-total_start:.2f}s")
    print(f"Outputs saved in: {args.output}")


if __name__ == "__main__":
    main()


 