import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


labelme_images = r"D:\aotto\quexian-jiance\data-20260331\images"
labelme_json = r"D:\aotto\quexian-jiance\data-20260331\images"
coco_dir = r"D:\aotto\quexian-jiance\data-20260331\coco"

ratio = (0.8, 0.1, 0.1)  # train, val, test
SEED = 42


def _validate_ratio(split_ratio: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(split_ratio) != 3:
        raise ValueError("ratio must be a 3-tuple: (train, val, test)")
    if any(r < 0 for r in split_ratio):
        raise ValueError("ratio values must be non-negative")
    total = sum(split_ratio)
    if total <= 0:
        raise ValueError("ratio sum must be > 0")
    return tuple(r / total for r in split_ratio)


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _shape_to_coco(shape: dict) -> tuple[list[float], list[list[float]], float]:
    points = shape.get("points", [])
    if len(points) < 2:
        raise ValueError("invalid shape: points must contain at least 2 points")

    shape_type = shape.get("shape_type", "polygon")

    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points[:2]
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        w = max(0.0, right - left)
        h = max(0.0, bottom - top)
        bbox = [left, top, w, h]
        segmentation = [[left, top, right, top, right, bottom, left, bottom]]
        area = w * h
        return bbox, segmentation, area

    polygon = [[float(x), float(y)] for x, y in points]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    segmentation = [[coord for xy in polygon for coord in xy]]
    area = _polygon_area(polygon)
    return bbox, segmentation, area


def _ensure_dirs(output_dir: Path) -> None:
    for split in ("train2017", "val2017", "test2017", "annotations"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def _load_labelme_pairs(image_dir: Path, json_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for json_path in sorted(json_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        image_name = data.get("imagePath", "")
        image_path = (json_path.parent / image_name) if image_name else None
        if image_path is None or not image_path.exists():
            stem = json_path.stem
            candidates = list(image_dir.glob(f"{stem}.*"))
            if not candidates:
                print(f"[WARN] image not found for {json_path.name}, skip")
                continue
            image_path = candidates[0]
        pairs.append((image_path, json_path))
    return pairs


def _split_items(items: list, split_ratio: tuple[float, float, float]) -> dict[str, list]:
    train_r, val_r, _ = split_ratio
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return {
        "train2017": items[:n_train],
        "val2017": items[n_train : n_train + n_val],
        "test2017": items[n_train + n_val :],
    }


def _to_coco_json(
    split_items: list[tuple[Path, Path]],
    split_name: str,
    category_map: dict[str, int],
) -> dict:
    images = []
    annotations = []
    ann_id = 1

    for image_id, (image_path, json_path) in enumerate(split_items, start=1):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            print(f"[WARN] invalid image size in {json_path.name}, skip")
            continue

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if not label:
                continue
            if label not in category_map:
                continue
            try:
                bbox, segmentation, area = _shape_to_coco(shape)
            except Exception as exc:
                print(f"[WARN] bad shape in {json_path.name}: {exc}")
                continue

            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_map[label],
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": [float(v) for v in bbox],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": cid, "name": name, "supercategory": "none"} for name, cid in category_map.items()]
    return {
        "info": {"description": f"Labelme converted {split_name}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main() -> None:
    image_dir = Path(labelme_images)
    json_dir = Path(labelme_json)
    output_dir = Path(coco_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"labelme_images not found: {image_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"labelme_json not found: {json_dir}")

    split_ratio = _validate_ratio(ratio)
    _ensure_dirs(output_dir)

    pairs = _load_labelme_pairs(image_dir, json_dir)
    if not pairs:
        raise RuntimeError("no valid labelme json/image pairs found")

    random.seed(SEED)
    random.shuffle(pairs)
    split_map = _split_items(pairs, split_ratio)

    label_counter: defaultdict[str, int] = defaultdict(int)
    for _, json_path in pairs:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label:
                label_counter[label] += 1

    category_names = sorted(label_counter.keys())
    category_map = {name: idx + 1 for idx, name in enumerate(category_names)}

    for split_name, split_items in split_map.items():
        split_dir = output_dir / split_name
        for image_path, _ in split_items:
            shutil.copy2(image_path, split_dir / image_path.name)

        coco_data = _to_coco_json(split_items, split_name, category_map)
        ann_name = f"instances_{split_name}.json"
        ann_path = output_dir / "annotations" / ann_name
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] {split_name}: images={len(coco_data['images'])}, anns={len(coco_data['annotations'])}")

    print(f"[DONE] COCO2017-style dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()