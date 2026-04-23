import json
import shutil
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw

# 输入目录：Labelme json 所在目录
LABELME_JSON_DIR = r"D:\aotto\dingweixiao\aaa_cut_data\json"
# 可选：原始图片目录（当 imagePath 只是文件名或与 json 不同目录时使用）
IMAGE_DIR = r"D:\aotto\dingweixiao\aaa_cut_data\images"

# 输出文件：坏样本清单
BAD_SAMPLES_REPORT = r"D:\aotto\dingweixiao\aaa_cut_data\bad_samples_report.txt"
# 坏样本剪切目录（会创建 bad_json/ 与 bad_images/）
BAD_SAMPLES_DIR = r"D:\aotto\dingweixiao\aaa_cut_data\bad_samples"
# 是否执行剪切（True=移动文件；False=只生成报告）
MOVE_BAD_SAMPLES = True


def polygons_to_mask(img_shape: tuple[int, int], polygons: list[list[float]]) -> np.ndarray:
    mask = np.zeros(img_shape, dtype=np.uint8)
    pil_mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(pil_mask).polygon(xy=xy, outline=1, fill=1)
    return np.array(pil_mask, dtype=bool)


def validate_shape(shape: dict, image_h: int, image_w: int) -> list[str]:
    errors: list[str] = []
    points = shape.get("points")
    label = shape.get("label", "<unknown>")
    shape_type = shape.get("shape_type", "polygon")

    if not isinstance(points, list) or len(points) < 3:
        errors.append(f"label={label}: points不足3个或格式非法")
        return errors

    try:
        contour = np.array(points, dtype=np.float64)
    except Exception:
        errors.append(f"label={label}: points无法转换为数值")
        return errors

    if contour.ndim != 2 or contour.shape[1] != 2:
        errors.append(f"label={label}: points维度非法，期望(N,2)，实际{contour.shape}")
        return errors

    if np.any(np.isnan(contour)) or np.any(np.isinf(contour)):
        errors.append(f"label={label}: points包含NaN或Inf")

    # 面积接近0通常是退化多边形
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area <= 0:
        errors.append(f"label={label}: 多边形面积为0（退化）")

    # 按你的转换逻辑检查是否会产生空mask
    mask = polygons_to_mask((image_h, image_w), contour.tolist())
    index = np.argwhere(mask == 1)
    if index.size == 0:
        errors.append(
            f"label={label}: polygon栅格化后mask为空（可能点越界/自交/极小目标），shape_type={shape_type}"
        )

    return errors


def main() -> None:
    json_dir = Path(LABELME_JSON_DIR)
    image_dir = Path(IMAGE_DIR) if IMAGE_DIR else None
    report_path = Path(BAD_SAMPLES_REPORT)
    bad_root = Path(BAD_SAMPLES_DIR)
    bad_json_dir = bad_root / "bad_json"
    bad_img_dir = bad_root / "bad_images"

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON目录不存在: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    total = len(json_files)
    bad_entries: list[str] = []
    bad_file_count = 0
    moved_json_count = 0
    moved_img_count = 0

    for i, jf in enumerate(json_files, start=1):
        file_errors: list[str] = []
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            bad_entries.append(f"{jf}\n  - JSON解析失败: {exc}\n")
            bad_file_count += 1
            continue

        image_h = data.get("imageHeight")
        image_w = data.get("imageWidth")
        if not isinstance(image_h, int) or not isinstance(image_w, int) or image_h <= 0 or image_w <= 0:
            file_errors.append(f"图像尺寸非法: imageHeight={image_h}, imageWidth={image_w}")

        shapes = data.get("shapes")
        if not isinstance(shapes, list) or len(shapes) == 0:
            file_errors.append("shapes为空或格式非法")
            shapes = []

        if isinstance(image_h, int) and isinstance(image_w, int) and image_h > 0 and image_w > 0:
            for idx, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    file_errors.append(f"shape[{idx}] 不是dict")
                    continue
                shape_errors = validate_shape(shape, image_h, image_w)
                for err in shape_errors:
                    file_errors.append(f"shape[{idx}] {err}")

        if file_errors:
            bad_file_count += 1
            joined = "\n  - ".join(file_errors)
            bad_entries.append(f"{jf}\n  - {joined}\n")

            if MOVE_BAD_SAMPLES:
                bad_json_dir.mkdir(parents=True, exist_ok=True)
                bad_img_dir.mkdir(parents=True, exist_ok=True)

                # 1) 移动坏 json
                dst_json = bad_json_dir / jf.name
                if dst_json.exists():
                    dst_json = bad_json_dir / f"{jf.stem}__dup{jf.suffix}"
                shutil.move(str(jf), str(dst_json))
                moved_json_count += 1

                # 2) 尝试移动对应图片（优先用 imagePath，其次同名多后缀）
                image_path_val = ""
                if isinstance(data, dict):
                    image_path_val = str(data.get("imagePath", "") or "")

                candidate_img: Path | None = None
                if image_path_val:
                    p = Path(image_path_val)
                    if p.is_absolute() and p.exists():
                        candidate_img = p
                    else:
                        # 相对路径按原 json 所在目录解析
                        p_rel = (jf.parent / p).resolve()
                        if p_rel.exists():
                            candidate_img = p_rel

                if candidate_img is None:
                    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"):
                        p2 = jf.with_suffix(ext)
                        if p2.exists():
                            candidate_img = p2
                            break

                # 3) 如果仍未找到，尝试在 IMAGE_DIR 里按 imagePath 文件名 / json同名文件查找
                if candidate_img is None and image_dir is not None and image_dir.exists():
                    fallback_names: list[str] = []
                    if image_path_val:
                        fallback_names.append(Path(image_path_val).name)
                    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"):
                        fallback_names.append(jf.stem + ext)

                    for name in fallback_names:
                        p3 = image_dir / name
                        if p3.exists():
                            candidate_img = p3
                            break

                if candidate_img is not None and candidate_img.exists():
                    dst_img = bad_img_dir / candidate_img.name
                    if dst_img.exists():
                        dst_img = bad_img_dir / f"{candidate_img.stem}__dup{candidate_img.suffix}"
                    shutil.move(str(candidate_img), str(dst_img))
                    moved_img_count += 1

        if i % 2000 == 0:
            print(f"已检查 {i}/{total} ...")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"总JSON数: {total}\n")
        f.write(f"坏样本文件数: {bad_file_count}\n")
        f.write(f"已移动坏JSON数: {moved_json_count}\n")
        f.write(f"已移动对应图片数: {moved_img_count}\n")
        f.write(f"MOVE_BAD_SAMPLES: {MOVE_BAD_SAMPLES}\n")
        f.write(f"IMAGE_DIR: {IMAGE_DIR}\n")
        f.write("=" * 80 + "\n\n")
        for entry in bad_entries:
            f.write(entry)
            f.write("\n")

    print(f"检查完成: 总数={total}, 坏样本文件数={bad_file_count}")
    if MOVE_BAD_SAMPLES:
        print(f"已剪切坏样本: bad_json={moved_json_count}, bad_images={moved_img_count} -> {bad_root}")
    print(f"坏样本报告已保存: {report_path}")


if __name__ == "__main__":
    main()
