import json
import math
import os

import cv2

from function import *

"""
读取 images/json 数据，按原脚本思路汇总定位框并裁剪图片，
在 target_dir 下输出裁剪后的 images 和 json。
每个 json 标注框仍然对应一张裁剪图
裁剪区域会尽量扩成 128x128
以标注框中心为主进行扩展
如果目标靠近边缘，会自动平移裁剪框保证尽量得到完整 128x128
如果原标注框本身某一边已经大于 128，那一边不会强行缩小，避免把目标截断
所以现在输出不再是“刚好等于标注框大小”，而是“包含该标注框的最小 128x128 扩展区域”
"""

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_image_files(directory):
    if not os.path.isdir(directory):
        return []
    image_files = [
        file_name for file_name in os.listdir(directory)
        if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort()
    return image_files


def get_nth_image_path(directory, n):
    image_files = list_image_files(directory)
    if len(image_files) == 0:
        return None

    if n > len(image_files):
        n = len(image_files)
    return os.path.join(directory, image_files[n - 1])


def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def points_to_box(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin = int(math.floor(min(xs)))
    ymin = int(math.floor(min(ys)))
    xmax = int(math.ceil(max(xs)))
    ymax = int(math.ceil(max(ys)))
    return [xmin, ymin, xmax, ymax]


def box_to_points(xmin, ymin, xmax, ymax):
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def find_image_path(image_dir, image_name=None, base_name=None):
    if image_name:
        image_path = os.path.join(image_dir, os.path.basename(image_name))
        if os.path.exists(image_path):
            return image_path

    if base_name:
        for ext in IMAGE_EXTENSIONS:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                return image_path

    return None


def get_fixed_crop_range(center, box_start, box_end, crop_size, limit):
    box_size = box_end - box_start
    target_size = max(crop_size, box_size)

    if limit <= target_size:
        return 0, limit

    start = int(round(center - target_size / 2))
    end = start + target_size

    if start > box_start:
        start = box_start
        end = start + target_size
    if end < box_end:
        end = box_end
        start = end - target_size

    if start < 0:
        start = 0
        end = target_size
    elif end > limit:
        end = limit
        start = limit - target_size

    return start, end


def build_crop_shape(shape, crop_pos):
    sub_box = points_to_box(shape.get("points", []))
    n_x1, n_y1, n_x2, n_y2 = newPos2(parBoxPos=crop_pos, sonBoxPos=sub_box)
    new_shape = dict(shape)
    new_shape["points"] = box_to_points(n_x1, n_y1, n_x2, n_y2)
    new_shape["shape_type"] = "polygon"
    new_shape["flags"] = shape.get("flags", {})
    new_shape["attributes"] = shape.get("attributes", {})
    new_shape["kie_linking"] = shape.get("kie_linking", [])
    new_shape["description"] = shape.get("description", "")
    new_shape["text"] = shape.get("text", "")
    new_shape["difficult"] = shape.get("difficult", False)
    return new_shape


def save_preview_image(image_path, json_data, preview_path):
    image = cv2.imread(image_path)
    if image is None:
        return

    for shape in json_data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xmin, ymin, xmax, ymax = points_to_box(points)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    cv2.imwrite(preview_path, image)


def cut_images_from_json(dataset_dir, dataset_prefix, dist_dir, labels, image_dir_name="images", json_dir_name="json"):
    image_dir = os.path.join(dataset_dir, image_dir_name)
    json_dir = os.path.join(dataset_dir, json_dir_name)
    dist_image_dir = os.path.join(dist_dir, image_dir_name)
    dist_json_dir = os.path.join(dist_dir, json_dir_name)
    os.makedirs(dist_image_dir, exist_ok=True)
    os.makedirs(dist_json_dir, exist_ok=True)

    preview_saved = False
    json_files = sorted([file_name for file_name in os.listdir(json_dir) if file_name.endswith(".json")])
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        json_data = load_json_file(json_path)
        base_name = os.path.splitext(json_file)[0]
        image_path = find_image_path(image_dir, image_name=json_data.get("imagePath"), base_name=base_name)
        if image_path is None:
            continue

        img_file = cv2.imread(image_path)
        if img_file is None:
            continue

        h, w = img_file.shape[:2]
        output_base_name = dataset_prefix + base_name
        if not preview_saved:
            save_preview_image(image_path, json_data, os.path.join(result_dir, output_base_name + ".jpg"))
            preview_saved = True

        obj_iter = 0
        for shape in json_data.get("shapes", []):
            if shape.get("label") not in labels:
                continue

            points = shape.get("points", [])
            if len(points) < 2:
                continue

            obj_iter += 1
            box_x1, box_y1, box_x2, box_y2 = points_to_box(points)
            x_mid = (box_x1 + box_x2) / 2
            y_mid = (box_y1 + box_y2) / 2
            cut_x1, cut_x2 = get_fixed_crop_range(x_mid, box_x1, box_x2, cut_size, w)
            cut_y1, cut_y2 = get_fixed_crop_range(y_mid, box_y1, box_y2, cut_size, h)
            if cut_x1 >= cut_x2 or cut_y1 >= cut_y2:
                continue
            crop_pos = (cut_x1, cut_y1, cut_x2, cut_y2)

            cut_shape = build_crop_shape(shape, crop_pos)
            cut_shapes = [cut_shape]

            cut_img = img_file[cut_y1:cut_y2, cut_x1:cut_x2]
            if cut_img.size == 0:
                continue
            cut_file_stem = "{}_{}".format(output_base_name, obj_iter)
            cut_img_filename = cut_file_stem + ".jpg"
            cut_json_filename = cut_file_stem + ".json"

            cv2.imwrite(os.path.join(dist_image_dir, cut_img_filename), cut_img)

            new_json_data = {
                "version": json_data.get("version", "3.2.3"),
                "flags": json_data.get("flags", {}),
                "shapes": cut_shapes,
                "imagePath": cut_img_filename,
                "imageData": None,
                "imageHeight": cut_y2 - cut_y1,
                "imageWidth": cut_x2 - cut_x1,
                "text": json_data.get("text", ""),
                "description": json_data.get("description", ""),
            }
            with open(os.path.join(dist_json_dir, cut_json_filename), "w", encoding="utf-8") as f:
                json.dump(new_json_data, f, ensure_ascii=False, indent=2)


root_dir = r"D:\aotto\dingweixiao\aaa_origin_data"
result_dir = r"D:\aotto\dingweixiao\aaa_cut-result"
target_dir = r"D:\aotto\dingweixiao\aaa_cut_data"
image_dir_name = "images"
json_dir_name = "json"

CLASS_NAMES = (
    "dingweishao",
    "dingweixiao",
    "dingweixao",
)
cut_size = 128

os.makedirs(result_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
os.makedirs(os.path.join(target_dir, image_dir_name), exist_ok=True)
os.makedirs(os.path.join(target_dir, json_dir_name), exist_ok=True)

dataset_list = os.listdir(root_dir)
dataset_items = []
root_has_data_dirs = (
    os.path.isdir(os.path.join(root_dir, image_dir_name))
    and os.path.isdir(os.path.join(root_dir, json_dir_name))
)
if root_has_data_dirs:
    dataset_items.append(("", root_dir))
else:
    for dirname in dataset_list:
        dirpath = os.path.join(root_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        if (
            os.path.isdir(os.path.join(dirpath, image_dir_name))
            and os.path.isdir(os.path.join(dirpath, json_dir_name))
        ):
            dataset_items.append((dirname, dirpath))

for dirname, dirpath in dataset_items:
    dataset_name = dirname if dirname else os.path.basename(root_dir.rstrip("\\/"))
    dataset_prefix = "" if dirname == "" else dirname + "_"
    print(target_dir)

    cut_images_from_json(
        dataset_dir=dirpath,
        dataset_prefix=dataset_prefix,
        dist_dir=target_dir,
        labels=CLASS_NAMES,
        image_dir_name=image_dir_name,
        json_dir_name=json_dir_name,
    )
