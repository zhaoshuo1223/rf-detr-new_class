import os
import json
import random
import shutil

from labelme import utils
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw


#-------------------------- 配置区 --------------------------

# 1. LabelMe JSON 标注文件所在目录
LABELME_JSON_DIR = r"D:\aotto\dingweixiao\aaa_cut_data\json"

# 2. 对应的图片所在目录
IMAGE_DIR = r"D:\aotto\dingweixiao\aaa_cut_data\images"

# 3. 输出的 COCO 格式数据集根目录
OUTPUT_COCO_PATH = r"D:\aotto\dingweixiao\aaa_coco_data"

# 4. 训练集、验证集和测试集比例 (train_ratio:valid_ratio:test_ratio)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 5. 随机种子（用于可重复的数据分割）
RANDOM_SEED = 42

# 6. 类别定义（如果你有多个类别，可以在这里添加）
CATEGORIES = [
    {
        'id': 0,
        'name': 'dingweixiao',  # 修改为你的物体类别名称
        'supercategory': 'dingweixiao',
    },
    # {
    #     'id': 1,
    #     'name': 'buding',
    #     'supercategory': 'buding',
    # }
]

# 7. 图片文件扩展名（支持的格式）
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json", split_ratio=(0.8, 0.1, 0.1), label_map=None, exclude_labels=None, include_segmentation=False):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json (base path, will add _train, _val, _test)
        :param split_ratio: tuple of (train_ratio, val_ratio, test_ratio)
        :param label_map: dict mapping original labels to new labels, e.g. {"old_label1": "new_label1"}
        :param exclude_labels: list of labels to exclude from conversion
        :param include_segmentation: whether to include segmentation field in COCO annotations (default: True)
                                    If False, only bbox will be included (suitable for detection-only tasks)
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.split_ratio = split_ratio
        self.label_map = label_map if label_map else {}
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()
        self.include_segmentation = include_segmentation
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.image_paths = {}  # 存储图片ID到原始路径的映射

        self.save_json()

    def data_transfer(self):
        # 使用固定的 CATEGORIES 定义（与 json-coco-seg.py 保持一致）
        self.categories = CATEGORIES.copy()
        
        # 创建类别名称到ID的映射
        category_name_to_id = {cat['name']: cat['id'] for cat in CATEGORIES}
        
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                image_info = self.image(data, num, json_file)
                self.images.append(image_info)
                
                # 保存原始图片路径
                image_path = data.get("imagePath", "")
                if not image_path:
                    # 如果imagePath为空，尝试从file_name推断
                    file_name = image_info.get("file_name", "")
                    if file_name:
                        json_dir = os.path.dirname(json_file)
                        image_path = os.path.join(json_dir, file_name)
                
                if not os.path.isabs(image_path):
                    # 如果是相对路径，则相对于JSON文件所在目录
                    json_dir = os.path.dirname(json_file)
                    image_path = os.path.join(json_dir, image_path)
                
                # 如果路径不存在，尝试在JSON文件同目录下查找同名图片文件
                if not os.path.exists(image_path):
                    json_dir = os.path.dirname(json_file)
                    file_name = image_info.get("file_name", "")
                    if file_name:
                        # 尝试直接使用文件名
                        alt_path = os.path.join(json_dir, file_name)
                        if os.path.exists(alt_path):
                            image_path = alt_path
                        else:
                            # 尝试查找同名的其他格式图片
                            base_name = os.path.splitext(file_name)[0]
                            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                                alt_path = os.path.join(json_dir, base_name + ext)
                                if os.path.exists(alt_path):
                                    image_path = alt_path
                                    break
                
                self.image_paths[num] = image_path
                
                for shapes in data["shapes"]:
                    original_label = shapes.get("label", "unknown")
                    
                    # 如果该label在排除列表中，跳过
                    if original_label in self.exclude_labels:
                        continue
                    
                    # 应用label映射
                    label_name = self.label_map.get(original_label, original_label)
                    
                    # 检查类别名称是否在 CATEGORIES 中定义
                    if label_name not in category_name_to_id:
                        print(f"警告: 类别 '{label_name}' 不在 CATEGORIES 定义中，跳过该标注")
                        continue
                    
                    # 直接使用整数 category_id（与 json-coco-seg.py 保持一致）
                    category_id = category_name_to_id[label_name]
                    points = shapes["points"]
                    shape_type = shapes.get("shape_type", "polygon")
                    self.annotations.append(self.annotation(points, category_id, num, shape_type))
                    self.annID += 1

    def image(self, data, num, json_file=None):
        image = {}
        # img = utils.img_b64_to_arr(data["imageData"])
        height = data["imageHeight"]
        width = data["imageWidth"]
        # img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        
        # 获取文件名
        image_path = data.get("imagePath", "")
        if image_path:
            # 处理Windows和Linux路径分隔符
            file_name = os.path.basename(image_path.replace("\\", os.sep).replace("/", os.sep))
        else:
            # 如果imagePath为空，尝试从JSON文件名推断
            if json_file:
                json_base = os.path.splitext(os.path.basename(json_file))[0]
                # 尝试查找同名的图片文件
                json_dir = os.path.dirname(json_file)
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                    potential_file = os.path.join(json_dir, json_base + ext)
                    if os.path.exists(potential_file):
                        file_name = json_base + ext
                        break
                else:
                    file_name = json_base + ".jpg"  # 默认使用jpg
            else:
                file_name = f"image_{num}.jpg"
        
        image["file_name"] = file_name

        self.height = height
        self.width = width

        return image

    def annotation(self, points, category_id, num, shape_type="polygon"):
        """
        创建COCO格式的annotation
        
        Args:
            points: 标注点坐标
            category_id: 类别ID（整数，与 json-coco-seg.py 保持一致）
            num: 图像ID
            shape_type: 形状类型（"rectangle", "polygon", "circle"等）
        """
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # 根据include_segmentation参数决定是否包含segmentation字段
        # 对于矩形标注，如果不需要分割任务，可以不包含segmentation
        if self.include_segmentation:
            annotation["segmentation"] = [list(np.asarray(points).flatten())]
        # 如果include_segmentation=False，则不包含segmentation字段（COCO格式中segmentation是可选的）
        
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = category_id  # 直接使用整数 category_id（与 json-coco-seg.py 保持一致）
        annotation["id"] = self.annID
        return annotation

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def split_data(self):
        """Split data into train, val, test sets"""
        # Get all image IDs
        image_ids = list(range(len(self.images)))
        
        # Shuffle the image IDs
        random.shuffle(image_ids)
        
        # Calculate split indices
        total = len(image_ids)
        train_end = int(total * self.split_ratio[0])
        val_end = train_end + int(total * self.split_ratio[1])
        
        train_ids = set(image_ids[:train_end])
        val_ids = set(image_ids[train_end:val_end])
        test_ids = set(image_ids[val_end:])
        
        # Split images
        train_images = [img for img in self.images if img["id"] in train_ids]
        val_images = [img for img in self.images if img["id"] in val_ids]
        test_images = [img for img in self.images if img["id"] in test_ids]
        
        # Split annotations
        train_annotations = [ann for ann in self.annotations if ann["image_id"] in train_ids]
        val_annotations = [ann for ann in self.annotations if ann["image_id"] in val_ids]
        test_annotations = [ann for ann in self.annotations if ann["image_id"] in test_ids]
        
        return {
            "train": {"images": train_images, "annotations": train_annotations},
            "valid": {"images": val_images, "annotations": val_annotations},
            "test": {"images": test_images, "annotations": test_annotations}
        }

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        
        # Split data
        split_data = self.split_data()
        
        # Create output directory if it doesn't exist
        if os.path.isdir(self.save_json_path):
            output_dir = self.save_json_path
        else:
            output_dir = os.path.dirname(os.path.abspath(self.save_json_path))
            if not output_dir:
                output_dir = os.path.abspath(".")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split
        for split_name, data in split_data.items():
            # Create split directory
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Copy images and update file paths
            updated_images = []
            for img in data["images"]:
                img_id = img["id"]
                original_path = self.image_paths.get(img_id, "")
                
                if original_path and os.path.exists(original_path):
                    # Get filename
                    filename = img["file_name"]
                    # Copy image to split directory
                    dest_path = os.path.join(split_dir, filename)
                    shutil.copy2(original_path, dest_path)
                    
                    # Update file_name in JSON (relative to split directory)
                    updated_img = img.copy()
                    updated_img["file_name"] = filename
                    updated_images.append(updated_img)
                else:
                    print(f"Warning: Image file not found for image_id {img_id}: {original_path}")
                    updated_images.append(img)
            
            data_coco = {
                "images": updated_images,
                "categories": self.categories,
                "annotations": data["annotations"]
            }
            
            # Save JSON file in split directory with COCO standard naming
            json_path = os.path.join(split_dir, f"_annotations.coco.json")
            
            print(f"Saving {split_name} set to: {split_dir}")
            print(f"  Images: {len(updated_images)}")
            print(f"  Annotations: {len(data['annotations'])}")
            print(f"  JSON file: {json_path}")
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_coco, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    total_ratio = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, should sum to 1.0")

    labelme_json = glob.glob(os.path.join(LABELME_JSON_DIR, "*.json"))
    print(f"Found {len(labelme_json)} labelme JSON files")

    split_ratio = (TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    labelme2coco(
        labelme_json,
        OUTPUT_COCO_PATH,
        split_ratio,
        label_map={},
        exclude_labels=None,
        include_segmentation=False,
    )
