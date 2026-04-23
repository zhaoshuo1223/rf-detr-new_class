import os
import json
import numpy as np
from datetime import datetime
import cv2
from pathlib import Path
import shutil
import random

# -------------------------- 配置区 --------------------------

# 1. LabelMe JSON 标注文件所在目录
LABELME_JSON_DIR = r"D:\aotto\quexian-jiance\data-20260331\images"

# 2. 对应的图片所在目录
IMAGE_DIR = r"D:\aotto\quexian-jiance\data-20260331\images"

# 3. 输出的 COCO 格式数据集根目录
OUTPUT_COCO_PATH = r"D:\aotto\quexian-jiance\data-20260331\coco"

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
        'name': 'huahen',
        'supercategory': 'huahen',
    },
    {
        'id': 1,
        'name': 'kailie',  # 修改为你的物体类别名称
        'supercategory': 'kailie',
    },
    {
        'id': 2,
        'name': 'kong',
        'supercategory': 'kong',
    }
]

# 7. 图片文件扩展名（支持的格式）
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# -------------------------- 辅助函数 --------------------------

def create_coco_structure():
    """创建 COCO JSON 的基本结构"""
    return {
        "info": {
            "description": "Converted from LabelMe Format",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

def points_to_flat_list(points):
    """
    将点列表转换为扁平化的坐标列表
    [[x1, y1], [x2, y2], ...] -> [x1, y1, x2, y2, ...]
    """
    return [coord for point in points for coord in point]

def calculate_bbox(points):
    """
    从点列表计算边界框
    返回: [x, y, width, height]
    """
    points_array = np.array(points)
    x_min, y_min = points_array.min(axis=0)
    x_max, y_max = points_array.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

def calculate_area(points):
    """
    使用鞋带公式计算多边形面积
    """
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def find_image_file(json_filename, image_dir):
    """
    根据 JSON 文件名查找对应的图片文件
    """
    base_name = Path(json_filename).stem
    
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path, base_name + ext
    
    return None, None

def get_image_size(image_path):
    """获取图片尺寸"""
    img = cv2.imread(image_path)
    if img is not None:
        height, width = img.shape[:2]
        return width, height
    return None, None

# -------------------------- 主转换函数 --------------------------

def convert_labelme_to_coco(labelme_dir, image_dir, output_coco_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将 LabelMe 格式转换为 COCO 格式，并按比例分割为训练集、验证集和测试集
    
    Args:
        labelme_dir: LabelMe JSON 文件目录
        image_dir: 图片文件目录
        output_coco_path: 输出的 COCO 数据集根目录
        train_ratio: 训练集比例（默认 0.8）
        valid_ratio: 验证集比例（默认 0.1）
        test_ratio: 测试集比例（默认 0.1）
        random_seed: 随机种子（用于可重复的数据分割）
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 验证比例
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"警告: 比例总和为 {total_ratio}，不等于 1.0")
        # 自动调整
        scale = 1.0 / total_ratio
        train_ratio *= scale
        valid_ratio *= scale
        test_ratio *= scale
        print(f"已自动调整为: train={train_ratio:.3f}, valid={valid_ratio:.3f}, test={test_ratio:.3f}")
    
    # 创建输出目录结构
    train_dir = os.path.join(output_coco_path, 'train')
    valid_dir = os.path.join(output_coco_path, 'valid')
    test_dir = os.path.join(output_coco_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建训练集、验证集和测试集的 COCO 结构
    coco_train = create_coco_structure()
    coco_valid = create_coco_structure()
    coco_test = create_coco_structure()
    
    # 获取所有 JSON 文件
    json_files = sorted([f for f in os.listdir(labelme_dir) if f.endswith('.json')])
    
    if len(json_files) == 0:
        print(f"错误: 在 {labelme_dir} 中没有找到 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 随机打乱文件列表
    json_files_shuffled = json_files.copy()
    random.shuffle(json_files_shuffled)
    
    # 计算分割点
    total_files = len(json_files_shuffled)
    train_count = int(total_files * train_ratio)
    valid_count = int(total_files * valid_ratio)
    test_count = total_files - train_count - valid_count
    
    print(f"\n数据分割:")
    print(f"  训练集: {train_count} 张 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {valid_count} 张 ({valid_ratio*100:.1f}%)")
    print(f"  测试集: {test_count} 张 ({test_ratio*100:.1f}%)")
    print("-" * 60)
    
    # 分割文件列表
    train_files = json_files_shuffled[:train_count]
    valid_files = json_files_shuffled[train_count:train_count + valid_count]
    test_files = json_files_shuffled[train_count + valid_count:]
    
    # 处理训练集
    print("\n处理训练集...")
    train_image_id = 1
    train_annotation_id = 1
    
    for json_file in train_files:
        result = process_single_file(
            json_file, labelme_dir, image_dir, train_dir,
            train_image_id, train_annotation_id, coco_train
        )
        if result:
            train_image_id, train_annotation_id = result
    
    # 处理验证集
    print("\n处理验证集...")
    valid_image_id = 1
    valid_annotation_id = 1
    
    for json_file in valid_files:
        result = process_single_file(
            json_file, labelme_dir, image_dir, valid_dir,
            valid_image_id, valid_annotation_id, coco_valid
        )
        if result:
            valid_image_id, valid_annotation_id = result
    
    # 处理测试集
    print("\n处理测试集...")
    test_image_id = 1
    test_annotation_id = 1
    
    for json_file in test_files:
        result = process_single_file(
            json_file, labelme_dir, image_dir, test_dir,
            test_image_id, test_annotation_id, coco_test
        )
        if result:
            test_image_id, test_annotation_id = result
    
    # 保存训练集 JSON
    train_json_path = os.path.join(train_dir, '_annotations.coco.json')
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_train, f, indent=2, ensure_ascii=False)
    
    # 保存验证集 JSON
    valid_json_path = os.path.join(valid_dir, '_annotations.coco.json')
    with open(valid_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_valid, f, indent=2, ensure_ascii=False)
    
    # 保存测试集 JSON
    test_json_path = os.path.join(test_dir, '_annotations.coco.json')
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_test, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print(f"📁 输出目录: {output_coco_path}")
    print(f"\n📊 训练集统计:")
    print(f"   - 图片数量: {len(coco_train['images'])}")
    print(f"   - 标注数量: {len(coco_train['annotations'])}")
    print(f"   - JSON 文件: {train_json_path}")
    print(f"\n📊 验证集统计:")
    print(f"   - 图片数量: {len(coco_valid['images'])}")
    print(f"   - 标注数量: {len(coco_valid['annotations'])}")
    print(f"   - JSON 文件: {valid_json_path}")
    print(f"\n📊 测试集统计:")
    print(f"   - 图片数量: {len(coco_test['images'])}")
    print(f"   - 标注数量: {len(coco_test['annotations'])}")
    print(f"   - JSON 文件: {test_json_path}")
    print("=" * 60)


def process_single_file(json_file, labelme_dir, image_dir, output_dir, 
                       image_id_counter, annotation_id_counter, coco_output):
    """
    处理单个文件：读取 LabelMe JSON，转换为 COCO 格式，复制图片
    根据 CATEGORIES 中定义的类别，为每个类别创建独立的标注
    
    Returns:
        (new_image_id, new_annotation_id) 如果成功，否则 None
    """
    json_path = os.path.join(labelme_dir, json_file)
    
    try:
        # 读取 LabelMe JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 查找对应的图片文件
        image_path, image_filename = find_image_file(json_file, image_dir)
        
        if image_path is None:
            print(f"  警告: 找不到 {json_file} 对应的图片，跳过")
            return None
        
        # 获取图片尺寸
        width, height = get_image_size(image_path)
        if width is None:
            print(f"  警告: 无法读取图片 {image_filename}，跳过")
            return None
        
        # 复制图片到输出目录
        output_image_path = os.path.join(output_dir, image_filename)
        shutil.copy2(image_path, output_image_path)
        
        # 添加图片信息
        image_info = {
            "id": image_id_counter,
            "file_name": image_filename,
            "width": width,
            "height": height,
        }
        coco_output["images"].append(image_info)
        
        # 处理标注
        shapes = labelme_data.get('shapes', [])
        
        # 创建类别名称到ID的映射
        category_name_to_id = {cat['name']: cat['id'] for cat in CATEGORIES}
        
        # 获取所有定义的类别名称
        category_names = [cat['name'] for cat in CATEGORIES]
        
        # 按照类别名称动态分组 shapes
        shapes_by_category = {}
        for category_name in category_names:
            shapes_by_category[category_name] = [s for s in shapes if s['label'] == category_name]
        
        current_annotation_id = annotation_id_counter
        
        # 为每个类别创建独立的标注
        annotation_counts = {}
        for category_name in category_names:
            category_shapes = shapes_by_category[category_name]
            annotation_counts[category_name] = len(category_shapes)
            
            for shape in category_shapes:
                process_single_shape(
                    shape, image_id_counter, current_annotation_id,
                    coco_output, category_name_to_id[category_name]
                )
                current_annotation_id += 1
        
        # 生成统计信息字符串
        stats_parts = [f"{name}: {count}" for name, count in annotation_counts.items()]
        stats_str = ", ".join(stats_parts)
        total_annotations = sum(annotation_counts.values())
        
        print(f"  ✓ {json_file} -> {image_filename} ({stats_str}, 总计: {total_annotations} 个标注)")
        return (image_id_counter + 1, current_annotation_id)
        
    except Exception as e:
        print(f"  错误: 处理 {json_file} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_single_shape(shape, image_id, annotation_id, coco_output, category_id):
    """
    处理单个形状，创建独立的标注
    
    Args:
        shape: LabelMe 格式的形状对象
        image_id: 图片ID
        annotation_id: 标注ID
        coco_output: COCO输出字典
        category_id: 类别ID
    """
    points = shape['points']
    
    # 构建 segmentation（单个多边形）
    segmentation = [points_to_flat_list(points)]
    
    # 计算边界框
    bbox = calculate_bbox(points)
    
    # 计算面积
    area = calculate_area(points)
    area = max(0, area)  # 确保面积非负
    
    # 创建 annotation
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": float(area),
        "bbox": bbox,
        "iscrowd": 0,
    }
    
    coco_output["annotations"].append(annotation)

# -------------------------- 主程序 --------------------------

def main():
    if not os.path.exists(LABELME_JSON_DIR):
        print(f"错误: 找不到 JSON 目录 {LABELME_JSON_DIR}")
        return
    
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 找不到图片目录 {IMAGE_DIR}")
        return
    
    # 验证比例
    total_ratio = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"警告: 比例总和为 {total_ratio}，不等于 1.0")
        print(f"将自动调整比例")
    
    print("=" * 60)
    print("LabelMe 转 COCO 格式数据集")
    print("=" * 60)
    print(f"输入:")
    print(f"  LabelMe JSON 目录: {LABELME_JSON_DIR}")
    print(f"  图片目录: {IMAGE_DIR}")
    print(f"输出:")
    print(f"  COCO 数据集目录: {OUTPUT_COCO_PATH}")
    print(f"分割比例: 训练集 {TRAIN_RATIO*100:.1f}% : 验证集 {VALID_RATIO*100:.1f}% : 测试集 {TEST_RATIO*100:.1f}%")
    print("=" * 60)
    
    convert_labelme_to_coco(
        LABELME_JSON_DIR, 
        IMAGE_DIR, 
        OUTPUT_COCO_PATH,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )

if __name__ == '__main__':
    main()
