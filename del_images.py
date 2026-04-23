import os
import glob

imgs_dir = r'D:\aotto\dingweixiao\aaa_origin_data\JPEGImages'
label_dir = r'D:\aotto\dingweixiao\aaa_origin_data\json'
def delete_unmatched_images():
    """
    根据标签文件夹中的文件名，删除图片文件夹中不对应的文件
    """
    # 获取标签文件夹中的所有标签文件名（不包含扩展名）
    label_files = glob.glob(os.path.join(label_dir, '*.json'))
    label_names = set()
    
    for label_path in label_files:
        # 获取文件名（不包含扩展名）
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        label_names.add(label_name)
    
    print(f"找到 {len(label_names)} 个标签文件")
    
    # 获取图片文件夹中的所有图片文件
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    img_files = []
    
    for ext in img_extensions:
        img_paths = glob.glob(os.path.join(imgs_dir, ext))
        img_files.extend(img_paths)
    
    print(f"找到 {len(img_files)} 个图片文件")
    
    # 找出需要删除的图片文件
    files_to_delete = []
    for img_path in img_files:
        # 获取图片文件名（不包含扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 如果图片文件对应的标签不存在，则标记为删除
        if img_name not in label_names:
            files_to_delete.append(img_path)
    
    print(f"需要删除 {len(files_to_delete)} 个不匹配的图片文件")
    
    # 删除不匹配的图片文件
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"已删除: {os.path.basename(file_path)}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {os.path.basename(file_path)}: {e}")
    
    print(f"成功删除 {deleted_count} 个图片文件")
    print(f"剩余 {len(img_files) - deleted_count} 个图片文件")

if __name__ == "__main__":
    delete_unmatched_images()
