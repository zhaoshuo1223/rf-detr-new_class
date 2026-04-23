import os
import glob

imgs_dir = r'D:\aotto\dingweixiao\aaa_origin_data\JPEGImages'
label_dir = r'D:\aotto\dingweixiao\aaa_origin_data\json'

def delete_unmatched_labels():
    """
    根据图片文件夹中的文件名，删除标签文件夹中不对应的文件
    """
    # 获取图片文件夹中的所有图片文件名（不包含扩展名）
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    img_files = set()
    
    for ext in img_extensions:
        img_paths = glob.glob(os.path.join(imgs_dir, ext))
        for img_path in img_paths:
            # 获取文件名（不包含扩展名）
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_files.add(img_name)
    
    print(f"找到 {len(img_files)} 个图片文件")
    
    # 获取标签文件夹中的所有标签文件
    label_files = glob.glob(os.path.join(label_dir, '*.json'))
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 找出需要删除的标签文件
    files_to_delete = []
    for label_path in label_files:
        # 获取标签文件名（不包含扩展名）
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        
        # 如果标签文件对应的图片不存在，则标记为删除
        if label_name not in img_files:
            files_to_delete.append(label_path)
    
    print(f"需要删除 {len(files_to_delete)} 个不匹配的标签文件")
    
    # 删除不匹配的标签文件
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"已删除: {os.path.basename(file_path)}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {os.path.basename(file_path)}: {e}")
    
    print(f"成功删除 {deleted_count} 个标签文件")
    print(f"剩余 {len(label_files) - deleted_count} 个标签文件")

if __name__ == "__main__":
    delete_unmatched_labels()