import json
import random
import shutil
import glob
import xml.etree.ElementTree as ET
import cv2
import os
import math
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
创建txt文件
"""
def createTxt(dist_dir, ratio = 0.8, train_name = "trainval.txt", test_name = "test.txt"):
    dist_anno, dist_image, path_txt = getSubDir(dist_dir)
    path_train_txt = os.path.join(path_txt, train_name)
    path_test_txt = os.path.join(path_txt, test_name)

    img_list = []
    for img1 in os.listdir(dist_image):
        if img1.endswith(".jpg"):
            # 检查xml和jpg是否匹配
            image_id, _ = os.path.splitext(img1)
            anno_file_path = os.path.join(dist_anno, image_id + ".xml")
            image_file_path = os.path.join(dist_image, img1)
            if os.path.exists(anno_file_path) and os.path.exists(image_file_path):
                img_list.append(img1)
            else:
                os.remove(anno_file_path)
                os.remove(image_file_path)

    # 乱序
    len(img_list)
    random.shuffle(img_list)

    # 按比例分割
    total = len(img_list)
    offset = int(total * ratio)
    train_img_list = img_list[:offset]
    test_img_list = img_list[offset:]

    # write train
    train_txt = open(path_train_txt, "w")
    for img_file in tqdm(train_img_list):
        image_id, _ = os.path.splitext(img_file)

        line = str(image_id) + "\n"
        train_txt.write(line)
    test_txt = open(path_test_txt, "w")
    for img_file in tqdm(test_img_list):
        image_id, _ = os.path.splitext(img_file)

        line = str(image_id) + "\n"
        test_txt.write(line)
    train_txt.close()
    test_txt.close()


"""
拷贝目录
遍历srcDir中的文件，拷贝至distDir
"""
def copyDir(srcDir, distDir, preName = ""):
    src_file_list = os.listdir(srcDir)

    for src_file in tqdm(src_file_list):
        src_file_path = os.path.join(srcDir, src_file)
        dist_file_path = os.path.join(distDir, "{}-{}".format(preName, src_file.replace(' ', '_'))) # 将文件名中空格替换为下划线
        shutil.copy(src_file_path, dist_file_path)

"""
返回子目录名称
"""
def getSubDir(dir, anno = "Annotations", jpeg = "JPEGImages"):
    anno_dir = os.path.join(dir, anno)
    image_dir = os.path.join(dir, jpeg)
    txt_dir = os.path.join(dir, "ImageSets/Main")
    if not os.path.isdir(anno_dir):
        os.makedirs(anno_dir)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)
    return anno_dir, image_dir, txt_dir


"""
格式化目录
1：最后有'/'的话去掉
2：若不存在则创建
"""
def formatDir(dir):
    return 1

"""
判断xml和jpg是否都存在
"""
def checkExist(dir, filename, delete=False):
    xml_dir, jpg_dir, txt_dir = getSubDir(dir)
    fileid = os.path.basename(filename)
    fileid = fileid[:-4]
    xml_path = os.path.join(xml_dir, fileid + ".xml")
    jpg_path = os.path.join(jpg_dir, fileid + ".jpg")

    if os.path.exists(xml_path) and os.path.exists(jpg_path):
        return True
    else:
        if delete:
            os.remove(xml_path)
            os.remove(jpg_path)
        return False

"""
重命名xml和jpg
"""
def rename():
    return 1

"""
格式化xml文件缩进
"""
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

"""
判断子框是否在父框内
"""
def isInBox( parBoxPos, sonBoxPos ):
    x1, y1, x2, y2 = parBoxPos
    s_x1, s_y1, s_x2, s_y2 = sonBoxPos
    # 子框的左上角在父框中，右下角也在父框中
    # if s_x1 >= x1 and s_x1 < x2 and s_y1 >= y1 and s_y1 < y2:
    if s_x1 >= x1 and s_y1 >= y1 and s_x2 <= x2 and s_y2 <= y2:
        return True
    else:
        return False

"""
判断子框中心点是否在父框内
"""
def isInBox2( parBoxPos, sonBoxPos ):
    x1, y1, x2, y2 = parBoxPos
    s_x1, s_y1, s_x2, s_y2 = sonBoxPos
    # 子框的中心点在父框中
    mid_sx = (s_x1 + s_x2)/2
    mid_sy = (s_y1 + s_y2)/2
    if mid_sx >= x1 and mid_sy >= y1 and mid_sx <= x2 and mid_sy <= y2:
        return True
    else:
        return False


"""
创建一个空的xml
"""
def getNewXml( img_filename, dirPath, width, height):
    width = str(width)
    height = str(height)

    root = ET.Element("annotation")
    root.attrib = {"verified.":"no"}

    tag_folder = ET.SubElement(root, "folder")
    tag_folder.text = "JPEGImages"

    tag_filename = ET.SubElement(root, "filename")
    tag_filename.text = img_filename

    tag_path = ET.SubElement(root, "path")
    tag_path.text = dirPath + img_filename

    tag_source = ET.SubElement(root, "source")
    tag_database = ET.SubElement(tag_source, "database")
    tag_database.text = "Unknown"

    tag_size = ET.SubElement(root, "size")
    tag_width = ET.SubElement(tag_size, "width")
    tag_width.text = width
    tag_height = ET.SubElement(tag_size, "height")
    tag_height.text = height
    tag_depth = ET.SubElement(tag_size, "depth")
    tag_depth.text = '1'

    tag_segmented = ET.SubElement(root, "segmented")
    tag_segmented.text = '0'

    # ET.dump(root)
    return root

"""
计算子框在裁剪后的坐标
"""
def newPos( parBoxPos, sonBoxPos ):
    x1, y1, x2, y2 = parBoxPos
    s_x1, s_y1, s_x2, s_y2 = sonBoxPos

    new_x1 = s_x1 - x1
    new_x2 = s_x2 - x1
    new_y1 = s_y1 - y1
    new_y2 = s_y2 - y1

    return (new_x1, new_y1, new_x2, new_y2)

"""
计算子框在裁剪后的坐标
"""
def newPos2( parBoxPos, sonBoxPos ):
    x1, y1, x2, y2 = parBoxPos
    s_x1, s_y1, s_x2, s_y2 = sonBoxPos

    if s_x1 < x1:
        s_x1 = x1
    if s_y1 < y1:
        s_y1 = y1
    if s_x2 > x2:
        s_x2 = x2
    if s_y2 > y2:
        s_y2 = y2

    new_x1 = s_x1 - x1
    new_x2 = s_x2 - x1
    new_y1 = s_y1 - y1
    new_y2 = s_y2 - y1

    return (new_x1, new_y1, new_x2, new_y2)

"""
在 xml 中增加一个object节点
"""
def newObjElement(root, x1, y1, x2, y2, label, line = None):
    x1 = str(x1)
    y1 = str(y1)
    x2 = str(x2)
    y2 = str(y2)

    tag_object = ET.SubElement(root, "object")
    tag_name = ET.SubElement(tag_object, "name")
    tag_name.text = label
    tag_pose = ET.SubElement(tag_object, "pose")
    tag_pose.text = "Unspecified"
    if line is not None:
        tag_line = ET.SubElement(tag_object, "line")
        tag_line.text = line
    tag_truncated = ET.SubElement(tag_object, "truncated")
    tag_truncated.text = '0'
    tag_difficult = ET.SubElement(tag_object, "difficult")
    tag_difficult.text = '0'

    tag_bndbox = ET.SubElement(tag_object, "bndbox")
    tag_xmin = ET.SubElement(tag_bndbox, "xmin")
    tag_xmin.text = x1
    tag_ymin = ET.SubElement(tag_bndbox, "ymin")
    tag_ymin.text = y1
    tag_xmax = ET.SubElement(tag_bndbox, "xmax")
    tag_xmax.text = x2
    tag_ymax = ET.SubElement(tag_bndbox, "ymax")
    tag_ymax.text = y2

    # tree = ET.ElementTree(root)
    return root

def cut_image(src_dir, dist_dir, labels, border ):
    dist_xml_dir, dist_image_dir, dist_txt_dir = getSubDir(dist_dir)
    src_xml_dir, src_image_dir, src_txt_dir = getSubDir(src_dir)
    # 遍历xml文件
    jpg_files = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for jpg_file in tqdm(jpg_files):

        filepath, img_filename = os.path.split(jpg_file)
        filename = img_filename[:-4]

        xml_filename = filename + ".xml"

        src_xml_file = os.path.join(src_xml_dir, xml_filename)
        if not os.path.exists(src_xml_file):
            continue

        xml_tree = ET.parse(src_xml_file)
        xml_root = xml_tree.getroot()

        # 读原图片
        img_file = cv2.imread(jpg_file)
        size = img_file.shape
        h = size[0]
        w = size[1]
        # 遍历object
        obj_iter = 0
        for member in xml_root.findall("object"):
            # 如果遇到parent_box_label
            obj_iter = obj_iter + 1
            name = member.find("name").text

            # 判断子框在父框内的方法为子框左上角在父框内，所以令父框向左上角扩大10px
            x1, x2, y1, y2 = expandBorder(x1=member.find("bndbox")[0].text,
                                          y1=member.find("bndbox")[1].text,
                                          x2=member.find("bndbox")[2].text,
                                          y2=member.find("bndbox")[3].text,
                                          border=border,
                                          width=w,
                                          height=h)
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            # 清晰度
            clear = None
            if member.find("clear") is not None:
                clear = member.find("clear").text
                # if int(clear) < 6: # 清晰度低于6的不处理
                #     continue

            pos = (x1, y1, x2, y2)
            if name in labels:
                # 生成一个新的xml文件
                sub_filename = filename.replace(' ', '_') + "_" + name

                # 清晰度
                # if clear is not None:
                #     sub_filename = sub_filename + "_clear_" + clear

                # print( "get sub file name: {}".format(sub_filename))
                newXmlRoot = getNewXml(img_filename = sub_filename + ".jpg",
                                   dirPath=dist_image_dir,
                                   width=x2-x1,
                                   height=y2-y1)

                if clear is not None:
                    tag_path = ET.SubElement(newXmlRoot, "clear")
                    tag_path.text = clear

                # 开始根据pos遍历子label
                for sub_member in xml_root.findall("object"):

                    sub_name = sub_member.find("name").text
                    s_x1 = math.floor(float(sub_member.find("bndbox")[0].text))
                    s_y1 = math.floor(float(sub_member.find("bndbox")[1].text))
                    s_x2 = math.floor(float(sub_member.find("bndbox")[2].text))
                    s_y2 = math.floor(float(sub_member.find("bndbox")[3].text))
                    line = None
                    if sub_member.find("line") is not None:
                        line = sub_member.find("line").text
                    sub_pos = (s_x1, s_y1, s_x2, s_y2)
                    # 如果是子label，并且在parent box内，那么计算新的pos信息，将pos信息追加到xml文件中
                    if sub_name not in labels and isInBox(parBoxPos=pos, sonBoxPos=sub_pos):
                        n_x1, n_y1, n_x2, n_y2 = newPos(parBoxPos=pos, sonBoxPos=sub_pos)
                        newXmlRoot = newObjElement(newXmlRoot, n_x1, n_y1, n_x2, n_y2, sub_name, line)
                        # ET.dump(newXmlRoot)

                # 最后导出xml文件，并裁剪图片
                indent(newXmlRoot)
                tree = ET.ElementTree(newXmlRoot)
                # tree.write("{}{}_{}_clear_{}.xml".format(dist_xml_dir, sub_filename, obj_iter,clear), encoding="utf-8")
                cut_xml_filename = "{}_{}_clear_{}.xml".format(sub_filename, obj_iter,clear)
                tree.write(os.path.join(dist_xml_dir, cut_xml_filename), encoding="utf-8")
                # 裁剪图片
                img_cut = img_file[y1:y2, x1:x2]
                # 导出图片
                cut_img_filename = "{}_{}_clear_{}.jpg".format(sub_filename, obj_iter,clear)
                cv2.imwrite(os.path.join(dist_image_dir, cut_img_filename), img_cut)
                # cv2.imwrite("{}{}_{}_clear_{}.jpg".format(dist_image_dir, sub_filename, obj_iter,clear), img_cut)


def cut_dwx(src_dir, dist_dir, labels, cut_size, pos):
    dist_xml_dir, dist_image_dir, dist_txt_dir = getSubDir(dist_dir)
    src_xml_dir, src_image_dir, src_txt_dir = getSubDir(src_dir)
    # 遍历xml文件
    jpg_files = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for jpg_file in tqdm(jpg_files):

        filepath, img_filename = os.path.split(jpg_file)
        filename = img_filename[:-4]

        xml_filename = filename + ".xml"

        src_xml_file = os.path.join(src_xml_dir, xml_filename)
        if not os.path.exists(src_xml_file):
            continue

        xml_tree = ET.parse(src_xml_file)
        xml_root = xml_tree.getroot()

        # 读原图片
        img_file = cv2.imread(jpg_file)
        size = img_file.shape
        h = size[0]
        w = size[1]
        # 遍历object
        obj_iter = 0

        # 遍历定位销框
        pos_dict = json.loads(pos)
        pos_dict["coord"] = [x for x in pos_dict["coord"] if x["type"] == "rect"]
        for coord in pos_dict["coord"]:

        # for member in xml_root.findall("object"):
            # 如果遇到 labels，裁剪图片和坐标框
            obj_iter = obj_iter + 1
            # name = member.find("name").text
            # if name not in labels:
            #     continue

            # 找member的中心点
            # x1=member.find("bndbox")[0].text
            # y1=member.find("bndbox")[1].text
            # x2=member.find("bndbox")[2].text
            # y2=member.find("bndbox")[3].text
            x1 = coord["coords"]["x1"]
            y1 = coord["coords"]["y1"]
            x2 = coord["coords"]["x2"]
            y2 = coord["coords"]["y2"]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            x_mid = ( x1 + x2 ) / 2
            y_mid = ( y1 + y2 ) / 2

            cut_x1 = int(x_mid - cut_size / 2)
            if cut_x1 < 0:
                cut_x1 = 0
            cut_x2 = int(x_mid + cut_size / 2)
            if cut_x2 > w:
                cut_x2 = w
            cut_y1 = int(y_mid - cut_size / 2)
            if cut_y1 < 0:
                cut_y1 = 0
            cut_y2 = int(y_mid + cut_size / 2)
            if cut_y2 > w:
                cut_y2 = w

            # 要裁剪的坐标区域
            cut_pos = (cut_x1, cut_y1, cut_x2, cut_y2)

            # if name in labels:
            if True:
                # 生成一个新的xml文件
                sub_filename = filename.replace(' ', '_')

                newXmlRoot = getNewXml(img_filename = sub_filename + ".jpg",
                                   dirPath=dist_image_dir,
                                   width=cut_x2-cut_x1,
                                   height=cut_y2-cut_y1)


                # 开始根据pos遍历子label
                for sub_member in xml_root.findall("object"):

                    sub_name = sub_member.find("name").text
                    s_x1 = math.floor(float(sub_member.find("bndbox")[0].text))
                    s_y1 = math.floor(float(sub_member.find("bndbox")[1].text))
                    s_x2 = math.floor(float(sub_member.find("bndbox")[2].text))
                    s_y2 = math.floor(float(sub_member.find("bndbox")[3].text))
                    line = None
                    # if sub_member.find("line") is not None:
                    #     line = sub_member.find("line").text
                    sub_pos = (s_x1, s_y1, s_x2, s_y2)
                    # 如果是子label，并且在parent box内，那么计算新的pos信息，将pos信息追加到xml文件中
                    if sub_name in labels and isInBox(parBoxPos=cut_pos, sonBoxPos=sub_pos):
                        n_x1, n_y1, n_x2, n_y2 = newPos(parBoxPos=cut_pos, sonBoxPos=sub_pos)
                        newXmlRoot = newObjElement(newXmlRoot, n_x1, n_y1, n_x2, n_y2, sub_name, line)
                        # ET.dump(newXmlRoot)

                # 最后导出xml文件，并裁剪图片
                indent(newXmlRoot)
                tree = ET.ElementTree(newXmlRoot)
                # tree.write("{}{}_{}_clear_{}.xml".format(dist_xml_dir, sub_filename, obj_iter,clear), encoding="utf-8")
                cut_xml_filename = "{}_{}.xml".format(sub_filename, obj_iter)
                tree.write(os.path.join(dist_xml_dir, cut_xml_filename), encoding="utf-8")
                # 裁剪图片
                img_cut = img_file[cut_y1:cut_y2, cut_x1:cut_x2]
                # 导出图片
                cut_img_filename = "{}_{}.jpg".format(sub_filename, obj_iter)
                cv2.imwrite(os.path.join(dist_image_dir, cut_img_filename), img_cut)
                # cv2.imwrite("{}{}_{}_clear_{}.jpg".format(dist_image_dir, sub_filename, obj_iter,clear), img_cut)

def cut_dwx2(src_dir, dist_dir, labels, cut_size, boxs):
    dist_xml_dir, dist_image_dir, dist_txt_dir = getSubDir(dist_dir)
    src_xml_dir, src_image_dir, src_txt_dir = getSubDir(src_dir)
    # 遍历xml文件
    jpg_files = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for jpg_file in tqdm(jpg_files):

        filepath, img_filename = os.path.split(jpg_file)
        filename = img_filename[:-4]

        xml_filename = filename + ".xml"

        src_xml_file = os.path.join(src_xml_dir, xml_filename)
        if not os.path.exists(src_xml_file):
            continue

        xml_tree = ET.parse(src_xml_file)
        xml_root = xml_tree.getroot()

        # 读原图片
        img_file = cv2.imread(jpg_file)
        size = img_file.shape
        h = size[0]
        w = size[1]
        # 遍历object
        obj_iter = 0

        # 遍历定位销框
        # pos_dict = json.loads(pos)
        # pos_dict["coord"] = [x for x in pos_dict["coord"] if x["type"] == "rect"]
        for coord in boxs:

        # for member in xml_root.findall("object"):
            # 如果遇到 labels，裁剪图片和坐标框
            obj_iter = obj_iter + 1
            # name = member.find("name").text
            # if name not in labels:
            #     continue

            # 找member的中心点
            # x1=member.find("bndbox")[0].text
            # y1=member.find("bndbox")[1].text
            # x2=member.find("bndbox")[2].text
            # y2=member.find("bndbox")[3].text
            x1 = coord[0]
            y1 = coord[1]
            x2 = coord[2]
            y2 = coord[3]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            x_mid = ( x1 + x2 ) / 2
            y_mid = ( y1 + y2 ) / 2

            cut_x1 = int(x_mid - cut_size / 2)
            if cut_x1 < 0:
                cut_x1 = 0
            cut_x2 = int(x_mid + cut_size / 2)
            if cut_x2 > w:
                cut_x2 = w
            cut_y1 = int(y_mid - cut_size / 2)
            if cut_y1 < 0:
                cut_y1 = 0
            cut_y2 = int(y_mid + cut_size / 2)
            if cut_y2 > w:
                cut_y2 = w

            # 要裁剪的坐标区域
            cut_pos = (cut_x1, cut_y1, cut_x2, cut_y2)

            # if name in labels:
            if True:
                # 生成一个新的xml文件
                sub_filename = filename.replace(' ', '_')

                newXmlRoot = getNewXml(img_filename = sub_filename + ".jpg",
                                   dirPath=dist_image_dir,
                                   width=cut_x2-cut_x1,
                                   height=cut_y2-cut_y1)


                # 开始根据pos遍历子label
                for sub_member in xml_root.findall("object"):

                    sub_name = sub_member.find("name").text
                    s_x1 = math.floor(float(sub_member.find("bndbox")[0].text))
                    s_y1 = math.floor(float(sub_member.find("bndbox")[1].text))
                    s_x2 = math.floor(float(sub_member.find("bndbox")[2].text))
                    s_y2 = math.floor(float(sub_member.find("bndbox")[3].text))
                    line = None
                    # if sub_member.find("line") is not None:
                    #     line = sub_member.find("line").text
                    sub_pos = (s_x1, s_y1, s_x2, s_y2)
                    # 如果是子label，并且在parent box内，那么计算新的pos信息，将pos信息追加到xml文件中
                    # if sub_name in labels and isInBox2(parBoxPos=cut_pos, sonBoxPos=sub_pos):
                    if isInBox2(parBoxPos=cut_pos, sonBoxPos=sub_pos):
                        n_x1, n_y1, n_x2, n_y2 = newPos2(parBoxPos=cut_pos, sonBoxPos=sub_pos)
                        newXmlRoot = newObjElement(newXmlRoot, n_x1, n_y1, n_x2, n_y2, sub_name, line)
                        # ET.dump(newXmlRoot)

                # 最后导出xml文件，并裁剪图片
                indent(newXmlRoot)
                tree = ET.ElementTree(newXmlRoot)
                # tree.write("{}{}_{}_clear_{}.xml".format(dist_xml_dir, sub_filename, obj_iter,clear), encoding="utf-8")
                cut_xml_filename = "{}_{}.xml".format(sub_filename, obj_iter)
                tree.write(os.path.join(dist_xml_dir, cut_xml_filename), encoding="utf-8")
                # 裁剪图片
                img_cut = img_file[cut_y1:cut_y2, cut_x1:cut_x2]
                # 导出图片
                cut_img_filename = "{}_{}.jpg".format(sub_filename, obj_iter)
                print(os.path.join(dist_image_dir, cut_img_filename))
                cv2.imwrite(os.path.join(dist_image_dir, cut_img_filename), img_cut)
                # cv2.imwrite("{}{}_{}_clear_{}.jpg".format(dist_image_dir, sub_filename, obj_iter,clear), img_cut)


def expandBorder(x1, x2, y1, y2, border, width, height):
    n_x1 = math.floor(float(x1)) - border
    if n_x1 < 0:
        n_x1 = 0
    n_y1 = math.floor(float(y1)) - border
    if n_y1 < 0:
        n_y1 = 0
    n_x2 = math.floor(float(x2)) + border
    if n_x2 > width:
        n_x2 = width
    n_y2 = math.floor(float(y2)) + border
    if n_y2 > height:
        n_y2 = height
    return str(n_x1), str(n_x2), str(n_y1), str(n_y2)

# def initEmpthFile(file_path):
#     # 是否存在
#
#     # 创建目录
#
#     # 创建文件