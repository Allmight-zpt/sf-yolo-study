import os
from lxml import etree
import cv2
 
# 修改输入图片文件夹
img_folder = "foggy_cityscapes_beta_0.01_voc_format/VOC2007/JPEGImages/"
img_list = os.listdir(img_folder)
img_list.sort()
# 修改输入标签文件夹
label_folder = "foggy_cityscapes_beta_0.01_voc_format/VOC2007/Annotations/"
label_list = os.listdir(label_folder)
label_list.sort()
# 输出图片文件夹位置
path = os.getcwd()
output_folder = path + '/' + str("output_voc")
os.mkdir(output_folder)

# 读取 xml 文件信息，并返回字典形式
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回 tag对应的信息
        return {xml.tag: xml.text}
 
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
 
 
# xml 标注文件的可视化
def xmlShow(img,xml):
    image = cv2.imread(img)
    with open(xml, encoding='gb18030', errors='ignore') as fid:  # 防止出现非法字符报错
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]  # 读取 xml文件信息
 
    ob = []         # 存放目标信息
    for i in data['object']:        # 提取检测框
        name = str(i['name'])        # 检测的目标类别
 
        bbox = i['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
 
        tmp = [name,xmin,ymin,xmax,ymax]    # 单个检测框
        ob.append(tmp)
 
    # 绘制检测框
    for name,x1,y1,x2,y2 in ob:
        cv2.rectangle(image,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)    # 绘制矩形框
        cv2.putText(image,name,(x1,y1-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,thickness=1,color=(0,0,255))
 

    return image
 
    # 展示图像
    # cv2.imshow('test',image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    for i in range(len(img_list)):
        image_path = img_folder + "/" + img_list[i]
        label_path = label_folder + "/" + label_list[i]
        img = xmlShow(img=image_path, xml=label_path)
        cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)