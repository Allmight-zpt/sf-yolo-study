# 标注格式转化工具包
## 介绍
### 工具包介绍
- 该工具包用于将原生json格式的标注转化为voc和yolo格式的标注并进行可视化
- gtFine和leftImg8bit_foggy文件夹中有少量的原始数据供测试

### Cityscapes数据集介绍
- Cityscapes包含50个城市不同场景、不同背景、不同季节的街景，提供5000张精细标注的图像、20000张粗略标注的图像、30类标注物体。用PASCAL VOC标准的 intersection-over-union （IoU）得分来对算法性能进行评价。 Cityscapes数据集共有fine和coarse两套评测标准，前者提供5000张精细标注的图像，后者提供5000张精细标注外加20000张粗糙标注的图像。
- 该数据集包含如下：leftImg8bit数据集的图片（data）部分，gtFine数据集的标注（annotation）部分
- 在leftImg8bit/train下有18个子文件夹对应16个德国城市、1个法国城市和1个瑞士城市；在leftImg8bit/val下有3个子文件夹对应3个德国城市；在leftImg8bit/test下有6个子文件夹对应6个德国城市。
- 在gtFine/train下有18个子文件夹对应leftImg8bit/train里面的文件夹，但是不一样的leftImg8bit里面的一张原图，对应着gtFine里面有6个文件分别是color.png、instanceIds.png、instanceTrainIds.png、labelIds.png、labelTrainIds.png、polygons.json（实际从官网下载到的数据集只有4个文件：color.png、instanceIds.png、labelIds.png、polygons.json）。分别表示不同的标签类型，此处目标检测任务只需要使用其中的polygons.json文件。

### 数据集下载
```
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
```

## 使用说明
- 将原生json标注格式转化为VOC标注格式
```
python cityscapes_foggy_to_voc.py(或cityscapes_to_voc.py)
```

- 可视化VOC标注格式，确认是否转化成功
```

python voc_show.py
```

- 将VOC标注格式转化为yolo标注格式
```
python voc2yolo.py
```

- 可视化yolo标注格式，确认是否转化成功
```
python yolo_show.py
```

- 将数据划分为test和train两个文件夹
```
python split_test_train.py
```

## 参考
格式转换：https://blog.csdn.net/moutain9426/article/details/120670104 <br>
数据集下载：https://blog.csdn.net/zisuina_2/article/details/116302128