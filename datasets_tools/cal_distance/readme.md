## 介绍
- 该工具包是**Augment Core**模块的实现，主要有图片距离计算，自适应风格增强等功能。
- 主要有以下几个脚本构成
  1. adain_add_style.py 基于adain模型给图片进行风格增强
  2. compress_radios.py 基于png压缩算法计算给定图片压缩后的bit length
  3. compress_dis.py 基于bit length计算图片之间的距离
  4. augment_core.py 基于图片距离进行自适应风格增强

## 使用方法
### step1: adain_add_style.py 执行下面的命令进行风格增强，得到不同风格强度的数据集
```shell
python ./adain_add_style.py --data ./sub_datasets_foggy/origin_data --decoder_path ../../TargetAugment_train/models/city2foggy/decoder_iter_160000.pth --encoder_path ../../TargetAugment_train/pre_trained/vgg16_ori.pth --fc1 ../../TargetAugment_train/models/city2foggy/fc1_iter_160000.pth --fc2 ../../TargetAugment_train/models/city2foggy/fc2_iter_160000.pth --style_add_alpha 0.4 --style_path ../../TargetAugment_train/data/meanfoggy/meanfoggy.jpg --device 0 --save_style_samples
```

### step2: resize_origin_images.py 执行下面的命令将原始图片resize成给定大小，与进行风格增强后的图片大小保持一致
```shell
python ./resize_origin_images.py
```


### step3: compress_radios.py 计算不同数据集的compress radios，生成compress_radios.csv文件
```shell
python compress_radios.py
```

### step4: compress_dis.py 基于compress_radios.csv文件，计算不同domain之间的距离并完成分类，生成compress_ids.csv文件
```shell
python compress_dis.py
```

### step5: augment_core.py 基于compress_dis.csv文件进行自适应增强逻辑
```shell
python augment_core.py --decoder_path ../../TargetAugment_train/models/city2foggy/decoder_iter_160000.pth --encoder_path ../../TargetAugment_train/pre_trained/vgg16_ori.pth --fc1 ../../TargetAugment_train/models/city2foggy/fc1_iter_160000.pth --fc2 ../../TargetAugment_train/models/city2foggy/fc2_iter_160000.pth --style_path ../../TargetAugment_train/data/meanfoggy/meanfoggy.jpg --device 0 --save_style_samples
```

## 将AC结果融入Mean-teacher框架中
- 使用AC得到自适应增强猴的图片之后，在MT训练过程中student模型的输入不需要进行实时的风格增强直接读取现成的图片（AC生成结果）即可，**但需要确保图片名一一对应**