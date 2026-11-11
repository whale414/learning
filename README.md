# 视觉组考核

## 项目简介：

本项目为考核相关代码：

1. 图像接入与相机参数：
订阅压缩图像和相机参数，通过 cv_bridge 将压缩图像转化为可以被toshare识别的形式，最后发布到“/ft/debug/image”上面

2. 


## 编译和运行：

### 1.编译：

'''
cv ~/vision_final_test
catkin_make
source devel/setup.bash
'''

### 2.运行：

1. 终端1：
'''
roslaunch ft_detector launch.launch
'''
<!--这个终端只是单纯打开rqt，但由于没有bag的回放，无法生成对应包内画面-->

2. 终端2：
'''
rosvag play --clock -l /root/data/data01/ <!--加上对应包名字-->
'''