# eye_state_det
该项目为检测人眼是处在张开还是闭合的完整项目,包含具体的训练和测试过程的代码。<br>
## 环境依赖：
opencv<br>
dlib<br>
caffe<br>
## 文件说明
dataset里面是训练模型用到的图片数据集<br>
eye_model是已经训练好的模型<br>
test里面包好测试代码<br>
train里面包含神经网络训练的相关文件<br>
## 快速执行
    python dlib_main.py(注：将caffe的路径改为你电脑本地的路径)
## 训练
1、将dataset里面的图片数据装换成训练用到的hdf5格式数据集<br>
2、修改网络文件里面对应的路径<br>
3、执行<br>
##
    sh train.sh
    



