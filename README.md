# eye_state_det
该项目是通过神经网络训练检测人眼识别的完整项目,包含具体的训练和测试过程的代码。<br>
## 环境依赖：
opencv<br>
dlib<br>
caffe<br>
## 文件说明
dataset里面是训练模型用到的图片数据集<br>
eye_model是已经训练好的模型<br>
genData为生成训练数据文件<br>
test里面包好测试代码<br>
train里面包含神经网络训练的相关文件<br>
## 快速执行
    python dlib_main.py(注：将caffe的路径改为你电脑本地的路径)
## 训练
1、修改creat_hdf5.py里面对应的文件路径，生成训练所需的hdf5数据集   <br>
#####
    list_file = "val.txt"#txt文件
    path_data = ""#当前项目的根目录
    path_save = ""#生成的hdf5存储的文件夹路径
    size_hdf5 = 1000#设置每个hdf5的数量大小
    tag = 'val_'#生成hdf5文件前缀
#####
     python creat_hdf5.py
 
2、修改网络文件里面对应的路径<br>
3、执行<br>
#####
    sh train.sh
    



