# -*- encoding=utf 8-*-
"""
@author: Roger
@date :

"""


import os
import cv2
import h5py
import numpy as np
import random

def write_hdf5(file, data,label_landmarks):
  # transform to np array
  data_arr = np.array(data, dtype = np.float32)
  # print data_arr.shape
  # if no swapaxes, transpose to num * channel * width * height ???
  # data_arr = data_arr.transpose(0, 3, 2, 1)
  label_landmarks_arr = np.array(label_landmarks, dtype = np.float32)
  with h5py.File(file, 'w') as f:
    f['X'] = data_arr
    f['y'] = label_landmarks_arr

# list_file format:
# image_path | label_landmarks(10)
def convert_dataset_to_hdf5(list_file, path_data, path_save,size_hdf5, tag):
  with open(list_file, 'r') as f:

    annotations = f.readlines()
  random.shuffle(annotations)
  random.shuffle(annotations)
  random.shuffle(annotations)
  random.shuffle(annotations)
  random.shuffle(annotations)
  random.shuffle(annotations)


  num = len(annotations)
  print "%d pics in total" % num
  random.shuffle(annotations)

  data = []

  label_landmarks = []
  count_data = 0
  count_hdf5 = 0
  for line in annotations:
    # print(line)
    line_split = line.strip().split(' ')
    assert len(line_split) == 2
    path_full = os.path.join(path_data, line_split[0])
    # #print path_full
    datum = cv2.imread(path_full)
    if datum is None:
      continue

    datum = cv2.resize(datum, (24, 24))

    landmarks = line_split[1:]

    tmp = datum[:, :, 2].copy()
    datum[:, :, 2] = datum[:, :, 0]
    datum[:, :, 0] = tmp
    datum = (datum - 127.5) * 0.0078125 # [0,255] -> [-1,1]
    transposed_img = datum.transpose((2, 0, 1))

    data.append(transposed_img)
    label_landmarks.append(landmarks)
    count_data = count_data + 1
    if 0 == count_data % size_hdf5:
      path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
      write_hdf5(path_hdf5, data,label_landmarks)
      count_hdf5 = count_hdf5 + 1
      data = []
      label_landmarks = []
      print count_data
  # handle the rest
  if data:
    path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
    write_hdf5(path_hdf5, data,label_landmarks)
  print "count_data: %d" % count_data

def main():

  list_file = "val.txt"#txt文件
  path_data = ""#当前项目的根目录
  path_save = ""#生成的hdf5存储的文件夹路径
  size_hdf5 = 1000#设置每个hdf5的数量大小
  tag = 'val_'#生成hdf5文件前缀

  assert os.path.exists(path_data)
  if not os.path.exists(path_save):
    os.makedirs(path_save)
  assert size_hdf5 > 0

  # convert
  convert_dataset_to_hdf5(list_file, path_data, path_save, size_hdf5, tag)


if __name__ == '__main__':
    main()
