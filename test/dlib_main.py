# -*- coding: UTF-8 -*-
# @Time      :
# @author    : Roger
# @software  : pyCharm Community Edition

"""
该项目为基于dlib人脸检测进行的人眼状态的识别
"""
import cv2
import dlib
from eye_cnn import detectFace_eye
# 使用默认人脸识别的模型
detector = dlib.get_frontal_face_detector()
# 获取人脸关键点预训练模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#视频测试
def videoTest():
    cap = cv2.VideoCapture(0)
    while(True):
        ret,frame = cap.read()#返回两个值，第一个为bool类型，如果读到帧返回True,如果没读到帧返回False,第二个值为帧图像
        if(ret == True):
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 1)
            for face in dets:
                shape = predictor(gray, face)  # 寻找人脸的68个标定点
                # detect left eye
                lx0 = shape.parts()[17].x
                ly0 = shape.parts()[19].y
                lx1 = shape.parts()[29].x
                ly1 = shape.parts()[29].y
                lEye_img = frame[int(ly0):int(ly1), int(lx0):int(lx1)]
                lEye = detectFace_eye(lEye_img)

                j = lEye.argsort()[-1]
                if j == 0:
                    cv2.rectangle(frame, (int(lx0), int(ly0)), (int(lx1), int(ly1)), (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, (int(lx0), int(ly0)), (int(lx1), int(ly1)), (255, 255, 0), 1)

                # detect right eye
                rx0 = shape.parts()[29].x
                ry0 = shape.parts()[24].y
                rx1 = shape.parts()[26].x
                ry1 = shape.parts()[29].y
                rEye_img = frame[int(ry0):int(ry1), int(rx0):int(rx1)]
                rEye = detectFace_eye(rEye_img)

                z = rEye.argsort()[-1]

                if z == 0:
                    cv2.rectangle(frame, (int(rx0), int(ry0)), (int(rx1), int(ry1)), (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, (int(rx0), int(ry0)), (int(rx1), int(ry1)), (255, 255, 0), 1)

                # 遍历所有点，打印出其坐标，并圈出来
                for i, pt in enumerate(shape.parts()):
                    pt_pos = (pt.x, pt.y)
                    cv2.circle(frame, pt_pos, 2, (0, 255, 0), 1)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1)==27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    #视频测试
    videoTest()



