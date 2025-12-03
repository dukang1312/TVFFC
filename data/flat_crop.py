# # 导入所需库
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import math
# from mpl_toolkits.mplot3d import Axes3D
# import warnings
# # 原函数
# def Z(x,y):
#     return 2*(x-10)**2 + y**2
# # x方向上的梯度
# def dx(x):
#     return 4*x-4
# # y方向上的梯度
# def dy(y):
#     return 2*y
# # 初始值
# X = x_0 = 150
# Y = y_0 = 2
# # 学习率
# alpha = 0.1
# # 保存梯度下降所经过的点
# globalX = [x_0]
# globalY = [y_0]
# globalZ = [Z(x_0,y_0)]
# # 迭代30次
# for i in range(30):
#     temX = X - alpha * dx(X)
#     temY = Y - alpha * dy(Y)
#     temZ = Z(temX, temY)
#     # X,Y 重新赋值
#     X = temX
#     Y = temY
#     # 将新值存储起来
#     globalX.append(temX)
#     globalY.append(temY)
#     globalZ.append(temZ)
# # 打印结果
# print(u"最终结果为:(x,y,z)=(%.5f, %.5f, %.5f)" % (X, Y, Z(X,Y)))
# print(u"迭代过程中取值")
# num = len(globalX)
# for i in range(num):
#     print(u"x%d=%.5f, y%d=%.5f, z%d=%.5f" % (i,globalX[i],i,globalY[i],i,globalZ[i]))
from PIL import Image
img = Image.open("G:/new/datasets/cityscapes/train_A/float_image_shuffle_1_group_1_2907062_out_1755054_out_2286131_out_1701035_out_2265038_out_1743099_out_2625076_out_cropped_420_179.png")
import numpy as np

print(img)
