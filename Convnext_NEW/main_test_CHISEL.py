import h5py
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
import os
import math
import time
# from scipy.fftpack import fft2, ifft2, fft, ifft
from modelss.CHISEL.CHISEL import CHISEL
from functional import pre_reshape_11, compute_MSE_r_test1, compute_MSE_theta_test1, compute_MSE_2D1, fft_shrink, add_noise_improve, real_imag_stack, tensor_reshape, Beam_Squint_trajectory, CBS_theta
from thop import profile
# from tensorboardX import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 1
BATCH_SIZE = 1
LR = 0.0005
img_height = 64
img_width = 32
Step = 0

RMSE_R = np.zeros([EPOCH,1], dtype=np.double)
RMSE_THETA = np.zeros([EPOCH,1], dtype=np.double)
RMSE_2D_g = np.zeros([EPOCH,1], dtype=np.double)

mat = h5py.File('./datasets/test_5000_5m_50m_y_20dB.mat', 'r')  # 读取文件，得到字典
x_train = mat['x1']  # 获取H_ori数据
x_train = np.transpose(x_train)
print(np.shape(x_train))

x_train1 = mat['x2']  # 获取H_ori数据
x_train1 = np.transpose(x_train1)
print(np.shape(x_train1))

x_train2 = mat['x3']  # 获取H_ori数据
x_train2 = np.transpose(x_train2)
print(np.shape(x_train2))

x_train3 = mat['x4']  # 获取H_ori数据
x_train3 = np.transpose(x_train3)
print(np.shape(x_train3))

x_train4 = mat['x5']  # 获取H_ori数据
x_train4 = np.transpose(x_train4)
print(np.shape(x_train4))

x_train5 = mat['x6']  # 获取H_ori数据
x_train5 = np.transpose(x_train5)
print(np.shape(x_train5))

x_train6 = mat['x7']  # 获取H_ori数据
x_train6 = np.transpose(x_train6)
print(np.shape(x_train6))

x_train7 = mat['x8']  # 获取H_ori数据
x_train7 = np.transpose(x_train7)
print(np.shape(x_train7))

x_train8 = mat['label']  # 获取H_ori数据
x_train8 = np.transpose(x_train8)
print(np.shape(x_train8))

H_get = pre_reshape_11(x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, img_height, img_width*2)
print(np.shape(H_get))  # 4000 64 32
train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)

G_net = CHISEL(input_channel=8, layersize=1, numoffilters=8, t = 7)
device_ids = [0]
# print(G_net)

# input = torch.randn(1, 8, 64, 64)
# flops, params = profile(G_net , inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (CHISEL, params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，

RMSE_2D_CDF = np.zeros([(len(train_loader)),1], dtype=np.double)

G = nn.DataParallel(G_net, device_ids=device_ids).cuda()

G.load_state_dict(torch.load('./model_pt/CHISEL_7_NLoS.pt'))

if torch.cuda.is_available():
    device_ = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device_ = torch.device("cpu")
    print("Running on the CPU")


T1 = time.perf_counter()
for epoch in range(EPOCH):

    for i, x in enumerate(train_loader, 0):
        # x = x.cuda()

        real_img = torch.zeros([BATCH_SIZE, 64, 64])
        real_img = real_img + x[:, 8, :, :]
        # print(real_img[:, 0, :2])
        # print(np.shape(x[:, 0, :, :]))
        real_img = Variable(real_img.cuda())  # H

        fake_img = G(x[:,:8,:,:])  # 随机噪声输入到生成器中，得到一副假的图片   4*2

        r_theta_hat = fake_img.cpu().detach().numpy()
        # print(real_img[:, 0, 1])

        r_theta = np.array([real_img[:, 0, 0].cpu().detach().numpy(), real_img[:, 0, 1].cpu().detach().numpy()]).T
        # print(r_theta[:, 1])
        RMSE_r = compute_MSE_r_test1(r_theta_hat, r_theta)
        RMSE_theta = compute_MSE_theta_test1(r_theta_hat, r_theta)
        RMSE_2D = compute_MSE_2D1(r_theta_hat, r_theta)

        RMSE_R[epoch, :] = RMSE_r / (len(train_loader)) + RMSE_R[epoch, :]
        RMSE_THETA[epoch, :] = RMSE_theta / (len(train_loader)) + RMSE_THETA[epoch, :]
        RMSE_2D_g[epoch, :] = RMSE_2D / (len(train_loader)) + RMSE_2D_g[epoch, :]

        RMSE_2D_CDF[i, :] = np.sqrt(RMSE_2D)


        T2 = time.perf_counter()
        if Step % 20 == 0:
            print("[epoch %d][%d/%d]   RMSE_r: %.8f   RMSE_theta: %.8f   RMSE_d: %.8f   Time: %.4f " % (
            epoch + 1, i + 1, len(train_loader), RMSE_r, RMSE_theta, RMSE_2D, (T2 - T1)))

        Step += 1


np.savetxt('./model_pt/NLoS_CDF1.csv', RMSE_2D_CDF, delimiter=',')
RMSE_R = np.sqrt(RMSE_R)
RMSE_THETA = np.sqrt(RMSE_THETA)
RMSE_2D_g = np.sqrt(RMSE_2D_g)
print("RMSE_R", RMSE_R)
print("RMSE_THETA", RMSE_THETA)
print("RMSE_2D_g", RMSE_2D_g)