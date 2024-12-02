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
from modelss.convnexts.convnext import ConvNeXt
from functional import pre_reshape_1, compute_MSE_r_train_GPU, compute_MSE_theta_train_GPU, compute_MSE_2D, fft_shrink, add_noise_improve, real_imag_stack, tensor_reshape, Beam_Squint_trajectory, CBS_theta
from thop import profile
# from tensorboardX import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 500
BATCH_SIZE = 1
LR = 0.0005
img_height = 64
img_width = 32
Step = 0

RMSE_R = np.zeros([EPOCH,1], dtype=np.double)
RMSE_THETA = np.zeros([EPOCH,1], dtype=np.double)
RMSE_2D_g = np.zeros([EPOCH,1], dtype=np.double)

# mat = h5py.File('./datasets/data_50000_5m_50m_y_1.mat', 'r')  # 读取文件，得到字典
# x_train = mat['x1']  # 获取H_ori数据
# x_train = np.transpose(x_train)
# print(np.shape(x_train))
#
# mat1 = h5py.File('./datasets/data_50000_5m_50m_y_2.mat', 'r')  # 读取文件，得到字典
# x_train1 = mat1['x2']  # 获取H_ori数据
# x_train1 = np.transpose(x_train1)
# print(np.shape(x_train1))
#
# mat2 = h5py.File('./datasets/data_50000_5m_50m_y_3.mat', 'r')  # 读取文件，得到字典
# x_train2 = mat2['x3']  # 获取H_ori数据
# x_train2 = np.transpose(x_train2)
# print(np.shape(x_train2))
#
# mat3 = h5py.File('./datasets/data_50000_5m_50m_y_4.mat', 'r')  # 读取文件，得到字典
# x_train3 = mat3['x4']  # 获取H_ori数据
# x_train3 = np.transpose(x_train3)
# print(np.shape(x_train3))
#
# mat4 = h5py.File('./datasets/data_50000_5m_50m_y_5.mat', 'r')  # 读取文件，得到字典
# x_train4 = mat4['x5']  # 获取H_ori数据
# x_train4 = np.transpose(x_train4)
# print(np.shape(x_train4))
#
# mat5 = h5py.File('./datasets/data_50000_5m_50m_y_6.mat', 'r')  # 读取文件，得到字典
# x_train5 = mat5['x6']  # 获取H_ori数据
# x_train5 = np.transpose(x_train5)
# print(np.shape(x_train5))
#
# mat6 = h5py.File('./datasets/data_50000_5m_50m_y_7.mat', 'r')  # 读取文件，得到字典
# x_train6 = mat6['x7']  # 获取H_ori数据
# x_train6 = np.transpose(x_train6)
# print(np.shape(x_train6))
#
# mat7 = h5py.File('./datasets/data_50000_5m_50m_y_8.mat', 'r')  # 读取文件，得到字典
# x_train7 = mat7['x8']  # 获取H_ori数据
# x_train7 = np.transpose(x_train7)
# print(np.shape(x_train7))
#
# mat6 = h5py.File('./datasets/data_50000_5m_50m_y_label.mat', 'r')  # 读取文件，得到字典
# x_train8 = mat6['label']  # 获取H_ori数据
# x_train8 = np.transpose(x_train8)
# print(np.shape(x_train8))
#
# H_get = pre_reshape_1(x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, img_height, img_width*2)
# print(np.shape(H_get))  # 4000 64 32
# train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)

G_net = ConvNeXt(in_chans=8, depths=[3, 3, 27, 3], dims=[128,256,512,1024], num_classes=2)
device_ids = [0]
print(G_net)

input = torch.randn(1, 8, 64, 64)
flops, params = profile(G_net , inputs=(input, ),verbose=True)
print("%s | %.2f | %.2f" % (ConvNeXt, params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，

G = nn.DataParallel(G_net, device_ids=device_ids).cuda()
# for param in G.parameters():
#     param.data = param.data.to(torch.double)

Loss = nn.MSELoss()
Loss.cuda()

if torch.cuda.is_available():
    device_ = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device_ = torch.device("cpu")
    print("Running on the CPU")

g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
# g_optimizer = torch.optim.AdamW(G.parameters(), lr=LR, weight_decay=0.005, betas=(0.9, 0.999), eps=1e-8 )

T1 = time.perf_counter()
for epoch in range(EPOCH):

    for i, x in enumerate(train_loader, 0):
        x = x.cuda()

        real_img = torch.zeros([BATCH_SIZE, 64, 64], device=device_)
        real_img = real_img + x[:, 8, :, :]
        # print(real_img[:, 0, :2])
        # print(np.shape(x[:, 0, :, :]))

        fake_img = G(x[:,:8,:,:])  # 随机噪声输入到生成器中，得到一副假的图片   4*2
        # print(np.shape(fake_img))
        # print(type(fake_img))
        g_loss = (Loss(fake_img, real_img[:, 0, :2]))
        # g_loss = (Loss(fake_img[:,0] , real_img[:, 0, 0])) + (Loss(fake_img[:,1], real_img[:, 0, 1]))

        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        r_theta_hat = fake_img
        # print(real_img[:, 0, 1])

        r_theta = real_img[:, 0, :2]
        # print(r_theta[:, 1])
        RMSE_r = compute_MSE_r_train_GPU(r_theta_hat, r_theta)
        RMSE_theta = compute_MSE_theta_train_GPU(r_theta_hat, r_theta)
        # RMSE_2D = compute_MSE_2D(r_theta_hat, r_theta)

        RMSE_R[epoch, :] = RMSE_r.detach().cpu().numpy() / (len(train_loader)) + RMSE_R[epoch, :]
        RMSE_THETA[epoch, :] = RMSE_theta.detach().cpu().numpy() / (len(train_loader)) + RMSE_THETA[epoch, :]
        # RMSE_2D_g[epoch,:] = RMSE_2D / 3125 + RMSE_2D_g[epoch,:]

        T2 = time.perf_counter()
        if Step % 20 == 0:
            print("[epoch {:2d}][{:3d}/{:3d}]   C_loss: {:.4f}   RMSE_r: {:.8f}   RMSE_theta: {:.8f}   Time: {:.4f}".format( epoch + 1, i + 1, len(train_loader), g_loss, RMSE_r, RMSE_theta,(T2 - T1)))

        Step += 1

np.savetxt('./model_pt/convnext_RMSE_r_2.csv', RMSE_R, delimiter=',')
np.savetxt('./model_pt/convnext_RMSE_theta_2.csv', RMSE_THETA, delimiter=',')
# np.savetxt('./model_pt/convnext_RMSE_2D.csv', RMSE_2D_g, delimiter=',')

torch.save(G.state_dict(), './model_pt/convnext_2.pt')
