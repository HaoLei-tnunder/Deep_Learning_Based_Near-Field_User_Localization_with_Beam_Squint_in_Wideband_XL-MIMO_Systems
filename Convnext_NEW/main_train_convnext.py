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
from scipy.fftpack import fft2, ifft2, fft, ifft
from modelss.convnexts.convnext import ConvNeXt
from functional import pre_reshape, compute_MSE_r, compute_MSE_theta, compute_MSE_2D, fft_shrink, add_noise_improve, real_imag_stack, tensor_reshape, Beam_Squint_trajectory, CBS_theta
from thop import profile
from tensorboardX import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 400
BATCH_SIZE = 100
LR = 0.001
focus = 0.1
img_height = 32
img_width = 16
img_channels = 3

Step = 0

Nt = 256
fc = 100e9
c = 3e8
B = 6e9
lambda_c = c / fc
d = lambda_c / 2
M = 512
f = np.zeros(M)
for m in range(1, M + 1):
    f[m - 1] = fc + B / 2 * (2 * m / (M - 1) - 1)

Rmin = 5
Rmax = 50
user_theta_max = 60 / 180 * np.pi
user_theta_min = -60 / 180 * np.pi
r0 = (Rmin + Rmax) / 2  # start
theta0 = user_theta_max
rc = (Rmin + Rmax) / 2  # end
thetac = user_theta_min
theta_M, _ = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)


RMSE_R = np.zeros([EPOCH,1], dtype=np.double)
RMSE_THETA = np.zeros([EPOCH,1], dtype=np.double)
RMSE_2D_g = np.zeros([EPOCH,1], dtype=np.double)

mat = h5py.File('./datasets/data_50000_5m_50m_hsn.mat', 'r')  # 读取文件，得到字典
x_train = mat['x']  # 获取H_ori数据
data_len = np.shape(x_train)[0]
x_train = np.transpose(x_train)
print(np.shape(x_train))

mat1 = h5py.File('./datasets/data_50000_5m_50m_hsn_label.mat', 'r')  # 读取文件，得到字典
x_train1 = mat1['label']  # 获取H_ori数据
x_train1 = np.transpose(x_train1)
print(np.shape(x_train1))

H_get = pre_reshape(x_train, x_train1, img_height, img_width*2)
print(np.shape(H_get))  # 4000 64 32
train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)

G_net = ConvNeXt(in_chans=2,depths=[1, 1, 1, 1], dims=[96, 192, 384, 768], num_classes=2)
device_ids = [0]
# print(G_net)

# input = torch.randn(8, 3, 32, 32)
# flops, params = profile(G_net , inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (ConvNeXt, params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，

G = nn.DataParallel(G_net, device_ids=device_ids).cuda()
# for param in G.parameters():
#     param.data = param.data.to(torch.double)

# Loss = nn.L1Loss(reduction='mean')
Loss = nn.MSELoss()
Loss.cuda()

g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
# g_optimizer = torch.optim.AdamW(G.parameters(), lr=LR, weight_decay=0.005, betas=(0.9, 0.999), eps=1e-8 )

# G.load_state_dict(torch.load('./model_pt/convnext.pt'))

T1 = time.perf_counter()
for epoch in range(EPOCH):

    for i, x in enumerate(train_loader, 0):

        real_img = torch.zeros([BATCH_SIZE, 32, 32])
        real_img = real_img + x[:, 1, :, :]
        # print(real_img[:, 0, :2])
        # print(np.shape(x[:, 0, :, :]))
        real_img = Variable(real_img.cuda())  # H


        sx = x[:, 0, :, :].numpy()
        theta_train_data = np.zeros([BATCH_SIZE, img_height, img_width * 2], dtype=np.float32)  # 64 32
        H_train_data = sx
        H = fft_shrink(H_train_data, img_height, img_width)  # 64 32
        noise = add_noise_improve(H, 0, 20)                           # noise   ,  sigma
        H_n = noise + H

        y_n = np.reshape(H_n, [BATCH_SIZE, M], order='F')
        for ii in range(BATCH_SIZE):
            theta_hat = CBS_theta(y_n[ii,:], M, theta_M)
            theta_train_data[ii,:,:] = theta_hat

        # theta_fft_stack = tensor_reshape(theta_train_data)
        H_n_fft_r_i = real_imag_stack(H_n)                                  #64  64
        # H_n_fft_stack = tensor_reshape(H_n_fft_r_i)

        input1 = np.zeros([BATCH_SIZE,2,32,32], dtype=np.float32)
        input1[:, 0, :, :] = H_n_fft_r_i
        input1[:, 1, :, :] = theta_train_data
        H_n_fft_stack = torch.zeros([BATCH_SIZE, 2, 32, 32], dtype=torch.float32)
        H_n_fft_stack = H_n_fft_stack + input1
        H_n_fft_train = Variable(H_n_fft_stack.cuda())


        fake_img = G(H_n_fft_train)  # 随机噪声输入到生成器中，得到一副假的图片   4*2
        # print(np.shape(fake_img))
        # print(type(fake_img))
        g_loss = (Loss(fake_img, real_img[:, 0, :2]))
        # g_loss = (Loss(fake_img[:,0] , real_img[:, 0, 0])) + (Loss(fake_img[:,1], real_img[:, 0, 1]))

        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数


        r_theta_hat = fake_img.cpu().detach().numpy()
        # print(real_img[:, 0, 1])

        r_theta = np.array([real_img[:, 0, 0].cpu().detach().numpy(), real_img[:, 0, 1].cpu().detach().numpy()]).T
        # print(r_theta[:, 1])
        RMSE_r = compute_MSE_r(r_theta_hat, r_theta)
        RMSE_theta = compute_MSE_theta(r_theta_hat, r_theta)
        # RMSE_2D = compute_MSE_2D(r_theta_hat, r_theta)

        RMSE_R[epoch, :] = RMSE_r / (len(train_loader)) + RMSE_R[epoch, :]
        RMSE_THETA[epoch, :] = RMSE_theta / (len(train_loader)) + RMSE_THETA[epoch, :]
        # RMSE_2D_g[epoch,:] = RMSE_2D / 3125 + RMSE_2D_g[epoch,:]

        T5 = time.perf_counter()
        if Step % 20 == 0:
            print("[epoch %d][%d/%d]   g_loss: %.4f   RMSE_r: %.8f   RMSE_theta: %.8f   Time: %.4f " % (epoch + 1, i + 1, len(train_loader), g_loss, RMSE_r, RMSE_theta, (T5 - T1)))

        Step += 1

np.savetxt('./model_pt/convnext_RMSE_r_111_4.csv', RMSE_R, delimiter=',')
np.savetxt('./model_pt/convnext_RMSE_theta_111_4.csv', RMSE_THETA, delimiter=',')
# np.savetxt('./model_pt/convnext_RMSE_2D.csv', RMSE_2D_g, delimiter=',')

torch.save(G.state_dict(), './model_pt/convnext_111_4.pt')
