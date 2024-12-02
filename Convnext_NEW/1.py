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
from modelss.CGAN.cgan import generator, discriminator
from modelss.convnexts.convnext import ConvNeXt

from functional import pre_reshape_1, compute_MSE_r_1, compute_MSE_theta_1, compute_MSE_2D, fft_shrink, add_noise_improve, real_imag_stack, tensor_reshape, Beam_Squint_trajectory, CBS_theta
from thop import profile
# from tensorboardX import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 100
BATCH_SIZE = 1
LR1 = 0.0005
LR2 = 0.0005
LR3 = 0.0005
img_height = 64
img_width = 32
Step = 0

RMSE_R = np.zeros([EPOCH,1], dtype=np.double)
RMSE_THETA = np.zeros([EPOCH,1], dtype=np.double)
RMSE_2D_g = np.zeros([EPOCH,1], dtype=np.double)

mat = h5py.File('./datasets/data_1000_5m_50m_y_1.mat', 'r')  # 读取文件，得到字典
x_train = mat['x1']  # 获取H_ori数据
x_train = np.transpose(x_train)
print(np.shape(x_train))

mat1 = h5py.File('./datasets/data_1000_5m_50m_y_2.mat', 'r')  # 读取文件，得到字典
x_train1 = mat1['x2']  # 获取H_ori数据
x_train1 = np.transpose(x_train1)
print(np.shape(x_train1))

mat2 = h5py.File('./datasets/data_1000_5m_50m_y_3.mat', 'r')  # 读取文件，得到字典
x_train2 = mat2['x3']  # 获取H_ori数据
x_train2 = np.transpose(x_train2)
print(np.shape(x_train2))

mat3 = h5py.File('./datasets/data_1000_5m_50m_y_4.mat', 'r')  # 读取文件，得到字典
x_train3 = mat3['x4']  # 获取H_ori数据
x_train3 = np.transpose(x_train3)
print(np.shape(x_train3))

mat4 = h5py.File('./datasets/data_1000_5m_50m_y_5.mat', 'r')  # 读取文件，得到字典
x_train4 = mat4['x5']  # 获取H_ori数据
x_train4 = np.transpose(x_train4)
print(np.shape(x_train4))

mat5 = h5py.File('./datasets/data_1000_5m_50m_y_6.mat', 'r')  # 读取文件，得到字典
x_train5 = mat5['x6']  # 获取H_ori数据
x_train5 = np.transpose(x_train5)
print(np.shape(x_train5))

mat6 = h5py.File('./datasets/data_1000_5m_50m_y_label.mat', 'r')  # 读取文件，得到字典
x_train6 = mat6['label']  # 获取H_ori数据
x_train6 = np.transpose(x_train6)
print(np.shape(x_train6))

H_get = pre_reshape_1(x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, img_height, img_width*2)
print(np.shape(H_get))  # 4000 64 32
train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)

G_net = generator(in_chans=2, out_chans=6, embed_dim=96, depths=[1,1,1,1], num_heads=[8,8,8,8])
D_net = discriminator(in_chans=6, depths=[3, 3, 9, 3], dims=[96,192,384,768] , num_classes=1)
C_net = ConvNeXt(in_chans=6, depths=[3, 3, 9, 3], dims=[96,192,384,768], num_classes=2)
device_ids = [0]

# input = torch.randn(1, 2, 64, 64)
# flops, params = profile(G_net , inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (generator, params / (1000 ** 2), flops / (1000 ** 3)))  #这里除以1000的平方，是为了化成M的单位，
#
# input = torch.randn(1, 6, 64, 64)
# flops, params = profile(D_net , inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (discriminator, params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位，
#
# input = torch.randn(1, 6, 64, 64)
# flops, params = profile(C_net , inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (ConvNeXt, params / (1000 ** 2), flops / (1000 ** 3)))   #这里除以1000的平方，是为了化成M的单位，


G = nn.DataParallel(G_net, device_ids=device_ids).cuda()
D = nn.DataParallel(D_net, device_ids=device_ids).cuda()
C = nn.DataParallel(C_net, device_ids=device_ids).cuda()

BCE_loss = nn.BCELoss()
BCE_loss.cuda()
Loss = nn.MSELoss()
Loss.cuda()

if torch.cuda.is_available():
    device_ = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device_ = torch.device("cpu")
    print("Running on the CPU")

G_optimizer = torch.optim.Adam(G.parameters(), lr=LR1)
D_optimizer = torch.optim.Adam(D.parameters(), lr=LR2)
C_optimizer = torch.optim.Adam(C.parameters(), lr=LR3)

y_real_ = torch.ones([BATCH_SIZE,1])
y_fake_ = torch.zeros([BATCH_SIZE,1])
y_real_, y_fake_ = y_real_.cuda(), y_fake_.cuda()

print('training start!')
T1 = time.perf_counter()
start_time = time.time()

for epoch in range(EPOCH):

    for i, x in enumerate(train_loader, 0):

        # train discriminator D
        x = x.cuda()
        D.zero_grad()

        z_ = torch.zeros([BATCH_SIZE, 2, 64, 64], device=device_)
        z_[:, 0, :, :] = torch.randn(BATCH_SIZE, 64, 64)
        z_[:, 1, :, :] = z_[:, 1, :, :] + x[:, 6, :, :]
        # z_ = Variable(z_)
        G_result = G(z_)


        D_real_result = D(x[:, :6, :, :])
        D_fake_result = D(G_result)
        D_real_loss = BCE_loss(D_real_result, y_real_)
        D_fake_loss = BCE_loss(D_fake_result, y_fake_)

        real_img = torch.zeros([BATCH_SIZE, 64, 64], device=device_)
        real_img = real_img + x[:, 6, :, :]
        # real_img = Variable(real_img.cuda())

        C_real_result = C(x[:,:6,:,:])
        C_fake_result = C(G_result)
        C_real_loss = Loss(C_real_result, real_img[:, 0, :2])
        C_fake_loss = Loss(C_fake_result, real_img[:, 0, :2])

        D_train_loss = D_real_loss + D_fake_loss + C_real_loss + C_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # train generator G
        G.zero_grad()

        z_ = torch.zeros([BATCH_SIZE, 2, 64, 64], device=device_)
        z_[:, 0, :, :] = torch.randn(BATCH_SIZE, 64, 64)
        z_[:, 1, :, :] = z_[:, 1, :, :] + x[:, 6, :, :]
        # z_ = Variable(z_)
        G_result = G(z_)
        D_result = D(G_result)

        C_fake_result = C(G_result)
        C_fake_loss = Loss(C_fake_result, real_img[:, 0, :2])
        G_train_loss = BCE_loss(D_result, y_real_) + C_fake_loss

        G_train_loss.backward()
        G_optimizer.step()

        # train classifier C
        C.zero_grad()

        z_ = torch.zeros([BATCH_SIZE, 2, 64, 64], device=device_)
        z_[:, 0, :, :] = torch.randn(BATCH_SIZE, 64, 64)
        z_[:, 1, :, :] = z_[:, 1, :, :] + x[:, 6, :, :]
        # z_ = Variable(z_)
        G_result = G(z_)

        # real_img = torch.zeros([BATCH_SIZE, 64, 64])
        # real_img = real_img + x[:, 6, :, :]
        # real_img = Variable(real_img.cuda())

        C_real_result = C(x[:, :6, :, :])
        C_fake_result = C(G_result)
        C_real_loss = Loss(C_real_result, real_img[:, 0, :2])
        C_fake_loss = Loss(C_fake_result, real_img[:, 0, :2])
        C_train_loss = C_real_loss + C_fake_loss

        C_train_loss.backward()
        C_optimizer.step()

        r_theta_hat = C_real_result.cpu().detach().numpy()

        r_theta = np.array([real_img[:, 0, 0].cpu().detach().numpy(), real_img[:, 0, 1].cpu().detach().numpy()]).T
        # print(r_theta[:, 1])
        RMSE_r = compute_MSE_r_1(r_theta_hat, r_theta)
        RMSE_theta = compute_MSE_theta_1(r_theta_hat, r_theta)
        # RMSE_2D = compute_MSE_2D(r_theta_hat, r_theta)

        RMSE_R[epoch, :] = RMSE_r / (len(train_loader)) + RMSE_R[epoch, :]
        RMSE_THETA[epoch, :] = RMSE_theta / (len(train_loader)) + RMSE_THETA[epoch, :]
        # RMSE_2D_g[epoch,:] = RMSE_2D / 3125 + RMSE_2D_g[epoch,:]

        T2 = time.perf_counter()
        if Step % 20 == 0:
            # print("[epoch %d][%d/%d]   D_loss: %.4f   G_loss: %.4f   C_loss: %.4f   RMSE_r: %.8f   RMSE_theta: %.8f   Time: %.4f " % (epoch + 1, i + 1, len(train_loader), D_train_loss, G_train_loss, C_real_loss, RMSE_r, RMSE_theta, (T2 - T1)))
            print(
                "[epoch {:2d}][{:3d}/{:3d}]   D_loss: {:.4f}   G_loss: {:.4f}   C_loss: {:.4f}   RMSE_r: {:.8f}   RMSE_theta: {:.8f}   Time: {:.4f}".format(
                    epoch + 1, i + 1, len(train_loader), D_train_loss, G_train_loss, C_real_loss, RMSE_r, RMSE_theta,
                    (T2 - T1)))

        Step += 1

np.savetxt('./model_pt/GAN_convnext_RMSE_r_1.csv', RMSE_R, delimiter=',')
np.savetxt('./model_pt/GAN_convnext_RMSE_theta_1.csv', RMSE_THETA, delimiter=',')
# np.savetxt('./model_pt/convnext_RMSE_2D.csv', RMSE_2D_g, delimiter=',')

torch.save(C.state_dict(), './model_pt/GAN_C_convnext_1.pt')
torch.save(G.state_dict(), './model_pt/GAN_G_SUNet_1.pt')
torch.save(D.state_dict(), './model_pt/GAN_D_convnext_1.pt')



