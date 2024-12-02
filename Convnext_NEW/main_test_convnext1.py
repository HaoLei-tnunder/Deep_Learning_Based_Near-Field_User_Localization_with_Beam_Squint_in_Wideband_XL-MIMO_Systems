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
from modelss.convnexts.convnext import ConvNeXt
from functional import near_field_channel, Beam_Squint_trajectory, CBS_theta_high, generate_beam, CBS_r, pre_reshape_x, CBS_r_high
from numpy import floor, ceil
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 1
BATCH_SIZE = 1
LR = 0.001
img_height = 64
img_width = 32

SNR = np.arange(10, 31, 5)
len_snr = len(SNR)

Nt = 512
fc = 100e9
c = 3e8
B = 6e9
lambda_c = c / fc
d = lambda_c / 2
M = 2048
f = np.zeros(M)
for m in range(1, M + 1):
    f[m - 1] = fc + B / 2 * (2 * m / (M - 1) - 1)

N_iter = 2

Rmin = 5
Rmax = 50
user_theta_max = 60 / 180 * np.pi
user_theta_min = -60 / 180 * np.pi

# r0 = (Rmin + Rmax) / 2  # start
# theta0 = user_theta_max
# rc = (Rmin + Rmax) / 2  # end
# thetac = user_theta_min
# theta_M, _ = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)

RMSE_theta = np.zeros(len_snr, dtype=np.double)
RMSE_r = np.zeros(len_snr, dtype=np.double)
RMSE_theta_1 = np.zeros(len_snr, dtype=np.double)
RMSE_r_1 = np.zeros(len_snr, dtype=np.double)
RMSE_r_2 = np.zeros(len_snr, dtype=np.double)
RMSE_r_3 = np.zeros(len_snr, dtype=np.double)

G_net = ConvNeXt(in_chans=6, depths=[3, 3, 27, 3], dims=[256,512,1024,2048], num_classes=2)
device_ids = [0]
G = nn.DataParallel(G_net, device_ids=device_ids).cuda()

G.load_state_dict(torch.load('./model_pt/convnext_1.pt'))

t0 = time.time()

for i in range(len_snr):
    SNR_db = SNR[i]
    # SNR_linear = 10 ** (SNR[i] / 10)

    for i_iter in range(N_iter):

        elapsed_time = time.time() - t0
        print(f'  iteration:[{i_iter + 1}/{N_iter}] |  SNR:[{i + 1}/{len_snr}]  | run {elapsed_time:.4f} s')

        r = Rmin + np.random.rand() * (Rmax - Rmin)
        theta = user_theta_min + np.random.rand() * (user_theta_max - user_theta_min)
        # r = 15
        # theta = 20 / 180 * np.pi
        h = near_field_channel(Nt, d, fc, B, M, r, theta)
        p = np.zeros(Nt)
        n_k_min = max(1, int(floor(np.random.rand() * 4)))
        n_k_max = int(ceil(n_k_min + np.random.rand() * (4 - n_k_min)))
        # n_k_min = 2
        # n_k_max = 3
        p[(n_k_min * 128 - 128) : (n_k_max * 128 )] = 1
        for m in range(M):
            h[m, :] *= p

        theta_hat, y_n_1 = CBS_theta_high(h, Nt, M, B, d, f, SNR_db, Rmin, Rmax, user_theta_max, user_theta_min)

        r0 = Rmax
        theta0 = theta_hat
        rc = Rmin
        thetac = theta_hat
        _, r_M = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
        y_n_3, y_1 = generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db)
        r_hat = CBS_r(y_n_3, M, r_M)

        x = pre_reshape_x(y_n_1, y_n_3, theta_hat)
        x_tensor = torch.from_numpy(x)
        x_tensor = Variable(x_tensor.cuda())  # H
        x_tensor_float32 = x_tensor.float()
        fake_img = G(x_tensor_float32)
        r_theta_hat = fake_img.cpu().detach().numpy()
        theta_hat_1 = r_theta_hat[:,1]/1000
        r_hat_1 = r_theta_hat[:,0]/10

        r0 = Rmax
        theta0 = theta_hat_1
        rc = Rmin
        thetac = theta_hat_1
        _, r_M = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
        y_n_3, _ = generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db)
        r_hat_2 = CBS_r(y_n_3, M, r_M)

        r_hat_3 = CBS_r_high(h, theta_hat_1, Nt, M, B, d, f, Rmin, Rmax, SNR_db)


        RMSE_theta[i] += (theta_hat - theta) ** 2 / N_iter
        RMSE_r[i] += (r_hat - r) ** 2 / N_iter
        RMSE_theta_1[i] += (theta_hat_1[0] - theta) ** 2 / N_iter
        RMSE_r_1[i] += (r_hat_1[0] - r) ** 2 / N_iter
        RMSE_r_2[i] += (r_hat_2 - r) ** 2 / N_iter
        RMSE_r_3[i] += (r_hat_3[0] - r) ** 2 / N_iter


for i in range(len_snr):
    RMSE_theta[i] = np.sqrt(RMSE_theta[i])
    RMSE_r[i] = np.sqrt(RMSE_r[i])
    RMSE_theta_1[i] = np.sqrt(RMSE_theta_1[i])
    RMSE_r_1[i] = np.sqrt(RMSE_r_1[i])
    RMSE_r_2[i] = np.sqrt(RMSE_r_2[i])
    RMSE_r_3[i] = np.sqrt(RMSE_r_3[i])


print('RMSE_theta:  ', RMSE_theta)
print('RMSE_r:  ', RMSE_r)
print('RMSE_theta_1:  ', RMSE_theta_1)
print('RMSE_r_1:  ', RMSE_r_1)
print('RMSE_r_2:  ', RMSE_r_2)
print('RMSE_r_3:  ', RMSE_r_3)

# np.savetxt('./model_pt/mrdn_RMSE_theta.csv', RMSE_theta, delimiter=',')
# np.savetxt('./model_pt/mrdn_RMSE_r.csv', RMSE_r, delimiter=',')
# np.savetxt('./model_pt/mrdn_RMSE_theta_1.csv', RMSE_theta_1, delimiter=',')
# np.savetxt('./model_pt/mrdn_RMSE_r_1.csv', RMSE_r_1, delimiter=',')

plt.figure(facecolor='white')
plt.plot(SNR, RMSE_theta, 'k-^', linewidth=2, label='2 times beam sweeping')
plt.plot(SNR, RMSE_theta_1, 'r-^', linewidth=2, label='5 times beam sweeping')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE$_\\theta$ (rad)')
plt.grid(True)      # Enable grid
plt.box(on=True)    # Enable box
plt.legend()       # Show the legend
plt.show()          # Display the plot

plt.figure(facecolor='white')
plt.plot(SNR, RMSE_r, 'k-^', linewidth=2, label='2 times beam sweeping')
plt.plot(SNR, RMSE_r_1, 'r-^', linewidth=2, label='5 times beam sweeping')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE$_r$ (m)')
plt.grid(True)      # Enable grid
plt.box(on=True)    # Enable box
plt.legend()       # Show the legend
plt.show()          # Display the plot
