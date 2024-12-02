import numpy as np
import torch
import torch.nn as nn
import os
import time
from modelss.convnexts.convnext import ConvNeXt
from functional import near_field_channel, Beam_Squint_trajectory, CBS_theta_high, generate_beam, CBS_r
from numpy import floor, ceil
from joblib import Parallel, delayed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

EPOCH = 1

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

N_iter = 20

Rmin = 5
Rmax = 50
user_theta_max = 60 / 180 * np.pi
user_theta_min = -60 / 180 * np.pi

RMSE_theta = np.zeros(len_snr, dtype=np.double)
RMSE_r = np.zeros(len_snr, dtype=np.double)

G_net = ConvNeXt(in_chans=6, depths=[3, 3, 27, 3], dims=[256,512,1024,2048], num_classes=2)
device_ids = [0]
G = nn.DataParallel(G_net, device_ids=device_ids).cuda()
G.load_state_dict(torch.load('./model_pt/convnext_1.pt'))

t0 = time.time()

def process_iteration(i, SNR_db):
    RMSE_theta_i = 0
    RMSE_r_i = 0
    for i_iter in range(N_iter):
        elapsed_time = time.time() - t0
        print(f' iteration:[{i_iter + 1}/{N_iter}] | SNR:[{i + 1}/{len_snr}] | run {elapsed_time:.4f} s')


        r = Rmin + np.random.rand() * (Rmax - Rmin)
        theta = user_theta_min + np.random.rand() * (user_theta_max - user_theta_min)
        h = near_field_channel(Nt, d, fc, B, M, r, theta)
        p = np.zeros(Nt)
        n_k_min = max(1, int(floor(np.random.rand() * 4)))
        n_k_max = int(ceil(n_k_min + np.random.rand() * (4 - n_k_min)))
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

        RMSE_theta_i += (theta_hat - theta) ** 2 / N_iter
        RMSE_r_i += (r_hat - r) ** 2 / N_iter

    return (RMSE_theta_i, RMSE_r_i)

results = Parallel(n_jobs=-1)(delayed(process_iteration)(i, SNR_db) for i, SNR_db in enumerate(SNR))

for i, result in enumerate(results):
    RMSE_theta[i] = np.sqrt(np.mean([res[0] for res in result]))
    RMSE_r[i] = np.sqrt(np.mean([res[1] for res in result]))

print('RMSE_theta: ', RMSE_theta)
print('RMSE_r: ', RMSE_r)
