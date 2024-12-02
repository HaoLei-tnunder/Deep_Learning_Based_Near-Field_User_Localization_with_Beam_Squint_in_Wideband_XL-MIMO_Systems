from scipy.fftpack import fft2, ifft2, fft, ifft
import torch
import numpy as np
import math
from torch.autograd import Variable

def pre_reshape(x_train, x_train1, img_height, img_width):
    x, y, z = x_train.shape   # n  样本数
    H_reshape = np.zeros([x ,2 , img_height, img_width], dtype=np.float32)
    for i in range(x):
        H_reshape[i,0,:] += x_train[i,:,:]
        H_reshape[i,1,0] += x_train1[i,0]*10
        H_reshape[i,1,1] += x_train1[i,1]*1000
    return H_reshape     

def pre_reshape_1(x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, img_height, img_width):
    y = x_train.shape[2]    # n  样本数
    # x = int(img_height/2)
    H_reshape = np.zeros([y , 9 , img_height, img_width], dtype=np.float32)
    # H_reshape = np.ones([y ,4 , img_height, img_width], dtype=np.float32)
    for i in range(y):
        H_reshape[i,0,:,:] += x_train[:,:,i]
        H_reshape[i,1,:,:] += x_train1[:,:,i]
        H_reshape[i,2,:,:] += x_train2[:,:,i]
        H_reshape[i,3,:,:] += x_train3[:,:,i]
        H_reshape[i,4,:,:] += x_train4[:,:,i]
        H_reshape[i,5,:,:] += x_train5[:,:,i]
        H_reshape[i,6,:,:] += x_train6[:,:,i]
        H_reshape[i,7,:,:] += x_train7[:,:,i]

        H_reshape[i,8,0,0] += x_train8[i,0]*10
        H_reshape[i,8,0,1] += x_train8[i,1]*1000
    return H_reshape

def pre_reshape_11(x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7, x_train8, img_height, img_width):
    y = x_train.shape[2]    # n  样本数
    # x = int(img_height/2)
    H_reshape = np.zeros([y , 9 , img_height, img_width], dtype=np.float32)
    # H_reshape = np.ones([y ,4 , img_height, img_width], dtype=np.float32)
    for i in range(y):
        H_reshape[i,0,:,:] += x_train[:,:,i]
        H_reshape[i,1,:,:] += x_train1[:,:,i]
        H_reshape[i,2,:,:] += x_train2[:,:,i]
        H_reshape[i,3,:,:] += x_train3[:,:,i]
        H_reshape[i,4,:,:] += x_train4[:,:,i]
        H_reshape[i,5,:,:] += x_train5[:,:,i]
        H_reshape[i,6,:,:] += x_train6[:,:,i]
        H_reshape[i,7,:,:] += x_train7[:,:,i]

        H_reshape[i,8,0,0] += x_train8[i,0]*1
        H_reshape[i,8,0,1] += x_train8[i,1]*10
    return H_reshape

def compute_MSE_r_train( r_theta_hat , r_theta):
    # RMSE_2D = np.mean( np.square(r_theta_hat[:,0] - r_theta[:,0]) + np.square(r_theta_hat[:,1] - r_theta[:,1]) )
    # RMSE_2D = math.sqrt(RMSE_2D)
    RMSE_r = np.mean(np.square(r_theta_hat[:,0]/10 - r_theta[:,0]/10))
    RMSE_r = math.sqrt(RMSE_r)
    return RMSE_r

def compute_MSE_theta_train( r_theta_hat , r_theta):
    RMSE_theta = np.mean(np.square(r_theta_hat[:,1]/1000 - r_theta[:,1]/1000))
    RMSE_theta = math.sqrt(RMSE_theta)
    return RMSE_theta

def compute_MSE_r_test( r_theta_hat , r_theta):
    # RMSE_2D = np.mean( np.square(r_theta_hat[:,0] - r_theta[:,0]) + np.square(r_theta_hat[:,1] - r_theta[:,1]) )
    # RMSE_2D = math.sqrt(RMSE_2D)
    RMSE_r = np.mean(np.square(r_theta_hat[:,0]/10 - r_theta[:,0]/10))
    # RMSE_r = math.sqrt(RMSE_r)
    return RMSE_r

def compute_MSE_theta_test( r_theta_hat , r_theta):
    RMSE_theta = np.mean(np.square(r_theta_hat[:,1]/1000 - r_theta[:,1]/1000))
    # RMSE_theta = math.sqrt(RMSE_theta)
    return RMSE_theta

def compute_MSE_r_test1( r_theta_hat , r_theta):
    # RMSE_2D = np.mean( np.square(r_theta_hat[:,0] - r_theta[:,0]) + np.square(r_theta_hat[:,1] - r_theta[:,1]) )
    # RMSE_2D = math.sqrt(RMSE_2D)
    RMSE_r = np.mean(np.square(r_theta_hat[:,0]/1 - r_theta[:,0]/1))
    # RMSE_r = math.sqrt(RMSE_r)
    return RMSE_r

def compute_MSE_theta_test1( r_theta_hat , r_theta):
    RMSE_theta = np.mean(np.square(r_theta_hat[:,1]/10 - r_theta[:,1]/10))
    # RMSE_theta = math.sqrt(RMSE_theta)
    return RMSE_theta

def compute_MSE_r_train_GPU( r_theta_hat , r_theta):
    RMSE_r = torch.mean(torch.square(r_theta_hat[:,0]/10 - r_theta[:,0]/10))
    RMSE_r = torch.sqrt(RMSE_r)
    return RMSE_r

def compute_MSE_theta_train_GPU( r_theta_hat , r_theta):
    RMSE_theta = torch.mean(torch.square(r_theta_hat[:,1]/1000 - r_theta[:,1]/1000))
    RMSE_theta = torch.sqrt(RMSE_theta)
    return RMSE_theta

def compute_MSE_2D( r_theta_hat , r_theta):
    xy_hat = np.array(  [  np.multiply( r_theta_hat[:,0]/10 , np.cos(r_theta_hat[:,1]/1000) )  ,   np.multiply( r_theta_hat[:,0]/10 , np.sin(r_theta_hat[:,1]/1000) ) ]  ).T
    xy = np.array([np.multiply(r_theta[:, 0]/10, np.cos(r_theta[:, 1]/1000)), np.multiply(r_theta[:, 0]/10, np.sin(r_theta[:, 1]/1000))]).T
    RMSE_2D =  np.square(xy_hat[:,0] - xy[:,0]) + np.square(xy_hat[:,1] - xy[:,1])
    # RMSE_2D = math.sqrt(RMSE_2D)
    return RMSE_2D

def compute_MSE_2D1( r_theta_hat , r_theta):
    xy_hat = np.array(  [  np.multiply( r_theta_hat[:,0]/1 , np.cos(r_theta_hat[:,1]/10) )  ,   np.multiply( r_theta_hat[:,0]/1 , np.sin(r_theta_hat[:,1]/10) ) ]  ).T
    xy = np.array([np.multiply(r_theta[:, 0]/1, np.cos(r_theta[:, 1]/10)), np.multiply(r_theta[:, 0]/1, np.sin(r_theta[:, 1]/10))]).T
    RMSE_2D =  np.square(xy_hat[:,0] - xy[:,0]) + np.square(xy_hat[:,1] - xy[:,1])
    # RMSE_2D = math.sqrt(RMSE_2D)
    return RMSE_2D

def fft_shrink(H_shape, img_height, img_width):
    x = len(H_shape)    # n
    # H_real = np.zeros([x, img_height, img_width], dtype=np.complex128)
    # H_imag = np.zeros([x, img_height, img_width], dtype=np.complex128)
    # H_real = np.zeros([x, img_height, img_width], dtype=complex)
    # H_imag = np.zeros([x, img_height, img_width], dtype=complex)
    H_real_1 = np.zeros([x, img_height, img_width], dtype=complex)
    H_imag_1 = np.zeros([x, img_height, img_width], dtype=complex)    
    # for m in range(x):
    #     H_real[m, :, :] = H_real[m, :, :] + H_shape[m, :, 0:img_width]
    #     H_imag[m, :, :] = H_real[m, :, :] + 1j*H_shape[m, :, img_width:img_width*2]
    H_real_1 = H_real_1 + H_shape[:, :, 0:img_width]
    H_imag_1 = H_real_1 + 1j*H_shape[:, :, img_width:img_width*2]    
    return H_imag_1

def fft_shrink_tensor(H_shape, img_height, img_width, device):
    x = len(H_shape)    # n
    H_real = torch.zeros([x, img_height, img_width], dtype=complex)
    H_imag = torch.zeros([x, img_height, img_width], dtype=complex)
    H_imag = H_imag.to(device)
    H_real = H_real.to(device)
    for m in range(x):
        H_real[m, :, :] = H_real[m, :, :] + H_shape[m, :, 0:img_width]
        H_imag[m, :, :] = H_real[m, :, :] + 1j*H_shape[m, :, img_width:img_width*2]
    return H_imag

def add_noise(input, SNR):
    x, y = input.shape
    SNR_linear = 10**(SNR/10)
    Noise_map = np.zeros([ 32, 16], dtype=complex)
    Noise_map_real = np.zeros([1, 1], dtype=complex)
    Noise_map_imag = np.zeros([1, 1], dtype=complex)
    power = np.zeros([ 32, 16], dtype=complex)
    for i in range(x):
        for ii in range(y):
            power[i,ii] = abs(input[i,ii]) ** 2
            sigma2 = power[i,ii] / SNR_linear
            n_l = math.sqrt(sigma2)
            Noise_map_real = n_l * math.sqrt(1 / 2) * np.random.randn( 1, 1)
            Noise_map_imag = 1j * n_l * math.sqrt(1 / 2) * np.random.randn( 1, 1)
            Noise_map[i,ii] = Noise_map_real + Noise_map_imag
    return Noise_map, power

def add_noise_1(input, SNR):
    SNR_linear = 10**(SNR/10)
    power = np.abs(input) ** 2
    sigma2 = power / SNR_linear
    n_l = np.sqrt(sigma2)
    Noise_map_real = n_l * math.sqrt(1 / 2) * np.random.randn(*input.shape)
    Noise_map_imag = 1j * n_l * math.sqrt(1 / 2) * np.random.randn(*input.shape)
    Noise_map = Noise_map_real + Noise_map_imag
    return Noise_map

def add_noise_improve(input, SNRlow, SNRhign):
    NN = len(input)
    noise = np.zeros([NN, 64, 32], dtype=complex)
    # noise_1 = np.zeros([NN, 32, 16], dtype=complex)
    SNR_divide = np.random.uniform(SNRlow, SNRhign, size=[NN])                      #均匀分布[low,high)中随机采样
    for nx in range(NN):                                                            #0~NN-1
        # noise[nx, :, :], power = add_noise(input[nx, :, :], SNR_divide[nx])
        noise[nx, :, :] = add_noise_1(input[nx, :, :], SNR_divide[nx])
    return noise

def real_imag_stack(H):
    x = len(H)    # n
    H_real = np.zeros([x, 32, 16])           #32
    H_imag = np.zeros([x, 32, 16])
    H_out = np.zeros([x, 32, 32])
    # H_real = np.zeros([x, 32, 16], dtype=np.double)           #32
    # H_imag = np.zeros([x, 32, 16], dtype=np.double)
    # H_out = np.zeros([x, 32, 32], dtype=np.double)
    for i in range(x):
        H_real[i, :] = H[i, :].real
        H_imag[i, :] = H[i, :].imag
        H_out[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_out

def real_imag_stack_1(H):
    x = len(H)    # n
    H_abs = np.zeros([x, 32, 16])           #32
    H_phase = np.zeros([x, 32, 16])
    H_out = np.zeros([x, 32, 32])
    H_out[:, :, :16] = np.abs(H)
    H_out[:, :, 16:] = np.angle(H)
    # for i in range(x):
    #     H_real[i, :] = H[i, :].real
    #     H_imag[i, :] = H[i, :].imag
    #     H_out[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_out

def tensor_reshape(input1):
    x = len(input1)
    s = input1.shape[1]
    input = torch.from_numpy(input1)
    H = torch.zeros([x, 1, s, 32])                      #64   64
    # H = torch.zeros([x, 1, s, 32], dtype=torch.double)                      #64   64
    for i in range(x):
        H[i, 0, :, :] = input[i, 0:s, 0:32]
    return H

def ifft_tensor(H_shape):
    x = len(H_shape)    # n
    H_reshape_fft = np.zeros([x, 32, 16], dtype=complex)                #64 32
    # H_real = np.zeros([x, 128, 1])
    # H_imag = np.zeros([x, 128, 1])
    # H_get = np.zeros([x, 128, 2])
    for i in range(x):
        H_reshape_fft[i, :, :] = ifft2(H_shape[i, :, :])
        # H_real[i, :] = H_reshape_fft[i, :].real
        # H_imag[i, :] = H_reshape_fft[i, :].imag
        # H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_reshape_fft

def compute_NMSE(H_hat, H):
    H1 = np.reshape(H_hat, (len(H_hat), -1))
    H_pre1 = np.reshape(H, (len(H), -1))
    power = np.sum(abs(H_pre1) ** 2, axis=1)
    mse = np.sum(abs(  H_pre1 - H1) ** 2, axis=1)
    NMSE = 10 * math.log10(np.mean(mse / power))
    return NMSE

def near_field_channel(Nt, d, fc, B, M, r, theta):
    H = np.zeros((M, Nt), dtype=complex)
    c = 3e8
    f = np.zeros(M)
    nn = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    r0 = np.sqrt(r ** 2 + (nn * d) ** 2 - 2 * r * nn * d * np.sin(theta))
    for m in range(M):
        f[m] = fc + B / 2 * (2 * (m + 1) / (M - 1) - 1)
        H[m, :] = (1 / r) * np.exp(-1j * 2 * np.pi * f[m] * r0 / c)
    # beta = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    # H = beta * H
    return H

def generate_beamfoucing_vector(Nt, M, B, d, f, r0, theta0, rc, thetac):
    c = 3e8
    w = np.zeros((Nt, M), dtype=complex)
    nn = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    rr = np.sqrt(r0 ** 2 + (nn * d) ** 2 - 2 * r0 * nn * d * np.sin(theta0))
    rrc = np.sqrt(rc ** 2 + (nn * d) ** 2 - 2 * rc * nn * d * np.sin(thetac))
    phi = np.zeros(Nt)
    t = np.zeros(Nt)
    for n in range(Nt):
        for m in range(M):
            phi[n] = f[0] / c * rr[n]
            t[n] = f[M - 1] / B / c * rrc[n] - phi[n] / B
            w[n, m] = 1 / np.sqrt(Nt) * np.exp(-1j * 2 * np.pi * phi[n]) * np.exp(-1j * 2 * np.pi * (f[m] - f[0]) * t[n])
    return w

def Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc):
    theta_M = np.zeros(M)
    r_M = np.zeros(M)
    for m in range(M):
        theta_M[m] = np.arcsin((B - (f[m] - f[0])) * f[0] / B / f[m] * np.sin(theta0) + (B + f[0]) * (f[m] - f[0]) / B / f[m] * np.sin(thetac))
        r_M[m] = 1 / (1 / r0 * (B - (f[m] - f[0])) * f[0] / B / f[m] * np.cos(theta0)**2 / np.cos(theta_M[m])**2 + 1 / rc * (B + f[0]) * (f[m] - f[0]) / B / f[m] * np.cos(thetac)**2 / np.cos(theta_M[m])**2)
    return theta_M, r_M

def CBS_theta(y, M, theta_M):
    # y_abs_1 = np.zeros(M, dtype=complex)
    # for m in range(M):
    #     y_abs_1[m] = np.abs(y[m])
    y_abs_1 = np.abs(y)
    y_abs_1 = y_abs_1 / np.max(y_abs_1[:])
    theta_index = np.argmax(y_abs_1[:])
    theta_hat = theta_M[theta_index]
    return theta_hat

def CBS_r(y, M, r_M):
    y_abs_1 = np.zeros(M, dtype=complex)
    for m in range(M):
        y_abs_1[m] = np.abs(y[m])
    y_abs_1 = y_abs_1 / np.max(y_abs_1[:])
    r_index = np.argmax(y_abs_1[:])
    r_hat = r_M[r_index]
    return r_hat

def generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db):
    w = generate_beamfoucing_vector(Nt, M, B, d, f, r0, theta0, rc, thetac)
    y = np.zeros(M, dtype=complex)
    y_n = np.zeros(M, dtype=complex)
    for m in range(M):
        y[m] = np.conj(h[m, :]) @ w[:, m]
    y = np.reshape(y, [1,64,32], order='F')
    noise = add_noise_improve(y, SNR_db, SNR_db)
    noise = np.reshape(noise, [M], order='F')
    y = np.reshape(y, [M], order='F')
    y_n = y + noise
        # sigma2 = np.linalg.norm(y[m]) ** 2 / SNR_linear
        # noise = np.sqrt(sigma2) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        # y_n[m] += noise
    return y_n, y

def CBS_theta_high(h, Nt, M, B, d, f, SNR_db, Rmin, Rmax, user_theta_max, user_theta_min):
    T = 4     #  angle - 1
    theta_hat = np.zeros(T, dtype=np.float64)
    y_n_1 = np.zeros([M, T], dtype=complex)
    r0 = Rmin  # start
    theta0 = user_theta_max
    rc = Rmin  # end
    thetac = user_theta_min
    theta_M, _ = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
    w = generate_beamfoucing_vector(Nt, M, B, d, f, r0, theta0, rc, thetac)
    y = np.zeros(M, dtype=complex)
    y_n = np.zeros(M, dtype=complex)
    for m in range(M):
        y[m] = np.conj(h[m, :]) @ w[:, m]

    for i in range(T):
        y = np.reshape(y, [1, 64, 32], order='F')
        noise = add_noise_improve(y, SNR_db, SNR_db)
        noise = np.reshape(noise, [M], order='F')
        y = np.reshape(y, [M], order='F')
        y_n = y + noise
        y_n_1[:, i] = y_n
        # y_n_1[:, i] = y
        y_abs_1 = np.zeros(M, dtype=complex)
        y_abs_1 = np.abs(y_n)
        # y_abs_1 = np.abs(y)
        y_abs_1 = y_abs_1 / np.max(y_abs_1[:])
        theta_index = np.argmax(y_abs_1[:])
        theta_hat[i] = theta_M[theta_index]

    theta_hat_avg = np.sum(theta_hat)/T
    #
    # T = 2     #  angle - 2
    # theta_hat = np.zeros(T, dtype=np.float64)
    # y_n_2 = np.zeros([M, T], dtype=complex)
    # for i in range(T):
    #     r0 = Rmax    # start
    #     theta0 = theta_hat_avg + 2 / 180 * np.pi
    #     rc = Rmax   # end
    #     thetac = theta_hat_avg - 2 / 180 * np.pi
    #     theta_M, _ = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
    #     y_n, y = generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db)
    #     y_n_2[:, i] = y_n
    #     # y_n_2[:, i] = y
    #     y_abs_1 = np.zeros(M, dtype=complex)
    #     y_abs_1 = np.abs(y_n)
    #     y_abs_1 = y_abs_1 / np.max(y_abs_1[:])
    #     theta_index = np.argmax(y_abs_1[:])
    #     theta_hat[i] = theta_M[theta_index]
    # theta_hat_avg = np.sum(theta_hat)/T

    return theta_hat_avg, y_n_1



def CBS_r_high1(h, Nt, M, B, d, f, SNR_db, Rmin, Rmax, theta_hat):
    r0 = Rmax
    theta0 = theta_hat
    rc = Rmin
    thetac = theta_hat
    _, r_M = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
    T = 10
    r_hat = np.zeros(T, dtype=np.float64)
    for i in range(T):
        y_n_1, _ = generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db)
        r_hat[i] = CBS_r(y_n_1, M, r_M)
    r_hat_avg = np.sum(r_hat)/T
    return r_hat_avg

def denosie(y_n_1, G):
    H_n_2 = np.reshape(y_n_1, [1, 32, 16], order='F')
    H_n_fft_r_i = real_imag_stack(H_n_2)  # 64  64
    H_n_fft_stack = tensor_reshape(H_n_fft_r_i)
    H_n_fft_train = Variable(H_n_fft_stack.cuda())
    fake_img = G(H_n_fft_train)
    h_fft_pre = torch.zeros([1, 32, 32])
    for i_num in range(1):
        h_fft_pre[i_num, :, :] = H_n_fft_train[i_num, :, :] - fake_img[i_num, :, :]
    ssx = h_fft_pre.detach().numpy()
    ssx_i = np.zeros([1, 32, 32], dtype=complex)  # 64 X 64
    ssx_i = ssx + ssx_i
    H_fft_pre_last = fft_shrink(ssx_i, 32, 16)
    y_hat_2 = np.reshape(H_fft_pre_last, [512], order='F')
    return y_hat_2

def CBS_r_high_1(h, Nt, M, B, d, f, SNR_db, Rmin, Rmax, theta_hat, G):
    r0 = Rmax
    theta0 = theta_hat
    rc = Rmin
    thetac = theta_hat
    _, r_M = Beam_Squint_trajectory(B, M, f, theta0, r0, thetac, rc)
    T = 5
    r_hat = np.zeros(T, dtype=np.float64)
    for i in range(T):
        y_n_1, _ = generate_beam(h, Nt, M, B, d, f, r0, theta0, rc, thetac, SNR_db)
        y_hat = denosie(y_n_1, G)
        r_hat[i] = CBS_r(y_hat, M, r_M)
    r_hat_avg = np.sum(r_hat)/T
    return r_hat_avg


def pre_reshape_x(y_n_1, y_n_3, theta_hat):

    x = np.zeros([1, 6, 64, 32], dtype=complex)
    x_out = np.zeros([1, 6, 64, 64], dtype=np.double)
    x[0,0,:,:] = np.reshape(y_n_1[:,0], [1,1,64,32], order='F')
    x[0,1,:,:] = np.reshape(y_n_1[:,1], [1,1,64,32], order='F')
    x[0,2,:,:] = np.reshape(y_n_1[:,2], [1,1,64,32], order='F')
    x[0,3,:,:] = np.reshape(y_n_1[:,3], [1,1,64,32], order='F')
    x[0,4,:,:] = np.reshape(y_n_3, [1,1,64,32], order='F')

    x_out[0, :5, :, :32] = np.abs(x[0, :5, :, :])
    x_out[0, :5, :, 32:] = np.angle(x[0, :5, :, :])
    x_out[0, 5, :, :] = theta_hat

    return x_out

def generate_beamfoucing_vector_1(Nt, M, B, d, f, rc, thetac):
    c = 3e8
    w = np.zeros((Nt, M), dtype=complex)
    nn = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    rcc = np.zeros((8, Nt))

    for i in range(8):
        rcc[i, :] = np.sqrt(rc[i] ** 2 + (nn * d) ** 2 - 2 * rc[i] * nn * d * np.sin(thetac))

    phi = np.zeros((Nt, 1))
    t = np.zeros((Nt, 1))

    for n in range(Nt):
        for i in range(8):
            range_start = (i - 1) * 256 + 1
            range_end = i * 256

            phi[n] = f[range_start - 1] / c * rcc[i, n]
            t[n] = f[range_end - 1] / (f[256 - 1] - f[0]) / c * rcc[i, n] - phi[n] / (f[256 - 1] - f[0])

            w[n, range_start - 1:range_end] = 1 / np.sqrt(Nt) * np.exp(-1j * 2 * np.pi * phi[n]) * np.exp(
                -1j * 2 * np.pi * (f[range_start - 1:range_end] - f[range_start - 1]) * t[n])
    return w

def CBS_r_high(h, theta_hat_1, Nt, M, B, d, f, Rmin, Rmax, SNR_db):
    rcc = np.zeros((8, 1))
    thetacc = theta_hat_1
    SNR_linear = 10 ** (SNR_db / 10)
    r_start = Rmin
    r_end = Rmax
    xx = 0

    while (r_end - r_start) >= 0.5 and xx <= 5:
        xx += 1
        y2 = np.zeros(M, dtype=complex)
        y22 = np.zeros((8, 1))

        for i in range(8):
            rcc[i] = r_start + (i - 1) * (r_end - r_start) / 7

        w1 = generate_beamfoucing_vector_1(Nt, M, B, d, f, rcc, thetacc)

        for m in range(M):
            y2[m] = np.conj(h[m, :]) @ w1[:, m]

        power = np.abs(y2) ** 2
        sigma2 = power / SNR_linear

        n_l = np.sqrt(sigma2)
        Noise_map_real = n_l * math.sqrt(1 / 2) * np.random.randn(*y2.shape)
        Noise_map_imag = 1j * n_l * math.sqrt(1 / 2) * np.random.randn(*y2.shape)
        Noise_map = Noise_map_real + Noise_map_imag

        y2 = y2 + Noise_map

        y_abs_2 = np.abs(y2)

        for i in range(8):
            y22[i] = np.sum(y_abs_2[((i - 1) * 256 + 1):(i * 256)])

        sorted_indices = np.argsort(y22, axis=0)[::-1]
        top_three_indices = sorted_indices[:4].flatten()
        sorted_top_three_indices = np.sort(top_three_indices)

        r_start = rcc[sorted_top_three_indices[0]]
        r_end = rcc[sorted_top_three_indices[3]]

    index = np.argmax(y22)

    r_hat_avg = rcc[index]

    return r_hat_avg
