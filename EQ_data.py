import numpy as np
import matplotlib.pyplot as plt

def quantize(X,n_bits,l_min,l_max):
    q = 1 / (2 ** n_bits)
    X_real=np.real(X)
    X_real[X_real<l_min]=l_min+0.001
    X_real[X_real>l_max]=l_max-0.001
    X_real=(X_real+l_min)/(l_max-l_min)
    X_real_q = (q * (np.floor((X_real) / q)+0.5))*(l_max-l_min)-l_min
    X_imag=np.imag(X)
    X_imag[X_imag<l_min]=l_min+0.001
    X_imag[X_imag>l_max]=l_max-0.001
    X_imag = (X_imag + l_min) / (l_max - l_min)
    X_imag_q =(q * (np.floor((X_imag) / q)+0.5))*(l_max-l_min)-l_min
    return X_real_q+1j*X_imag_q

def batch_data_gen_quantized(batch_size,n_tasks, args,seed_task=0,seed_example=0,H=None,SNR_in_dB_all=None,l_min=-4,l_max=4):
    np.random.seed(seed_task)
    H = (np.random.randn(n_tasks, args.num_ant, args.num_ant) + 1j * np.random.randn(n_tasks, args.num_ant, args.num_ant)) / np.sqrt(2)
    SNR_in_dB_all = np.random.rand(n_tasks) * (args.SNR_dB_max-args.SNR_dB_min) + args.SNR_dB_min
    X,Y,SNR=[],[],[]
    np.random.seed(seed_example)
    Hs=[]
    for ii in range(batch_size):
        id_rand=int(np.random.randint(0,n_tasks))
        H_eval = H[id_rand]
        Hs.append(H_eval)
        SNR_in_dB=SNR_in_dB_all[id_rand]
        noiseVar = 10 ** (-SNR_in_dB / 10)
        x = ((np.random.randint(2, size=(args.num_ant, args.prompt_seq_length)) - 0.5) * 2 + 1j * (np.random.randint(2, size=(args.num_ant, args.prompt_seq_length)) - 0.5) * 2) / np.sqrt(2)
        n = np.sqrt(noiseVar) * (np.random.randn(args.num_ant, args.prompt_seq_length) + 1j * np.random.randn(args.num_ant, args.prompt_seq_length))
        y = H_eval @ x + n
        y_q = quantize(y, args.bits, l_min, l_max)
        Y.append(np.transpose(y_q))
        X.append(np.transpose(x))
        SNR.append(np.transpose(x*0)+SNR_in_dB/30+1j*(np.transpose(x*0)+SNR_in_dB/30))
    X=np.stack(X)
    Y=np.stack(Y)
    return X,Y,Hs,SNR
