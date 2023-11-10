import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pylab as pl

plt.rc('font', family='serif', serif='Computer Modern Roman', size=12)
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (5,4.5)
model_type='GPT2'
context_size=40
SNR_dB=30.0
pretrain=False

bits=4
i=0
n_tasks=4096
N_te=1000
SNR_to_test = np.linspace(0, 30, 16)

n_tasks=4096
ICL=np.load('./logs_test_bits_'+str(bits)+'/MSE_' + str(model_type) + '_SNR_AWARE_' + str(SNR_to_test) + '_tasks_' + str(n_tasks) + '_pretrain_' + str(pretrain)+'.npy', allow_pickle=True)
plt.errorbar(ICL[0],[d[-1] for d in ICL[1]],yerr=[2*d[-1] for d in ICL[2]]/np.sqrt(N_te),marker='s',color='tab:green',label='ICL',linestyle='-')

col = pl.cm.winter(np.linspace(0,0.7,3))

SNR=0
n_tasks=4096
ICL=np.load('./logs_test_bits_'+str(bits)+'/MSE_' + str(model_type) + '_SNR_' + str(SNR_to_test) + '_tasks_' + str(n_tasks) + '_SNR_' + str(SNR)+'.npy', allow_pickle=True)
plt.errorbar(ICL[0],[d[-1] for d in ICL[1]],yerr=[2*d[-1] for d in ICL[2]]/np.sqrt(N_te),marker='x',color=col[0],label='ICL trained at SNR 0 dB',linestyle='-.')

SNR=30
n_tasks=4096
ICL=np.load('./logs_test_bits_'+str(bits)+'/MSE_' + str(model_type) + '_SNR_' + str(SNR_to_test) + '_tasks_' + str(n_tasks) + '_SNR_' + str(SNR)+'.npy', allow_pickle=True)
plt.errorbar(ICL[0],[d[-1] for d in ICL[1]],yerr=[2*d[-1] for d in ICL[2]]/np.sqrt(N_te),marker='d',color=col[2],label='ICL trained at SNR 30 dB',linestyle='--')

n_tasks=4096
DMMSE=np.load('./logs_test_bits_'+str(bits)+'/MSE_dMMSE_Aware_SNR_'+str(SNR_to_test)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy', allow_pickle=True)
plt.errorbar(DMMSE[0],[d[-1] for d in DMMSE[1]],yerr= [d[-1] for d in DMMSE[2]]/np.sqrt(N_te),marker='^',color='k',label='MMSE, known task',linestyle='-')
n_tasks=4096
DMMSE_Lin=np.load('./logs_test_bits_'+str(bits)+'/MSE_Linear_dMMSE_Aware_SNR_'+str(SNR_to_test)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy', allow_pickle=True)
plt.errorbar(DMMSE_Lin[0],[d[-1] for d in DMMSE_Lin[1]],yerr=[d[-1] for d in DMMSE_Lin[2]]/np.sqrt(N_te),marker='o',color='tab:red',label='LMMSE, known task',linestyle='-')


i=i+1
plt.xlabel('SNR [dB]')
plt.ylabel('MSE')
plt.xlim([0,30])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./Results/MSE_vs_SNR.pdf')
plt.show()
plt.clf()
