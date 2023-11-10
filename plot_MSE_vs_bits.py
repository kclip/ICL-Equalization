import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pylab as pl

plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (5,4.5)
model_type='GPT2'
context_size=40
SNR_dB=10
pretrain=False
bits=4
i=0
n_tasks=4096
N_te=1000
BITS_TO_TEST=[1,2,3,4,5,6,7,8]

fig, ax = plt.subplots()
ICL=np.load('./logs_test_bits_'+str(BITS_TO_TEST)+'/MSE_' + str(model_type) + '_SNR_' + str(SNR_dB) + '_tasks_' + str(n_tasks) + '_pretrain_' + str(pretrain)+'sigmoid.npy', allow_pickle=True)
ax.errorbar(ICL[0],[d[-1] for d in ICL[1]],yerr=[2*d[-1] for d in ICL[2]]/np.sqrt(N_te),marker='x',color='tab:green',label='ICL',linestyle='-')

MMSE_Lin=np.load('./logs_test_bits_'+str(BITS_TO_TEST)+'/MSE_Linear_dMMSE_Aware_SNR_'+str(SNR_dB)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy', allow_pickle=True)
ax.errorbar(MMSE_Lin[0],[d[-1] for d in MMSE_Lin[1]],yerr=[2*d[-1] for d in MMSE_Lin[2]]/np.sqrt(N_te),marker='o',color='tab:red',label='LMMSE, known task',linestyle='-')

MMSE=np.load('./logs_test_bits_'+str(BITS_TO_TEST)+'/MSE_dMMSE_Aware_SNR_'+str(SNR_dB)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy', allow_pickle=True)
ax.errorbar(MMSE[0],[d[-1] for d in MMSE[1]],yerr= [2*d[-1] for d in MMSE[2]]/np.sqrt(N_te),marker='^',color='k',label='MMSE, known task',linestyle='-')

ax.set_xlabel(r'Quantization bits ($b$)')
ax.set_ylabel('MSE [dB]')
plt.grid()
plt.legend()
plt.xlim([1,6])
plt.yscale('log')
ax.minorticks_off()
plt.tight_layout()
plt.savefig('./Results/MSEvsBits.pdf')
plt.show()
plt.clf()
plt.close()

