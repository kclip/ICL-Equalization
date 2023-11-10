import matplotlib.pyplot as plt
import numpy as np
import torch
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (5,4.5)
model_type='GPT2'
context_size=40
SNR_dB=10
bits=4
N_te=1000
pretrain=False
n_tasks=16384

fig, ax = plt.subplots()

n_tasks=16384*2
ICL=np.load('./logs_test_bits_'+str(bits)+'/MSE_'+str(model_type)+'_SNR_'+str(SNR_dB)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy',allow_pickle=True)
plt.errorbar(ICL[0],[d[-1] for d in ICL[1]],yerr=[2*d[-1] for d in ICL[2]]/np.sqrt(N_te),marker='x',color='tab:green',label='ICL',linestyle='-')


n_tasks=16384*8
dMMSE_g=np.load('./logs_test_bits_'+str(bits)+'/MSE_dMMSE_Gauss_SNR_'+str(SNR_dB)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy',allow_pickle=True)
ax.errorbar(ICL[0],np.ones(len(ICL[0]))*[d[-1] for d in dMMSE_g[1]],yerr=np.ones(len(ICL[0]))*[2*d[-10] for d in dMMSE_g[2]]/np.sqrt(N_te),marker='o',color='tab:orange',label='MMSE known task distr.',linestyle='-')


n_tasks=16384*2
dMMSE=np.load('./logs_test_bits_'+str(bits)+'/MSE_dMMSE_Uniform_SNR_'+str(SNR_dB)+'_tasks_'+str(n_tasks)+'_pretrain_'+str(pretrain)+'.npy',allow_pickle=True)
ax.errorbar(dMMSE[0],[d[-1] for d in dMMSE[1]],yerr=[2*d[-10] for d in dMMSE[2]]/np.sqrt(N_te),marker='s',color='tab:blue',label='MMSE known pre-training distr.',linestyle='-')

labels_log=[r'$1$',r'$2$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$',r'$2^6$',r'$2^7$',r'$2^8$',r'$2^9$',r'$2^{10}$',r'$2^{11}$',r'$2^{12}$',r'$2^{13}$',r'$2^{14}$',r'$2^{15}$']


ax.set_xlabel(r'Pre-training tasks ($M$)')
ax.set_ylabel('MSE')
plt.grid()
plt.legend()
plt.xlim([np.min(ICL[0]),np.max(ICL[0])])
plt.xscale('log')
ax.minorticks_off()
plt.tight_layout()
ax.set_xticks(np.asarray(ICL[0]).astype(float), labels=labels_log)
plt.savefig('./Results/MSE_vs_tasks.pdf')
plt.show()
plt.clf()
plt.close()

