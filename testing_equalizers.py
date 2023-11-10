import numpy as np
import sys
import torch
import os
from EQ_data import batch_data_gen_quantized
from parameters import parameter_reading
from scipy.stats import norm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def complex_to_vec(X):
    "Converts complex matrix to real vector"
    X_vec = np.concatenate([np.real(X) , np.imag(X) ], axis=-1)
    return X_vec

def adjust_sequence_length(tensor, desired_length=20, padding_value=0):
    """Adjust the sequence length of the data tensor to the desired length."""
    bsize, current_length, dim = tensor.shape
    if current_length == desired_length:
        return tensor
    elif current_length < desired_length:
        padding_dims = (0, 0, 0, desired_length - current_length)
        return torch.nn.functional.pad(tensor, padding_dims, value=padding_value)
    else:
        return tensor[:, :desired_length, :]

def MMMSE_channel_aware(args,n_te,n_tasks,SNR_in_dB,TEST_ON_PRETRAIN,bits=8):
    '''MMSE solution obtained with perfect channel knowledge'''
    noiseVar = 10 ** (-SNR_in_dB / 10.)
    err = []
    seed_task = 1
    l_min = -4
    l_max = 4
    precompute_cdf=np.asarray([norm.cdf(x) for x in np.linspace(-3,3,6001)])
    np.random.seed(seed_task)
    quant_step=(l_max-l_min)/(2**bits)
    quant_levels=np.arange(2**(bits))*quant_step+quant_step/2-2**(bits-1)*quant_step
    Q_regions=(quant_levels[0:-1]+quant_levels[1:])/2
    Q_regions=np.append([-100],Q_regions)
    Q_regions = np.append( Q_regions,[100])
    Xs=np.asarray([[1+1j],[1-1j],[-1+1j],[-1-1j]])/np.sqrt(2)
    Xs=np.squeeze(np.vstack([np.stack([np.asarray([y,x])  for x in Xs]) for y in Xs]))
    '''Compute task Channel Matrices'''
    if TEST_ON_PRETRAIN:
        '''If test on pretrain: generate n_te examples using the same H_task matrices. This can be done fixing the seed_task equal to the one used to generate H_task'''
        x_te, y_te, H_task_gt,_ =  batch_data_gen_quantized(n_te,n_tasks, args,seed_task=1,seed_example=0)
    else:
        '''If test on true distribution: generate n_te examples, each using a different task. This is done fixing a different seed_task'''
        x_te, y_te, H_task_gt,_ = batch_data_gen_quantized(n_te, n_te, args, seed_task=2 ** 32 - 1, seed_example=0)
    i=0
    for (x_co,y_co) in zip(x_te, y_te):
        '''Compute the quantization index of each symbol'''
        y_te_re_id = ((np.real(y_co)-np.min(quant_levels))/quant_step).astype(int)
        y_te_im_id = ((np.imag(y_co)-np.min(quant_levels))/quant_step).astype(int)
        '''Obtain the final hypothesis wweighting each hypothesis by the likelihood'''
        y_eval=H_task_gt[i][np.newaxis,:,:]  @ (Xs.T)
        y_eval_re = np.transpose(np.real(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        y_eval_im = np.transpose(np.imag(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        '''Real Likelihood'''
        min_qre_re = ((((Q_regions[y_te_re_id] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_re = ((((Q_regions[y_te_re_id + 1] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_re[min_qre_re < 0] = 0
        min_qre_re[min_qre_re > 6000] = 6000
        max_qre_re[max_qre_re < 0] = 0
        max_qre_re[max_qre_re > 6000] = 6000
        log_P_re = np.log(precompute_cdf[max_qre_re] - precompute_cdf[min_qre_re]+10e-40)
        '''Imag Likelihood'''
        min_qre_im = ((((Q_regions[y_te_im_id] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_im = ((((Q_regions[y_te_im_id + 1] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_im[min_qre_im < 0] = 0
        min_qre_im[min_qre_im > 6000] = 6000
        max_qre_im[max_qre_im < 0] = 0
        max_qre_im[max_qre_im > 6000] = 6000
        log_P_im = np.log(precompute_cdf[max_qre_im] - precompute_cdf[min_qre_im]+10e-40)
        p_x=np.transpose(np.sum(log_P_im+log_P_re,axis=3), (0, 2, 1))
        c = np.max(p_x, axis=2)
        log_sum_exp = np.log(np.exp(p_x - c[:,:,np.newaxis]).sum(axis=2)) + c
        p_x=np.exp(p_x - log_sum_exp[:,:, np.newaxis])
        '''Compute the likelihood terms'''
        x_hats = np.squeeze(p_x@Xs)
        '''Calculate the error'''
        err.append(np.mean(np.power(complex_to_vec(x_co-x_hats),2),axis=1))
        i=i+1
    return np.mean(err,axis=0),np.std(err,axis=0)

def linear_MMSE_channel_aware(args,n_te,n_tasks,SNR_in_dB,TEST_ON_PRETRAIN,bits=8):
    '''Linear Regression Model with Channel Knowledge'''
    args.SNR_dB_min = SNR_in_dB
    args.SNR_dB_max = SNR_in_dB
    args.bits = bits
    seed_task=1
    noiseVar = 10 ** (-SNR_in_dB / 10.)
    '''MMSE and ZF Receiver Assuming Knowledge of channel Matrix H'''
    if TEST_ON_PRETRAIN:
        '''If test on pretrain: generate n_te examples using the same H_task matrices. This can be done fixing the seed_task equal to the one used to generate H_task'''
        x_te, y_te, H_task_gt,_ =  batch_data_gen_quantized(n_te,n_tasks, args,seed_task=1,seed_example=0)
    else:
        '''If test on true distribution: generate n_te examples, each using a different task. This is done fixing a different seed_task'''
        x_te, y_te, H_task_gt,_ = batch_data_gen_quantized(n_te, n_te, args, seed_task=2 ** 32 - 1, seed_example=0)
    i = 0
    loss_mse,loss_zf=[],[]
    for (x,y) in zip (x_te, y_te):
        H=H_task_gt[i]
        mmseEstimation = np.linalg.inv(H.conj().T @ H + noiseVar * np.eye(2)) @ H.conj().T @ np.transpose(y)
        loss_mse.append(np.mean(np.power(complex_to_vec(mmseEstimation - np.transpose(x)), 2),axis=0))
        i=i+1
    return np.mean(loss_mse,axis=0),np.std(loss_mse,axis=0)

def MMMSE_post_avg_uniform(args,n_te,n_tasks,SNR_in_dB,TEST_ON_PRETRAIN,bits=8):
    '''Optimal MMSE solution obtained from a uniform prior over the set of pretraining tasks'''
    noiseVar = 10 ** (-SNR_in_dB / 10.)
    err = []
    seed_task = 1
    l_min = -4
    l_max = 4
    precompute_cdf=np.asarray([norm.cdf(x) for x in np.linspace(-3,3,6001)])
    np.random.seed(seed_task)
    quant_step=(l_max-l_min)/(2**bits)
    quant_levels=np.arange(2**(bits))*quant_step+quant_step/2-2**(bits-1)*quant_step
    Q_regions=(quant_levels[0:-1]+quant_levels[1:])/2
    Q_regions=np.append([-100],Q_regions)
    Q_regions = np.append( Q_regions,[100])
    Xs=np.asarray([[1+1j],[1-1j],[-1+1j],[-1-1j]])/np.sqrt(2)
    Xs=np.squeeze(np.vstack([np.stack([np.asarray([y,x])  for x in Xs]) for y in Xs]))
    '''Compute task Channel Matrices'''
    H_task = (np.random.randn(n_tasks, args.num_ant, args.num_ant) + 1j * np.random.randn(n_tasks, args.num_ant, args.num_ant)) / np.sqrt(2)
    if TEST_ON_PRETRAIN:
        '''If test on pretrain: generate n_te examples using the same H_task matrices. This can be done fixing the seed_task equal to the one used to generate H_task'''
        x_te, y_te, H, _ = batch_data_gen_quantized(n_te, n_tasks, args, seed_task=seed_task, seed_example=0)
    else:
        '''If test on true distribution: generate n_te examples, each using a different task. This is done fixing a different seed_task'''
        x_te, y_te, _, _ = batch_data_gen_quantized(n_te, n_te, args, seed_task=2 ** 32 - 1, seed_example=0)
    for (x_co,y_co) in zip(x_te, y_te):
        y_hat=np.transpose(H_task @ (x_co.T),(0,2,1))
        '''Extract Real and Imaginary Part of Hx'''
        y_hat_re = np.real(y_hat)
        '''Compute the quantization index of each symbol'''
        y_te_re_id = ((np.real(y_co)-np.min(quant_levels))/quant_step).astype(int)
        y_te_im_id = ((np.imag(y_co)-np.min(quant_levels))/quant_step).astype(int)
        '''Compute the c.d.f. of the upper and lower bounds of the associated quantization region'''
        min_qre_re = ((((Q_regions[y_te_re_id] - y_hat_re)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        max_qre_re = ((((Q_regions[y_te_re_id+1] - y_hat_re)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        min_qre_re[min_qre_re<0]=0
        min_qre_re[min_qre_re > 6000] = 6000
        max_qre_re[max_qre_re < 0] = 0
        max_qre_re[max_qre_re > 6000] = 6000
        '''Compute log probability of the real symbols'''
        log_P_re= np.log(precompute_cdf[max_qre_re]-precompute_cdf[min_qre_re]+10e-20)
        '''Same thing for imaginary part'''
        y_hat_im = np.imag(y_hat)
        min_qre_im = ((((Q_regions[y_te_im_id] - y_hat_im)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        max_qre_im = ((((Q_regions[y_te_im_id+1] - y_hat_im)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        min_qre_im[min_qre_im < 0] = 0
        min_qre_im[min_qre_im > 6000] = 6000
        max_qre_im[max_qre_im < 0] = 0
        max_qre_im[max_qre_im > 6000] = 6000
        log_P_im = np.log(precompute_cdf[max_qre_im] - precompute_cdf[min_qre_im]+10e-20)
        '''Computation of the final posterior'''
        posterior=np.transpose(np.cumsum(np.sum(log_P_im+log_P_re,axis=2),axis=1))
        '''This part is the log sum exp trick for numerical stability'''
        c=np.max(posterior,axis=1)
        log_sum_exp=np.log(np.exp(posterior-c[:, np.newaxis]).sum(axis=1))+c
        '''Compute the likelihood terms'''
        posterior=np.exp(posterior-log_sum_exp[:, np.newaxis])
        '''Obtain the final hypothesis wweighting each hypothesis by the likelihood'''
        y_eval=H_task @ (Xs.T)
        y_eval_re = np.transpose(np.real(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        y_eval_im = np.transpose(np.imag(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        '''Real Likelihood'''
        min_qre_re = ((((Q_regions[y_te_re_id] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_re = ((((Q_regions[y_te_re_id + 1] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_re[min_qre_re < 0] = 0
        min_qre_re[min_qre_re > 6000] = 6000
        max_qre_re[max_qre_re < 0] = 0
        max_qre_re[max_qre_re > 6000] = 6000
        log_P_re = np.log(precompute_cdf[max_qre_re] - precompute_cdf[min_qre_re]+10e-20)
        '''Imag Likelihood'''
        min_qre_im = ((((Q_regions[y_te_im_id] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_im = ((((Q_regions[y_te_im_id + 1] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_im[min_qre_im < 0] = 0
        min_qre_im[min_qre_im > 6000] = 6000
        max_qre_im[max_qre_im < 0] = 0
        max_qre_im[max_qre_im > 6000] = 6000
        log_P_im = np.log(precompute_cdf[max_qre_im] - precompute_cdf[min_qre_im]+10e-20)
        p_x=np.transpose(np.sum(log_P_im+log_P_re,axis=3), (0, 2, 1))
        c = np.max(p_x, axis=2)
        log_sum_exp = np.log(np.exp(p_x - c[:,:,np.newaxis]).sum(axis=2)) + c
        '''Compute the likelihood terms'''
        x_hats = np.exp(p_x - log_sum_exp[:,:, np.newaxis])@Xs
        x_hat = [posterior[i,:]@x_hats[:,i+1,:]for i in range(0,args.prompt_seq_length-1)]
        '''Calculate the error'''
        err.append(np.asarray([np.mean(np.power(complex_to_vec(x_co[i+1]-x_hat[i]),2)) for i in range(0,len(x_hat))]))
    return np.mean(err,axis=0),np.std(err,axis=0)

def MMMSE_post_avg_gauss(args,n_te,n_tasks,n_int_samples,SNR_in_dB,TEST_ON_PRETRAIN,bits=8):
    '''MMSE solution obtained from a uniform prior over the set of pretraining tasks using numerical integration with n_int_samples MC samples '''
    args.SNR_dB_min = SNR_in_dB
    args.SNR_dB_max = SNR_in_dB
    args.bits = bits
    noiseVar = 10 ** (-SNR_in_dB / 10.)
    err = []
    seed_task = 1
    l_min = -4
    l_max = 4
    precompute_cdf=np.asarray([norm.cdf(x) for x in np.linspace(-3,3,6001)])
    np.random.seed(seed_task)
    quant_step=(l_max-l_min)/(2**bits)
    quant_levels=np.arange(2**(bits))*quant_step+quant_step/2-2**(bits-1)*quant_step
    Q_regions=(quant_levels[0:-1]+quant_levels[1:])/2
    Q_regions=np.append([-100],Q_regions)
    Q_regions = np.append( Q_regions,[100])
    Xs=np.asarray([[1+1j],[1-1j],[-1+1j],[-1-1j]])/np.sqrt(2)
    Xs=np.squeeze(np.vstack([np.stack([np.asarray([y,x])  for x in Xs]) for y in Xs]))
    '''Compute task Channel Matrices'''
    H_task = (np.random.randn(n_int_samples, args.num_ant, args.num_ant) + 1j * np.random.randn(n_int_samples, args.num_ant, args.num_ant)) / np.sqrt(2)
    if TEST_ON_PRETRAIN:
        '''If test on pretrain: generate n_te examples using the same H_task matrices. This can be done fixing the seed_task equal to the one used to generate H_task'''
        x_te, y_te,H,_ = batch_data_gen_quantized(n_te, n_tasks, args, seed_task=seed_task, seed_example=0)
    else:
        '''If test on true distribution: generate n_te examples, each using a different task. This is done fixing a different seed_task'''
        x_te, y_te,_,_ = batch_data_gen_quantized(n_te, n_te, args, seed_task=2 ** 32 - 1, seed_example=0)
    for (x_co,y_co) in zip(x_te, y_te):
        y_hat=np.transpose(H_task @ (x_co.T),(0,2,1))
        '''Extract Real and Imaginary Part of Hx'''
        y_hat_re = np.real(y_hat)
        '''Compute the quantization index of each symbol'''
        y_te_re_id = ((np.real(y_co)-np.min(quant_levels))/quant_step).astype(int)
        y_te_im_id = ((np.imag(y_co)-np.min(quant_levels))/quant_step).astype(int)
        '''Compute the c.d.f. of the upper and lower bounds of the associated quantization region'''
        min_qre_re = ((((Q_regions[y_te_re_id] - y_hat_re)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        max_qre_re = ((((Q_regions[y_te_re_id+1] - y_hat_re)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        min_qre_re[min_qre_re<0]=0
        min_qre_re[min_qre_re > 6000] = 6000
        max_qre_re[max_qre_re < 0] = 0
        max_qre_re[max_qre_re > 6000] = 6000
        '''Compute log probability of the real symbols'''
        log_P_re= np.log(precompute_cdf[max_qre_re]-precompute_cdf[min_qre_re]+10e-20)
        '''Same thing for imaginary part'''
        y_hat_im = np.imag(y_hat)
        min_qre_im = ((((Q_regions[y_te_im_id] - y_hat_im)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        max_qre_im = ((((Q_regions[y_te_im_id+1] - y_hat_im)/np.sqrt(noiseVar)+3)/6)*6000).astype(int)
        min_qre_im[min_qre_im < 0] = 0
        min_qre_im[min_qre_im > 6000] = 6000
        max_qre_im[max_qre_im < 0] = 0
        max_qre_im[max_qre_im > 6000] = 6000
        log_P_im = np.log(precompute_cdf[max_qre_im] - precompute_cdf[min_qre_im]+10e-20)
        '''Computation of the final posterior'''
        posterior=np.transpose(np.cumsum(np.sum(log_P_im+log_P_re,axis=2),axis=1))
        '''This part is the log sum exp trick for numerical stability'''
        c=np.max(posterior,axis=1)
        log_sum_exp=np.log(np.exp(posterior-c[:, np.newaxis]).sum(axis=1))+c
        '''Compute the likelihood terms'''
        posterior=np.exp(posterior-log_sum_exp[:, np.newaxis])
        '''Obtain the final hypothesis wweighting each hypothesis by the likelihood'''
        y_eval=H_task @ (Xs.T)
        y_eval_re = np.transpose(np.real(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        y_eval_im = np.transpose(np.imag(y_eval), (0, 2, 1))[:,:,np.newaxis,:]
        '''Real Likelihood'''
        min_qre_re = ((((Q_regions[y_te_re_id] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_re = ((((Q_regions[y_te_re_id + 1] - y_eval_re)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_re[min_qre_re < 0] = 0
        min_qre_re[min_qre_re > 6000] = 6000
        max_qre_re[max_qre_re < 0] = 0
        max_qre_re[max_qre_re > 6000] = 6000
        log_P_re = np.log(precompute_cdf[max_qre_re] - precompute_cdf[min_qre_re]+10e-20)
        '''Imag Likelihood'''
        min_qre_im = ((((Q_regions[y_te_im_id] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        max_qre_im = ((((Q_regions[y_te_im_id + 1] - y_eval_im)/np.sqrt(noiseVar) + 3) / 6) * 6000).astype(int)
        min_qre_im[min_qre_im < 0] = 0
        min_qre_im[min_qre_im > 6000] = 6000
        max_qre_im[max_qre_im < 0] = 0
        max_qre_im[max_qre_im > 6000] = 6000
        log_P_im = np.log(precompute_cdf[max_qre_im] - precompute_cdf[min_qre_im]+10e-20)
        p_x=np.transpose(np.sum(log_P_im+log_P_re,axis=3), (0, 2, 1))
        c = np.max(p_x, axis=2)
        log_sum_exp = np.log(np.exp(p_x - c[:,:,np.newaxis]).sum(axis=2)) + c
        '''Compute the likelihood terms'''
        x_hats = np.exp(p_x - log_sum_exp[:,:, np.newaxis])@Xs
        x_hat = [posterior[i,:]@x_hats[:,i+1,:]for i in range(0,args.prompt_seq_length-1)]
        '''Calculate the error'''
        err.append(np.asarray([np.mean(np.power(complex_to_vec(x_co[i+1]-x_hat[i]),2)) for i in range(0,len(x_hat))]))
    return np.mean(err,axis=0),np.std(err,axis=0)

def ICL_equalizer(model,args, n_te, n_tasks, SNR_in_dB,TEST_ON_PRETRAIN,bits=8):
    '''Test In-Context Equalization at a fixed SNR level'''
    model.eval()  # Set the model to evaluation mode
    args.SNR_dB_min=SNR_in_dB
    args.SNR_dB_max=SNR_in_dB
    args.bits=bits
    with torch.no_grad():
        # Generate test data
        if TEST_ON_PRETRAIN:
            te_x, te_y,_ = batch_data_gen_simple_quantized(n_te,n_tasks, args,seed_task=1,seed_example=0)
        else:
            te_x, te_y,_ = batch_data_gen_simple_quantized(n_te, n_te, args, seed_task=2 ** 32 - 1, seed_example=0)
        te_x = complex_to_vec(te_x)
        te_y = complex_to_vec(te_y)
        # Adjust based on in-context size
        te_y = te_y[:, :args.context_size, :]
        te_x = te_x[:, :args.context_size, :]
        # In the test loop:
        te_y = adjust_sequence_length(te_y, args.context_size)
        te_x = adjust_sequence_length(te_x, args.context_size)
        # Model's forward pass
        output = model(torch.Tensor(te_y).to(device), torch.Tensor(te_x).to(device))
        loss_mean=torch.mean(torch.mean((output.cpu()-te_x).square(),axis=2),axis=0)
        loss_std=torch.std(torch.mean((output.cpu()-te_x).square(),axis=2),axis=0)
    return loss_mean,loss_std

'''EXAMPLE'''
args = parameter_reading()
MMMSE_post_avg_gauss(args,100,100,1000,10,False,)