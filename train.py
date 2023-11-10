import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from EQ_data import batch_data_gen_quantized

def train_step(model, ys_batch, xs_batch, xs_real, optimizer, loss_func):
    model.train()
    output = model(ys_batch, xs_batch)
    optimizer.zero_grad()
    loss =mean_squared_error(output, xs_real)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def val_step(model, ys_batch, xs_batch, xs_real, optimizer, loss_func):
    model.eval()
    output = model(ys_batch, xs_batch)
    loss =mean_squared_error(output, xs_real)
    return loss.detach().item()

def mean_squared_error(xs_pred, xs):
    loss_state = xs - xs_pred
    loss_state_post = torch.norm(loss_state, dim=2)
    return (loss_state_post).square().mean()

def complex_to_vec(X):
    "Converts complex matrix to real vector"
    X_vec = np.concatenate([np.real(X) , np.imag(X) ], axis=2)
    return X_vec

def trainNetwork(model_GPT2,args, n_tasks):
    loss_function_model_GPT2 = nn.MSELoss(reduction='mean', size_average=True)
    optimizer_model_GPT2 = optim.Adam(model_GPT2.parameters(), lr=args.learning_rate)
    n_val=int(1e3)
    seed_task=1
    x_val, y_val,_,_ = batch_data_gen_quantized(n_val,n_tasks, args=args,seed_task=seed_task,seed_example=2**32-1)
    x_val, y_val = complex_to_vec(x_val),complex_to_vec(y_val)
    n_it_per_epoch=10
    LOG=[]
    log_every=10
    best_val=100
    best_it=0
    for jj in range(args.epochs):
        x_tr,  y_tr,_,_= batch_data_gen_quantized(int(args.batch_size*n_it_per_epoch),n_tasks, args=args,seed_task=seed_task,seed_example=jj)
        x_tr,  y_tr = complex_to_vec(x_tr),complex_to_vec(y_tr)
        running_loss=0
        for ii in range(int(n_it_per_epoch)):
            batch_id=np.arange(ii*args.batch_size,(ii+1)*args.batch_size)
            loss, output = train_step(model_GPT2, ys_batch=torch.Tensor(y_tr[batch_id,:,:]).to(device), xs_batch=torch.Tensor(x_tr[batch_id,:,:]).to(device), xs_real= torch.Tensor(x_tr[batch_id,:,:]).to(device), optimizer=optimizer_model_GPT2, loss_func=loss_function_model_GPT2)
            running_loss=running_loss+loss/n_it_per_epoch
        if jj%log_every==0:
            val_loss=val_step(model_GPT2, ys_batch=torch.Tensor(y_val).to(device), xs_batch=torch.Tensor(x_val).to(device), xs_real= torch.Tensor(x_val).to(device), optimizer=optimizer_model_GPT2, loss_func=loss_function_model_GPT2)
            LOG.append([jj,running_loss,val_loss])
            if val_loss<100:
                torch.save(model_GPT2, './model_parameters/model_'+str(args.model_type)+'_tasks_'+str(n_tasks)+'_bits_'+str(args.bits)+'_sigmoid.pth')
                best_val=val_loss
                best_it=jj
                print('Epoch : '+str(jj)+' -- New Best Model with Validation Loss:' + str(val_loss))
            else:
                print('Epoch : '+str(jj)+' -- Best model at Epoch '+str(best_it)+' with Validation Loss:'+str(best_val))
            np.save('./logs_train/LOG_model_'+str(args.model_type)+'_tasks_'+str(n_tasks)+'_bits_'+str(args.bits)+'_sigmoid', LOG)



