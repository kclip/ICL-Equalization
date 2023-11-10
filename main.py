from train import trainNetwork
from parameters import parameter_reading
import torch
from models import build_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parameter_reading()


TASKs= [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,8192*2,8192*4]
for M in TASKs:
    model = build_model(embedding_dim=args.embedding_dim, n_positions=args.prompt_seq_length, num_heads=args.num_head,num_layers=args.num_layer,data_dim=args.data_dim).to(device)
    test_loss = trainNetwork(model.to(device),args,n_tasks=M)
    print("training is done")



