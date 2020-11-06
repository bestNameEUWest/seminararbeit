import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--torch_dataset_folder',type=str,default='torch_datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512) 

    parser.add_argument('--heads',type=int, default=8) 
    parser.add_argument('--layers',type=int,default=6) 
    parser.add_argument('--dropout',type=float,default=0.1)

    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=20)

    parser.add_argument('--batch_size',type=int,default=500) 

    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--print_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10) 
    parser.add_argument('--evaluate', type=bool, default=True)

    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:        
        os.mkdir(f'{args.torch_dataset_folder}')
    except:
        pass
    
    try:
        os.mkdir('output/Individual')
    except:
        pass
    try:
        os.mkdir(f'models/Individual')
    except:
        pass
    try:
        os.mkdir(f'{args.torch_dataset_folder}/Individual')
    except:
        pass
    
    try:
        os.mkdir(f'output/Individual/{args.name}')
    except:
        pass
    try:
        os.mkdir(f'models/Individual/{args.name}')
    except:
        pass
    try:
        os.mkdir(f'{args.torch_dataset_folder}/Individual/{args.name}')
    except:
        pass
    
    
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True    
        
    try:        
        train_dataset = torch.load(os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_train.pt'))
        val_dataset = torch.load(os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_val.pt'))
        test_dataset = torch.load(os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_test.pt'))
        print('Loaded pytorch data from drive.')
    except FileNotFoundError:
        print('No pytorch data on drive found. Data Preparing ...')
        if args.val_size==0:
            train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
            val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                        args.preds, delim=args.delim, train=False,
                                                                        verbose=args.verbose)
        else:
            train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs,
                                                                  args.preds, delim=args.delim, train=True,
                                                                  verbose=args.verbose)

        test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

        torch.save(train_dataset, os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_train.pt'))
        torch.save(val_dataset, os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_val.pt'))
        torch.save(test_dataset, os.path.join(args.torch_dataset_folder, 'Individual', args.name, f'torch_{args.dataset_name}_test.pt'))        
   
    
    import individual_TF    
    from itertools import product

    # prepare for hyperparameter searching
    # we currently leave layers at 4 and batch_size at 32 for quicker learning
    params = {
      'heads': [args.heads], #[4, 16],
      'layers': [4], #[4, 8], 
      'batch_size': [32] #[32, 512]
    }
    
    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))

    param_values = [p for p in params.values()]    
    for heads, layers, batch_size in product(*param_values):                
        args.heads = heads
        args.layers = layers
        args.batch_size = batch_size

        comment = f" batch_size={batch_size} layers={layers} heads={heads}"
        print(f"Training for {comment}")
        log=SummaryWriter(comment=comment) #'logs/Ind_%s'%model_name, 

        model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                      d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)

        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        means=[]
        stds=[]
        for i in np.unique(train_dataset[:]['dataset']):
            ind=train_dataset[:]['dataset']==i
            means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
            stds.append(
                torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
        mean=torch.stack(means).mean(0)
        std=torch.stack(stds).mean(0)

        scipy.io.savemat(f'models/Individual/{args.name}/norm.mat',{'mean':mean.cpu().numpy(),'std':std.cpu().numpy()})

        print('Training ...')
        epoch=0

        t0 = time.time()

        while epoch<args.max_epoch:

            epoch_loss=0
            e_t0 = time.time()
            model.train()

            train_batch_len = len(tr_dl)

            for id_b,batch in enumerate(tr_dl):
                optim.optimizer.zero_grad()
                inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
                target=torch.cat((target,target_c),-1)
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

                dec_inp = torch.cat((start_of_seq, target), 1)

                src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

                pred=model(inp, dec_inp, src_att, trg_att)

                loss = F.pairwise_distance(pred[:, :,0:2].contiguous().view(-1, 2),
                                          ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))
                loss.backward()
                optim.step()

                epoch_loss += loss.item()

            log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)        

            epoch+=1
            if epoch==1:
                torch.save(model.state_dict(),f'models/Individual/{args.name}/{epoch:05d}.pth')

            if epoch%args.print_step==0:         
                print("Epoch: %03i/%03i  Training time: %03.4f  Loss: %03.4f" % (epoch, args.max_epoch, time.time()-e_t0, epoch_loss))

        print("Total training time: %07.4f" % (time.time()-t0))



if __name__=='__main__':
    main()