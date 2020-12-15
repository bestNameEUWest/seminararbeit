import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import ast
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import pickle
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def main():
  parser=argparse.ArgumentParser(description='Train the individual Transformer model')
  parser.add_argument('--dataset_folder',type=str,default='datasets')
  parser.add_argument('--dataset_name',type=str,default='preparation')
  parser.add_argument('--obs',type=int,default=8)
  parser.add_argument('--preds',type=int,default=12)
  parser.add_argument('--emb_size',type=int,default=512) 
  parser.add_argument('--steps',type=int,default=1) 
  parser.add_argument('--col_names', type=str, default='["frame", "obj", "x", "y"]')

  parser.add_argument('--heads',type=int, default=8) 
  parser.add_argument('--layers',type=int,default=4) 
  parser.add_argument('--dropout',type=float,default=0.1)

  parser.add_argument('--cpu',action='store_true')
  parser.add_argument('--verbose',action='store_true')
  parser.add_argument('--max_epoch',type=int, default=20)

  parser.add_argument('--batch_size',type=int,default=512) 

  parser.add_argument('--resume_train',action='store_true')
  parser.add_argument('--delim',type=str,default='\t')
  parser.add_argument('--name', type=str, default="rounD")
  parser.add_argument('--factor', type=float, default=1.)
  parser.add_argument('--print_step', type=int, default=1)
  parser.add_argument('--warmup', type=int, default=10) 
  parser.add_argument('--evaluate', type=bool, default=True)

  parser.add_argument('--run_info', type=str, default=None)

  args=parser.parse_args()
  model_name=f'{args.name}_{args.dataset_name}'  
  args.col_names = ast.literal_eval(args.col_names)

  dataset_info  = {
    'dataset_name': args.dataset_name,
    'obs': args.obs,
    'preds': args.preds,
    'steps': args.steps,
    'col_names': args.col_names,    
  }

  pytorch_data_save = 'pytorch_data_save'

  try:
    os.makedirs(f'models/{args.name}')
  except:
    pass
      
  try:
    os.makedirs(f'output/{args.name}')
  except:
    pass

  try:
    os.makedirs(f'{pytorch_data_save}/{args.dataset_name}')
  except:
    pass
 
  # data preparation check
  try:
    datasets_list = os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "train"))
    datasets_list = os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "val"))
    datasets_list = os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "test"))
  except FileNotFoundError:
    format_raw_dataset(args.dataset_folder, args.dataset_name)
  
  device=torch.device("cuda")

  if args.cpu or not torch.cuda.is_available():
    device=torch.device("cpu")
  args.verbose=True
  
  now = datetime.now()
  save_dir_name = now.strftime("%d-%m-%Y_%Ss-%Mm-%Hh")          
  available, path = baselineUtils.is_data_prepared(pytorch_data_save, dataset_info)  

  if available:    
    train_dataset = torch.load(os.path.join(path, "train", 'train.pt'))    
    val_dataset = torch.load(os.path.join(path, "val", 'val.pt'))    
    test_dataset = torch.load(os.path.join(path, "test", 'test.pt'))        
    print(f'Loaded prepared data with: {dataset_info}')          
  else:
    print('No prepared pytorch data found on drive. Preparing data...')    
    train_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.obs,args.preds, features=args.col_names, delim=args.delim, train=True, verbose=args.verbose, step=args.steps)          
    val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.obs, args.preds, features=args.col_names, delim=args.delim, train=False, verbose=args.verbose, step=args.steps)      
    test_dataset =  baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.obs,args.preds, features=args.col_names, delim=args.delim, train=False, eval=True, verbose=args.verbose, step=args.steps)    

    dss = [train_dataset, val_dataset, test_dataset]
    labels = ['train', 'val', 'test']
    for ds, label in zip(dss, labels):
      save_dir_path = os.path.join(pytorch_data_save, args.dataset_name, save_dir_name, label)  
      try:
        os.makedirs(save_dir_path)
      except:
        pass
      torch.save(ds, os.path.join(save_dir_path, f'{label}.pt'))

    save_dir_path = os.path.join(pytorch_data_save, args.dataset_name, save_dir_name)
    torch.save(dataset_info, os.path.join(save_dir_path, 'info.pt'))
    print(f'Prepared and saved data with: {dataset_info}')


  import individual_TF    
  from itertools import product
    
  tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  
  feature_count = len(args.col_names) - 2   
  
  save_comment = f'h={args.heads}_l={args.layers}_s={args.steps}_me={args.max_epoch}_o={args.obs}_p={args.preds}'
  train_comment = (f'heads={args.heads} ' +
                    f'layers={args.layers} ' +
                    f'steps={args.steps} ' +
                    f'max_epochs={args.max_epoch} ' +
                    f'obs={args.obs} ' +
                    f'preds={args.preds} ')

  if args.run_info is not None:        
    save_comment = f'{args.run_info}_{save_comment}'
    train_comment = f'{args.run_info} {train_comment}'        
    log=SummaryWriter(log_dir=f'runs/{save_comment}')
  else:        
    log=SummaryWriter()


  model=individual_TF.IndividualTF(feature_count, 3, 3, N=args.layers,
                d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
  optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                      torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  input_means=[]
  input_stds=[]
  target_means=[]
  target_stds=[]
  
  for i in np.unique(train_dataset[:]['dataset']):
    ind=train_dataset[:]['dataset']==i

    # take feature_count "velocity" values    
    input_src = train_dataset[:]['src'][ind, 1:, feature_count:feature_count*2]
    input_trg = train_dataset[:]['trg'][ind, :, feature_count:feature_count*2]

    # calculate mean and std over the features
    input_src_mean = torch.cat((input_src, input_trg), 1).mean((0, 1))
    input_src_std = torch.cat((input_src, input_trg), 1).std((0, 1))

    # safe mean and std values of this dataset
    input_means.append(input_src_mean)
    input_stds.append(input_src_std)

    # safe only coordinate velocities mean and std
    target_means.append(input_src_mean[:2])
    target_stds.append(input_src_std[:2])
  
  # calculate the mean and std of the features of all datasets
  input_mean=torch.stack(input_means).mean(0)
  input_std=torch.stack(input_stds).std(0)        

  # calculate the mean and std of only the coordinate velocities of all datasets
  target_mean=torch.stack(target_means).mean(0)
  target_std=torch.stack(target_stds).std(0)        
  
  scipy.io.savemat(f'models/{args.name}/norm.mat',{'mean':input_mean.cpu().numpy(),'std':input_std.cpu().numpy()})
  
  print(f'Training for: {train_comment}')
  epoch=0

  t0 = time.time()

  while epoch<args.max_epoch:
    epoch_loss=0
    e_t0 = time.time()
    model.train()

    train_batch_len = len(tr_dl)

    for id_b,batch in enumerate(tr_dl):
      optim.optimizer.zero_grad()

      # the input consists of all features
      inp=(batch['src'][:,1:,feature_count:feature_count*2].to(device)-input_mean.to(device))/input_std.to(device)
      target=(batch['trg'][:, :-1, feature_count:feature_count+2].to(device)-target_mean.to(device))/target_std.to(device)
      target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
      target=torch.cat((target,target_c),-1)      
      start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
      
      dec_inp = torch.cat((start_of_seq, target), 1)

      src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
      trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

      pred=model(inp, dec_inp, src_att, trg_att)

      y_pred = pred[:, :,0:2].contiguous().view(-1, 2)


      y_real = ((batch['trg'][:, :, feature_count:feature_count+2].to(device)-target_mean.to(device))/target_std.to(device)).contiguous().view(-1, 2).to(device)
                
      loss = F.pairwise_distance(y_pred, y_real).mean() + torch.mean(torch.abs(pred[:,:,2]))


      loss.backward()
      optim.step()

      epoch_loss += loss.item()

    log.add_scalar(f'Loss/train/layers_{args.layers}', epoch_loss / len(tr_dl), epoch)        
    epoch+=1

    if epoch==1:
      torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')

    if epoch%args.print_step==0:         
      print("Epoch: %03i/%03i  Training time: %03.4f  Loss: %03.4f  Avg. Loss: %03.4f" % (epoch, args.max_epoch, time.time()-e_t0, epoch_loss, epoch_loss / len(tr_dl)))

  log.add_hparams({'heads': args.heads, 'layers': args.layers, 'observations': args.obs,
                  'predictions': args.preds, 'max_epoch': args.max_epoch, 'steps': args.steps},
                  {'hparam/avg loss': epoch_loss / len(tr_dl)})        
  print("Total training time: %07.4f" % (time.time()-t0))



if __name__=='__main__':
  main()