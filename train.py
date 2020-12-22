# system
import argparse
import os
import time
from datetime import datetime

# external
import numpy as np
import scipy.io
import pickle
import optuna
import ast

# pytorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,SGD, RMSprop, Adagrad
from torch.utils.tensorboard import SummaryWriter

# custom
import baselineUtils
import individual_TF    
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt


################################################################################
####################### less important preparation stuff #######################
################################################################################

# handle arguments and return args object
def argparser_function():  
  parser=argparse.ArgumentParser(description='Train the individual Transformer model')
  
  # usually constants
  parser.add_argument('--dataset_folder',type=str,default='datasets')
  parser.add_argument('--obs',type=int,default=8)
  parser.add_argument('--preds',type=int,default=12)
  parser.add_argument('--steps',type=int,default=10) 
  parser.add_argument('--cpu',action='store_true')  
  parser.add_argument('--verbose',action='store_true')  
  parser.add_argument('--resume_train',action='store_true')
  parser.add_argument('--delim',type=str,default='\t')
  parser.add_argument('--name', type=str, default="rounD")
  parser.add_argument('--print_step', type=int, default=1)
  parser.add_argument('--warmup', type=int, default=10) 
  parser.add_argument('--evaluate', type=bool, default=True)
  
  # variables
  parser.add_argument('--dataset_name',type=str,default='preparation')
  parser.add_argument('--col_names', type=str, default='["frame", "obj", "x", "y"]')
  parser.add_argument('--max_epoch',type=int, default=20)
  parser.add_argument('--batch_size',type=int,default=512)   
  parser.add_argument('--run_info', type=str, default=None)

  # default hyperparameters in case we do not use optuna
  parser.add_argument('--emb_size',type=int,default=512) 
  parser.add_argument('--heads',type=int, default=8) 
  parser.add_argument('--layers',type=int,default=4) 
  parser.add_argument('--dropout',type=float,default=0.1)    
  parser.add_argument('--factor', type=float, default=1.)
  
  args=parser.parse_args()
  args.col_names = ast.literal_eval(args.col_names)

  return args

# handle the preparation of datasets and folders
def dataset_and_folder_prep(args):
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

  model_name=f'{args.name}_{args.dataset_name}'  
  
  dataset_info  = {
    'dataset_name': args.dataset_name,
    'obs': args.obs,
    'preds': args.preds,
    'steps': args.steps,
    'col_names': args.col_names,    
  }

  pytorch_data_save = 'pytorch_data_save'
 
  # data preparation check
  try:
    os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "train"))
    os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "val"))
    os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "test"))
  except FileNotFoundError:
    format_raw_dataset(args.dataset_folder, args.dataset_name)
  
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

  return train_dataset, val_dataset, test_dataset

def means_and_stds(train_dataset, feature_count):
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

  return input_mean, input_std, target_mean, target_std




################################################################################
######################## more important learning stuff #########################
################################################################################


# define model based on some hyperparams
def define_model_and_optimizer(trial, warmup, feature_count, device):
  n_layers = trial.suggest_int('layers', 1, 32) 
  n_emb_size = 2**trial.suggest_int('emb_size', 4, 12) # must be tested for max value
  n_heads = 2**trial.suggest_int('heads', 1, 4)
  n_dropout = trial.suggest_float('dropout', 0.1, 0.9)

  print(f'n_layers: {n_layers}')
  print(f'n_emb_size: {n_emb_size}')
  print(f'n_heads: {n_heads}')
  print(f'n_dropout: {n_dropout}')

  model=individual_TF.IndividualTF(feature_count, 3, 3, N=n_layers, d_model=n_emb_size,
                                   d_ff=2048, h=n_heads, dropout=n_dropout,mean=[0,0],std=[0,0]).to(device)                              
  
  n_factor = 1.0
  optim = NoamOpt(n_emb_size, n_factor, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  return model, optim
  

# objective function to minimize for optuna
def objective(trial):
  args = argparser_function()
  feature_count = len(args.col_names) - 2 
  train_dataset, val_dataset, test_dataset = dataset_and_folder_prep(args)  
  input_mean, input_std, target_mean, target_std = means_and_stds(train_dataset, feature_count)
  scipy.io.savemat(f'models/{args.name}/norm.mat',{'mean':input_mean.cpu().numpy(),'std':input_std.cpu().numpy()})
  
  device=torch.device("cuda")

  if args.cpu or not torch.cuda.is_available():
    device=torch.device("cpu")
  args.verbose=True    
    
  tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  
  
  
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

  model, optim = define_model_and_optimizer(trial, len(tr_dl)*args.warmup, feature_count, device)    
   
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

    log.add_scalar(f'Loss/train/', epoch_loss / len(tr_dl), epoch)

    with torch.no_grad():
      model.eval()

      val_loss=0
      step=0
      model.eval()
      gt = []
      pr = []
      inp_ = []
      objs = []
      frames = []
      dt = []

      for id_b, batch in enumerate(val_dl):
        inp_.append(batch['src'])
        gt.append(batch['trg'][:, :, 0:2])
        frames.append(batch['frames'])
        objs.append(batch['objs'])
        dt.append(batch['dataset'])

        inp = (batch['src'][:, 1:, feature_count:feature_count*2].to(device) - input_mean.to(device)) / input_std.to(device)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
        dec_inp = start_of_seq

        for i in range(args.preds):
          trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
          out = model(inp, dec_inp, src_att, trg_att)
          dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

            
        preds_tr_b = (dec_inp[:, 1:, 0:2] * target_std.to(device) + target_mean.to(device)).cpu().numpy().cumsum(1) + batch['src'][:, -1:, 0:2].cpu().numpy()
        pr.append(preds_tr_b)

      objs = np.concatenate(objs, 0)
      frames = np.concatenate(frames, 0)
      dt = np.concatenate(dt, 0)
      gt = np.concatenate(gt, 0)
      dt_names = test_dataset.data['dataset_name']
      pr = np.concatenate(pr, 0)
      mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
      log.add_scalar('validation/MAD', mad, epoch)
      log.add_scalar('validation/FAD', fad, epoch)

      print("Epoch: %03i/%03i mad: %7.4f fad: %7.4f" % (epoch, args.max_epoch, mad, fad))      

    epoch+=1

    if epoch==1:
      torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')

    if epoch%args.print_step==0:         
      print("Epoch: %03i/%03i  Training time: %03.4f  Loss: %03.4f  Avg. Loss: %03.4f" % (epoch, args.max_epoch, time.time()-e_t0, epoch_loss, epoch_loss / len(tr_dl)))

  print("Total training time: %07.4f" % (time.time()-t0))
  
  return mad





if __name__=='__main__':
  study = optuna.create_study()
  study.optimize(objective, n_trials=10)

  pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
  complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial = study.best_trial
  print("  Value: ", trial.value)

  print("  Params: ")
  for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

