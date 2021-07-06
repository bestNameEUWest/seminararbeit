# system
import argparse
import os
import time
import math
from datetime import datetime

# external
import numpy as np
import pandas as pd
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
  parser.add_argument('--raw_dataset_folder',type=str,default='raw_data')
  parser.add_argument('--obs',type=int,default=8)
  parser.add_argument('--preds',type=int,default=12)
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
  parser.add_argument('--max_epoch',type=int, default=1000)
  parser.add_argument('--batch_size',type=int,default=512)   
  parser.add_argument('--run_info', type=str, default=None)
  parser.add_argument('--steps',type=int, default=5) 

  # default hyperparameters in case we do not use optuna
  parser.add_argument('--emb_size',type=int,default=32) 
  parser.add_argument('--heads',type=int, default=2) 
  parser.add_argument('--layers',type=int,default=2) 
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
    baselineUtils.format_raw_dataset(args.raw_dataset_folder , args.dataset_name, args.dataset_folder)
  
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

  # calculate the mean and std of our datasets
  if len(input_means) is 1: # if it is just one dataset
    # all input features
    input_mean=input_means[0]
    input_std=input_stds[0]        
    # target coordinate distances
    target_mean=target_means[0]
    target_std=target_stds[0]
  else: 
    input_mean=torch.stack(input_means).mean(0)
    input_std=torch.stack(input_stds).std(0)        

    target_mean=torch.stack(target_means).mean(0)
    target_std=torch.stack(target_stds).std(0)

  return input_mean, input_std, target_mean, target_std

cols = ''
def save_log_info(args, info):
  cols = "_".join(args.col_names[2:])
  path = f'logs/{cols}/{info.date[0]}'
    
  try:
    os.makedirs(path)
  except:
    pass

  info.to_csv(f'{path}/run_log.csv', index=False, header=True)

################################################################################
######################## more important learning stuff #########################
################################################################################

save_time = ''
# objective function to minimize for optuna
def objective(trial):
  args = argparser_function()
  feature_count = len(args.col_names) - 2
    
  #print(f'Steps: {args.steps}')
  train_dataset, val_dataset, test_dataset = dataset_and_folder_prep(args)

  input_mean, input_std, target_mean, target_std = means_and_stds(train_dataset, feature_count)
  scipy.io.savemat(f'models/{args.name}/norm.mat',{'mean':input_mean.cpu().numpy(),'std':input_std.cpu().numpy()})
  
  device=torch.device("cuda")

  if args.cpu or not torch.cuda.is_available():
    device=torch.device("cpu")
  args.verbose=True    

  args.layers = trial.suggest_int('layers', 1, 8) 
  args.emb_size = trial.suggest_int('emb_size', 16, 512)
  args.heads = trial.suggest_int('heads', 1, 8)
   
  # I used this for some quick tests regarding available memory size
  # args.layers = 8
  # args.emb_size = 512
  # args.heads = 2
    

  tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
   

  
  model=individual_TF.IndividualTF(feature_count, 3, 3, N=args.layers, d_model=args.emb_size,
                                   d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)                              
  
  optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  now = datetime.now()
  save_time = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss")   

  save_comment = f'{save_time}_he={args.heads}_la={args.layers}_es={args.emb_size}_ba={args.batch_size}'
  train_comment = (f'heads={args.heads} ' +
                    f'layers={args.layers} ' +
                    f'emb_size={args.emb_size} ' +
                    f'batch_size={args.batch_size} ')

  
  if args.run_info is not None:        
    save_comment = f'{args.run_info}_{save_comment}'
    train_comment = f'{args.run_info} {train_comment}'        
  
  cols = "_".join(args.col_names[2:])
  log=SummaryWriter(log_dir=f'runs/{cols}/{save_comment}')
   
  df_column_names = ["date", "current_epoch", "max_epoch", "training_time", "total_loss", "agv_loss", "mad", "fad",
                  "layers", "emb_size", "heads", "dropout",
                  "x", "y", "heading", "width", "length", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]

  log_data = pd.DataFrame(columns=df_column_names)
  next_row = {"date": save_time, "current_epoch": 0, "max_epoch": args.max_epoch, "training_time": 0.0, "total_loss": 0.0, "agv_loss": 0.0, "mad": 0.0, "fad": 0.0,
                        "layers": args.layers, "emb_size": args.emb_size, "heads": args.heads, "dropout": args.dropout}
      
  df_poss_cols= ["x", "y", "heading", "width", "length", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
  contain_vals = []
  for df_poss_col in df_poss_cols:
    if df_poss_col in args.col_names:
      contain_vals.append(1)
    else:
      contain_vals.append(0)
  dic = dict(zip(df_poss_cols, contain_vals))
  next_row.update(dic)

  print(f'Training for: {train_comment}')
  t0 = time.time()
  epoch=0
  epoch_check_freq = 50 # our k
  val_rel_err_thresh = 0.01
 
  val_err_min = math.inf  
  val_rel_err = math.inf

  val_errs = []
  
  while val_rel_err > val_rel_err_thresh and epoch < args.max_epoch:
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

    epoch+=1
    log.add_scalar(f'Loss/train/', epoch_loss / len(tr_dl), epoch)

    min_mad = math.inf
    with torch.no_grad():
      model.eval()

      val_loss=0
      step=0
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
      if mad < min_mad:
        min_mad = mad
      val_errs.append(mad)

      train_time = f'{time.time()-e_t0:03.4f}'
      avg_loss = epoch_loss / len(tr_dl)
      update = {"current_epoch": epoch, "training_time": train_time, "total_loss": epoch_loss, "agv_loss": avg_loss, "mad": mad, "fad": fad}
      next_row.update(update)
      log_data = log_data.append(next_row, ignore_index=True)
  
      if epoch==1:
        torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')

      if epoch%args.print_step==0:         
        print(f"Epoch: {epoch:03d} Training time: {train_time}  Loss: {epoch_loss:03.4f}  Avg. Loss: {avg_loss:03.4f} MAD: {mad:03.4f} FAD: {fad:03.4f}") 
      
      if epoch%epoch_check_freq==0:
        if epoch > epoch_check_freq:
          curr_val_err_min = min(val_errs)
          val_rel_err = (val_err_min - curr_val_err_min)/val_err_min
          print(f'val_err_min: {val_err_min:03.4f} curr_val_err_min: {curr_val_err_min:03.4f} val_rel_err: {val_rel_err:03.4f}')
        
        val_err_min = min(val_errs)
        val_errs = []        

        if val_rel_err < val_rel_err_thresh:
          print(f'Reached less than {val_rel_err_thresh} val error change over last {epoch_check_freq} epochs. Stopping training')
        if epoch >= args.max_epoch:
          print(f'Reached max epoch of {args.max_epoch}. Stopping training.')

  total_train_time = time.time()-t0
  print(f"Total training time: {total_train_time:07.4f}")

  log_data['total_train_time'] = total_train_time
  save_log_info(args, log_data)

  return min_mad


if __name__=='__main__':
  # set the hyperparam search space, here we just took a set of one value per parameter
  search_space = {"layers": [8],
                  "emb_size": [512],
                  "heads": [2]}
  study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
  study.optimize(objective, n_trials=1)

  pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
  complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial = study.best_trial
  print("  Value: ", trial.value)

  trials_df = study.trials_dataframe()
  trials_df.to_csv(f'logs/{cols}/{save_time}_trials_df.csv')

  print("  Params: ")
  for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

