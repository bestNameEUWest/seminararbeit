from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io


def create_dataset(dataset_folder, dataset_name, gt,horizon, features,delim="\t",train=True,eval=False,verbose=False, step=1):
  if train==True:
    data_type = "train"
    datasets_list = os.listdir(os.path.join(dataset_folder,dataset_name, data_type))
    full_dt_folder=os.path.join(dataset_folder,dataset_name, data_type)    
  if train==False and eval==False:
    data_type = "val"
    datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, data_type))
    full_dt_folder = os.path.join(dataset_folder, dataset_name, data_type)        
  if train==False and eval==True:
    data_type = "test"
    datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, data_type))
    full_dt_folder = os.path.join(dataset_folder, dataset_name, data_type)
    

  datasets_list=datasets_list
  data={}
  data_src=[]
  data_trg=[]
  data_seq_start=[]
  data_frames=[]
  data_dt=[]
  data_objs=[]

  val_src = []
  val_trg = []
  val_seq_start = []
  val_frames = []
  val_dt = []
  val_objs=[]


  for i_dt, dt in enumerate(datasets_list):
    raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim, na_values="?")
    raw_data = raw_data[features]
    #print(f'raw_data: \n{raw_data.head()}')
    raw_data.sort_values(by=[raw_data.columns[0], raw_data.columns[1]], inplace=True)
    inp,out,info=get_strided_data_clust(raw_data, gt, horizon, step)
    dt_frames=info['frames']
    dt_seq_start=info['seq_start']
    dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
    dt_objs=info['objs']   

    data_src.append(inp)
    data_trg.append(out)
    data_seq_start.append(dt_seq_start)
    data_frames.append(dt_frames)
    data_dt.append(dt_dataset)
    data_objs.append(dt_objs)

  data['src'] = np.concatenate(data_src, 0)
  data['trg'] = np.concatenate(data_trg, 0)
  data['seq_start'] = np.concatenate(data_seq_start, 0)
  data['frames'] = np.concatenate(data_frames, 0)
  data['dataset'] = np.concatenate(data_dt, 0)
  data['objs'] = np.concatenate(data_objs, 0)
  data['dataset_name'] = datasets_list

  mean= data['src'].mean((0,1))
  std= data['src'].std((0,1))

  
  return IndividualTfDataset(data, data_type, mean, std)



class IndividualTfDataset(Dataset):
  def __init__(self,data,name,mean,std):
    super(IndividualTfDataset,self).__init__()

    self.data=data
    self.name=name

    self.mean= mean
    self.std = std

  def __len__(self):
    return self.data['src'].shape[0]


  def __getitem__(self,index):
    return {'src':torch.Tensor(self.data['src'][index]),
            'trg':torch.Tensor(self.data['trg'][index]),
            'frames':self.data['frames'][index],
            'seq_start':self.data['seq_start'][index],
            'dataset':self.data['dataset'][index],
            'objs': self.data['objs'][index],
            }



def format_raw_dataset(raw_dataset_folder, dataset_name, target_dataset_folder):
  validation_ratio = 0.15

  test_dataset = '01_tracks.csv'
  relevant_cols =['frame', 'obj', 'x', 'y', 'heading', 'width', 'length', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']  # all relevant data
  datasets_list = [dir for dir in os.listdir(os.path.join(raw_dataset_folder, dataset_name))]
  os.makedirs(os.path.join(target_dataset_folder, dataset_name))      
  
  for (i_d, dataset) in enumerate(datasets_list):  
    data = pd.read_csv(os.path.join(raw_dataset_folder, dataset_name, dataset))
    data = data.rename(columns={'trackId': 'obj', 'xCenter': 'x', 'yCenter': 'y'})
    date = data[relevant_cols] 
    
    for set_type in ['train', 'val', 'test']:    
      try:
        path = os.path.join(target_dataset_folder, dataset_name, set_type)
        os.makedirs(path)   	
      except:
        pass

    if dataset == test_dataset:
      test_path = os.path.join(f'{target_dataset_folder}', dataset_name, 'test', f'{i_d:02d}_test.csv')
      data.to_csv(path_or_buf=test_path, index=False, sep='\t')
    else:
      size = data.shape[0]
      cutoff_num = int((1-validation_ratio)*size)
      data_train = data.head(cutoff_num)  
      data_val = data.tail(size - cutoff_num)  

      train_path = os.path.join(f'{target_dataset_folder}', dataset_name, 'train', f'{i_d:02d}_train.csv')
      val_path = os.path.join(f'{target_dataset_folder}', dataset_name, 'val', f'{i_d:02d}_val.csv')
      data_train.to_csv(path_or_buf=train_path, index=False, sep='\t')
      data_val.to_csv(path_or_buf=val_path, index=False, sep='\t')          


def is_data_prepared(save_dir, dataset_info):
  print('Checking if data is already prepared')
  save_file_name = 'info.pt'
  
  for (root, dirs, files) in os.walk(save_dir):
    for file in files:
      if save_file_name == file:
        save_file_path = os.path.join(root, save_file_name)        
        saved_info = torch.load(save_file_path)
        if saved_info == dataset_info:
          return True, root  
  return False, None

def get_strided_data_clust(dt, gt_size, horizon, step):
  inp_te = []
  dtt = dt.astype(np.float32)
  raw_data = dtt
  features_count = len(dt.columns) - 2 # features without 'frames' and 'obs'

  obj = raw_data.obj.unique()
  frame=[]
  obj_ids=[]
  for p in obj:
    for i in range(1+(raw_data[raw_data.obj == p].shape[0] - gt_size - horizon) // step):            
      frame.append(dt[dt.obj == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
      inp_te.append(raw_data[raw_data.obj == p].iloc[i * step:i * step + gt_size + horizon, 2:features_count+2].values)
      obj_ids.append(p)

  frames=np.stack(frame)
  inp_te_np = np.stack(inp_te)
  obj_ids=np.stack(obj_ids)

  # create "velocity" distance vectors, because it trains better this way
  #inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,features_count)),inp_te_np[:,1:,0:features_count] - inp_te_np[:, :-1, 0:features_count]),1)
  #inp_norm = np.concatenate((inp_te_np,inp_speed),2)

  inp_norm = inp_te_np
  #print(f'inp_norm: {inp_norm}')

  inp_mean=np.zeros(4)
  inp_std=np.ones(4)

  inp = inp_norm[:,:gt_size]
  out = inp_norm[:,gt_size:]
  info = {'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'objs':obj_ids}


  #print(f'inp: {inp[0]}')
  #print(f'out: {out[0]}')

  return inp, out, info


def distance_metrics(gt,preds):
  errors = np.zeros(preds.shape[:-1])
  for i in range(errors.shape[0]):
    for j in range(errors.shape[1]):
      errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
  return errors.mean(),errors[:,-1].mean(),errors
