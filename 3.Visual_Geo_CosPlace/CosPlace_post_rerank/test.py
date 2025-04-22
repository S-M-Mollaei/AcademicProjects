
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
        
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    
    database_utm = eval_ds.get_database_utm()
    logging.debug('shape of database utm is  {} :'.format(database_utm.shape))
    def re_ranking(preds,database_utm,distance,min_per,N):
            if N ==20:
              return(preds)
            else:
              if N==1:
                M=5
              if N==5:
                M=10
              if N == 10:
                M=20
              re_ranked = False
              preds = preds[:M]
              pred_utm1  = database_utm[preds]
              clustering1 = DBSCAN(eps=distance, min_samples=2).fit(pred_utm1)
              labels1 =  clustering1.labels_ 
              available_groups = set( labels1) -{-1}
              for group_ in  available_groups:
                group_size = np.count_nonzero(labels1 ==group_)
                if group_size > min_per* len(preds):
                  indices =  np.where(labels1 == group_)[0]
                  pred_1_0 = preds[indices]
                  pred_1_1 =  np.delete(preds, indices)
                  pred_new = np.concatenate(( pred_1_0,pred_1_1))
                  re_ranked = True

              if (re_ranked==False):
                  pred_new = preds

              return(pred_new)



    def remove_reduntant(preds,database_utm,distance,N,print_ =False):
            if N==1:
              return (preds)
            else:
              if N==5:
                M = 10 
              if N ==10:
                M=20
              else:
                M= 20
              preds = preds[:M]
              pred_utm1  = database_utm[preds]
              clustering1 = DBSCAN(eps=distance, min_samples=2).fit(pred_utm1)
              labels1 =  clustering1.labels_ 
              available_groups = set( labels1) -{-1}
              pred_new = np.copy(preds)
              for group_ in  available_groups:
                indices =  np.where(labels1 == group_)[0]
                #  we should sort this!! 
                indices.sort()
                indices = indices[::-1]
                if len(indices) >1:
                  
                  for ind in indices[:-1]:
                          try:
                            if len(pred_new)> N:
                              pred_new =  np.delete(pred_new,ind )
                            if print_:                                
                                  print('#############deleted##############')
                                  print(pred_new)
                                  print(ind)

                          except:
                            if print_:
                              print('********could not delete*****')
                              print(pred_new)
                              print(ind)
                del indices
              return(pred_new)            
    Rec ={}
    temp = 0
    for R_i in range(1,21):
      Rec['recalls'+str(R_i)] = np.zeros(len(RECALL_VALUES))    
    for query_index, preds in enumerate(predictions):
        check ={}
        for mm in range(1,21):
                  check[mm]= True
        for i, n in enumerate(RECALL_VALUES):
          ####################### re ranking ###################################### 
            preds1 = re_ranking(preds,database_utm,5,min_per= 0.5,N=n)
            preds2 = re_ranking(preds,database_utm,25,min_per= 0.6,N=n)
            preds3 = re_ranking(preds,database_utm,25,min_per= 0.4,N=n)
            preds4 = re_ranking(preds,database_utm,250,min_per= 0.6,N=n)
            preds5 = re_ranking(preds,database_utm,250,min_per= 0.8,N=n)
            preds6 = re_ranking(preds,database_utm,50,min_per=0.7,N=n)
            preds7 = re_ranking(preds,database_utm,2500,min_per=0.8,N=n)
            preds8 = re_ranking(preds,database_utm,100,min_per=0.8,N=n)
            preds9 = re_ranking(preds,database_utm,10,min_per=0.6,N=n)
            preds10 = re_ranking(preds,database_utm,25,min_per=0.8,N=n)

            if np.any(np.in1d(preds1[:n], positives_per_query[query_index])):
                if check[1]:
                    Rec['recalls1'][i:] += 1
                    check[1] = False
            if np.any(np.in1d(preds2[:n], positives_per_query[query_index])):
                if check[2]:
                    Rec['recalls2'][i:] += 1
                    check[2] = False
            if np.any(np.in1d(preds3[:n], positives_per_query[query_index])):
                if check[3]:
                    Rec['recalls3'][i:] += 1
                    check[3] = False      
            if np.any(np.in1d(preds4[:n], positives_per_query[query_index])):
                  if check[4]:
                    Rec['recalls4'][i:] += 1
                    check[4] = False 
            if np.any(np.in1d(preds5[:n], positives_per_query[query_index])):
                  if check[5]:
                    Rec['recalls5'][i:] += 1
                    check[5] = False      
            if np.any(np.in1d(preds6[:n], positives_per_query[query_index])):
                  if check[6]:
                    Rec['recalls6'][i:] += 1
                    check[6] = False

            if np.any(np.in1d(preds7[:n], positives_per_query[query_index])):
                  if check[7]:
                    Rec['recalls7'][i:] += 1
                    check[7] = False
                
            if np.any(np.in1d(preds8[:n], positives_per_query[query_index])):
                  if check[8]:
                    Rec['recalls8'][i:] += 1
                    check[8] = False
            if np.any(np.in1d(preds9[:n], positives_per_query[query_index])):
                  if check[9]:
                    Rec['recalls9'][i:] += 1
                    check[9] = False 

            if np.any(np.in1d(preds10[:n], positives_per_query[query_index])):
                  if check[10]:
                    Rec['recalls10'][i:] += 1
                    check[10] = False                                                   


        ########################### post processing ################################



            preds11 = remove_reduntant(preds,database_utm,5,N=n)
            preds12 = remove_reduntant(preds,database_utm,10,N=n)
            preds13 = remove_reduntant(preds,database_utm,20,N=n)
            preds14 = remove_reduntant(preds,database_utm,25,N=n)            
            preds15 = remove_reduntant(preds,database_utm,15,N=n)
            preds16 = remove_reduntant(preds,database_utm,50,N=n)
            preds17 = remove_reduntant(preds,database_utm,100,N=n)
            preds18 = remove_reduntant(preds,database_utm,500,N=n)
            preds19 = remove_reduntant(preds,database_utm,1000,N=n)
            preds20 = remove_reduntant(preds,database_utm,2000,N=n)
            if np.any(np.in1d(preds11[:n], positives_per_query[query_index])):
                  if check[11]:
                    Rec['recalls11'][i:] += 1
                    check[11] = False
            if np.any(np.in1d(preds12[:n], positives_per_query[query_index])):
                  if check[12]:
                    Rec['recalls12'][i:] += 1
                    check[12] = False
            if np.any(np.in1d(preds13[:n], positives_per_query[query_index])):
                  if check[13]:
                    Rec['recalls13'][i:] += 1
                    check[13] = False
            if np.any(np.in1d(preds14[:n], positives_per_query[query_index])):
                  if check[14]:
                    Rec['recalls14'][i:] += 1
                    check[14] = False

            if np.any(np.in1d(preds15[:n], positives_per_query[query_index])):
                  if check[15]:
                    Rec['recalls15'][i:] += 1
                    check[15] = False
            if np.any(np.in1d(preds16[:n], positives_per_query[query_index])):
                  if check[16]:
                    Rec['recalls16'][i:] += 1
                    check[16] = False
            if np.any(np.in1d(preds17[:n], positives_per_query[query_index])):
                  if check[17]:
                    Rec['recalls17'][i:] += 1
                    check[17] = False    

            if np.any(np.in1d(preds18[:n], positives_per_query[query_index])):
                if check[18]:
                  Rec['recalls18'][i:] += 1
                  check[18] = False
            if np.any(np.in1d(preds19[:n], positives_per_query[query_index])):
                  if check[19]:
                    Rec['recalls19'][i:] += 1
                    check[19] = False
            if np.any(np.in1d(preds20[:n], positives_per_query[query_index])):
                  if check[20]:
                    Rec['recalls20'][i:] += 1
                    check[20] = False


        


        

    recalls  = np.zeros(len(RECALL_VALUES))
    for R_i in range(1,21):
      Rec['recalls'+str(R_i)] = Rec['recalls'+str(R_i)]/eval_ds.queries_num * 100
      recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, Rec['recalls'+str(R_i)])])
      print('recalls'+str(R_i) )
      print(recalls_str) 
      if  Rec['recalls'+str(R_i)][0]>recalls[0]:
        recalls = Rec['recalls'+str(R_i)]



    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str
