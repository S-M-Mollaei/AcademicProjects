import sys
import torch
import logging
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import test
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.domain_dataset import domainDataset 
torch.backends.cudnn.benchmark = True  # Provides a speedup
from torch.utils.data import DataLoader, Dataset
args = parser.parse_arguments(is_training=True)

start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim,args.alpha_domain,GRL_output =True)
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)
# groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
#                        current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
current_group_num = 0
# dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
#                                             batch_size=args.batch_size, shuffle=True,
#                                             pin_memory=(args.device == "cuda"), drop_last=True)

dataloader = TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=0, min_images_per_class=args.min_images_per_class)

test_ds_dataloader = DataLoader(dataloader, args.domain_batch_size, shuffle=True)

# test_ds_dataloader = DataLoader(test_ds , args.domain_batch_size, shuffle=True)
domain_ds = domainDataset(args,args.domain_data)
domain_ds_dataloader = DataLoader(domain_ds, args.domain_batch_size, shuffle=True)

tot = 0
cor = 0
for images, indices in tqdm(domain_ds_dataloader, ncols=100):
  images = images.to(args.device)
  _,domain_output  = model(images)
  pred = torch.argmax(domain_output, dim=1).to("cpu") 
  
  domain_label = torch.zeros(domain_output.shape[0])
             
  domain_label[:] = 1
  tot += domain_label.size(0)
  cor += (domain_label ==  pred).sum().item()

acc = 100 * cor / tot   
print('domain accuracy in target dataset is: {}'.format(acc)) 


tot = 0
cor = 0
for images, indices,_ in tqdm(test_ds_dataloader, ncols=100):

  images = images.to(args.device)
  _,domain_output  = model(images)
 
  pred = torch.argmax(domain_output, dim=1).to("cpu") 

  domain_label = torch.zeros(domain_output.shape[0])   
          
  domain_label[:] = 0 

  tot += domain_label.size(0)
  cor += (domain_label ==  pred).sum().item()

acc = 100 * cor / tot   
print('domain accuracy in source dataset is: {}'.format(acc)) 




