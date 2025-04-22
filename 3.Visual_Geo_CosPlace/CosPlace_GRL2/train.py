
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from torch.utils.data import DataLoader
import test
import util
import parser
import commons
import cosface_loss
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.domain_dataset import domainDataset 
torch.backends.cudnn.benchmark = True  # Provides a speedup
import math

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim,args.alpha_domain,GRL_output =True)
print('\n \n alpha in domain adatation is{}: '.format(args.alpha_domain))
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
domain_criterion =  torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
#######################################################
domain_test_ds = domainDataset(args,args.domain_data)
domain_test_dataloader = DataLoader(domain_test_ds, args.domain_batch_size, shuffle=True)
print('-------checking  domain dataloader------' )
tem_img, tem_label = next(iter(domain_test_dataloader ))
print('len domain_ds_dataloader:',len(domain_test_dataloader ))
print('images shape in dataloader is :{} and label shape is: {}'.format(tem_img.shape, tem_label.shape))
#######################################################
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)

target_ds = TestDataset(args.target_data, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)

logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)
    model = model.train()
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    epoch_domain_losses = np.zeros((0, 1), dtype=np.float32)
    landa = 2/(1+math.exp(-0.5*epoch_num))-1
    print('landa is {}'.format(landa))

    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _ ,label_d1,domain_img,label_d2= next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        label_d1,domain_img,label_d2 = label_d1.to(args.device),domain_img.to(args.device),label_d2.to(args.device)

        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()
        
        if not args.use_amp16:
            descriptors,domain_output  = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            # loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            #  domain loss
            
    

            loss_domain1 = domain_criterion(domain_output,label_d1)          
 
            _,domain_output  = model(domain_img)
            loss_domain2 = domain_criterion(domain_output,label_d2)
            loss_domain = loss_domain1 +loss_domain2
            loss_do = loss_domain.item()
            total_loss = landa*loss_domain +loss      

              
            epoch_domain_losses = np.append(epoch_domain_losses,loss_do )            
            total_loss.backward()
            del loss, output, images,loss_domain2,domain_output
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            assert False, f"domain adaptaion was not modified for this setting ,do not use amp"

    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}, "f"domain_losses = {epoch_domain_losses.mean():.4f}")
    tot = 0
    cor = 0
    
    for iteration in tqdm(range(len(domain_test_dataloader )), ncols=100):
      images, targets= next(iter(domain_test_dataloader))
      images, targets = images.to(args.device), targets.to(args.device)
      _,domain_output  = model(images)
      pred = torch.argmax(domain_output, dim=1).to("cpu") 
      real_label = torch.argmax(targets, dim=1).to("cpu") 
      
      tot += real_label.size(0)
      cor += (real_label ==  pred).sum().item()
      acc = 100 * cor / tot  

    

    print('***********evaluating target domain*********** ')
   
    recalls_target, recalls_str_target = test.test(args, target_ds, model)
    logging.info(f"{target_ds}: {recalls_str_target}")
        
    print('to check : pred.shape:{} pred :{} '.format(pred.shape,pred))
    print('to check : real_label :{} '.format( real_label))
    print('\n target labels were predicted by acc equal to : {}'.format(acc)) 

    print('***********evaluating source domain*********** ')
    recalls, recalls_str = test.test(args, val_ds, model,source=True)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    print('*********************************************** ')
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)

    if args.domain_batch_size >0 :

        util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder,is_domain= True)

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")
