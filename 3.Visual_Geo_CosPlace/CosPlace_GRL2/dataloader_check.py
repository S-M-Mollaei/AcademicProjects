import parser
from torch.utils.data import DataLoader, Dataset
from datasets.domain_dataset import domainDataset 
args = parser.parse_arguments()

domain_ds = domainDataset(args,args.domain_data)
domain_ds_dataloader = DataLoader(domain_ds, args.domain_batch_size, shuffle=True)
images, targets= next(iter(domain_ds_dataloader))

print('images shape is: ',images.shape)

