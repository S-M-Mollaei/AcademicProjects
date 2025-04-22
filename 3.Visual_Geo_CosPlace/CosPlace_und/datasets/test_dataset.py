
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision
import torch.nn.functional as F


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)
        
        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
        self.undistortion_tensor = torch.load("/content/drive/MyDrive/Class_Vis_Geo/CosPlace_und/datasets/grid.torch")
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)


        if index <self.database_num:

          if index%3000==0:
            path = '/content/drive/MyDrive/Class_Vis_Geo/CosPlace_und/logs/test_set_sample/'+ str(index)+'distorted_'+'.jpg'
            pil_img.save(path)
            

          # Convert PIL image to torch.tensor
          tensor = torchvision.transforms.ToTensor()(pil_img)
          assert tensor.shape == torch.Size([3, 512, 512])
          # The tensor has shape [3, 512, 512], we need to add a dimension at the beginning
          tensor = tensor.reshape(1, 3, 512, 512)
          # Some cool functions to undistort your image
          
          tensor = F.grid_sample(tensor, self.undistortion_tensor)

          # Remove the extra dimension
          tensor = tensor.reshape(3, 512, 512)
          # Convert back to PIL image, so we can visualize it
          pil_img = torchvision.transforms.ToPILImage()(tensor)
          if index%3000==0:
            path = '/content/drive/MyDrive/Class_Vis_Geo/CosPlace_und/logs/test_set_sample/'+str(index)+'undistorted_'+'.jpg'
            pil_img.save(path)
            print('notice that undistortion is being applied')

        normalized_img = self.base_transform(pil_img)


        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query
