
"""
This code takes a distorted image and undistorts it.
The image should be converted to a torch.tensor for the undistortion.
"""

import torch
import torchvision
from PIL import Image
import torch.nn.functional as F

# Open the distorted image with PIL
img = Image.open("distorted_image.jpg")
img.show()
# Convert PIL image to torch.tensor
tensor = torchvision.transforms.ToTensor()(img)
assert tensor.shape == torch.Size([3, 512, 512])
# The tensor has shape [3, 512, 512], we need to add a dimension at the beginning
tensor = tensor.reshape(1, 3, 512, 512)

# Some cool functions to undistort your image
undistortion_tensor = torch.load("grid.torch")
tensor = F.grid_sample(tensor, undistortion_tensor)

# Remove the extra dimension
tensor = tensor.reshape(3, 512, 512)
# Convert back to PIL image, so we can visualize it
img = torchvision.transforms.ToPILImage()(tensor)
img.show()

