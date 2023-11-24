from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


data_path = '/media/18T/data_thesis/kognic/3302'

transform_img = transforms.Compose([
    transforms.Resize(size=700),
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
  root=data_path, transform=transform_img
)

image_data_loader = DataLoader(
  image_data, 
  # batch size is whole datset
  batch_size=len(image_data), 
  shuffle=False, 
  num_workers=0)

def mean_std(loader):
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)