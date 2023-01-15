import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class CamObjDataset(data.Dataset):
    def __init__(self, train_images, train_images_gt, trainsize):
        self.trainsize = trainsize
        self.images = train_images
        self.gts = train_images_gt
        # train_images and train_images_gt have been initialized in the start of the notebook
        self.size = len(self.images)
        # all the transforms in the transforms.Compose are applied to the input one by one
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        gt = Image.open(self.gts[index])
        image = image.convert('RGB')
        gt = gt.convert('L')
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def __len__(self):
        return self.size
    
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader