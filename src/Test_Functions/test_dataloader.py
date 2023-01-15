import torchvision.transforms as transforms
from PIL import Image

class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, test_images, test_images_gts, testsize):
        self.testsize = testsize
        self.images = test_images
        self.gts = test_images_gts
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.igt_transformndex = 0

    def load_data(self):
        image = Image.open(self.images)
        gts = Image.open(self.gts)
        image = image.convert('RGB')
        gts = gts.convert('L')
        image = self.transform(image).unsqueeze(0)
        return image, gts