from matplotlib import pyplot as plt
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from src.Model.RACOD import RACOD
from functools import partial
from src.Datasets.datasets_polyp import dataset_creation
from src.Train_Functions.train_dataloader import get_loader
from src.Train_Functions.trainer import trainer

print("Torch Version is: " + torch.__version__)
# GPU needs to be enabled for our model to be executed
print("Cuda Available" if torch.cuda.is_available() else "Cuda Not Available")
print(f"Cuda Version is: {torch.version.cuda if torch.cuda.is_available() else False}")
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

train_images, train_images_gt_object, _ = dataset_creation()

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=39,
                    help='epoch number, default=39')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='init learning rate, try `lr=0.000006`')
parser.add_argument('--batchsize', type=int, default=6,
                    help='training batch size')
# for polyp we use at the moment size 352x352 since the images in the dataset have similar resolution
parser.add_argument('--trainsize', type=int, default=352,
                    help='the size of training image, try small resolutions for speed (like 352 or 384)')
parser.add_argument('--gpu', type=int, default=0,
                    help='choose which gpu you use')

parser.add_argument('--save_model', type=str, default='Produced_Weights/RACOD/Polyp/')
parser.add_argument('--train_img_dir', type=str, default='Training Directory From Kvasir-SEG + CVC-ClinicDB')
parser.add_argument('--train_gt_dir', type=str, default='GT Objects Directory From Kvasir-SEG + CVC-ClinicDB')
args = parser.parse_args("")

model = RACOD(
    img_size=args.trainsize,
    embed_dims=[64, 128, 320, 512],
    num_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #depths=[3, 6, 40, 3],
    depths=[3, 8, 27, 3],
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0,
    drop_path_rate=0.0,
    decoder_channels=768,
    num_classes=1
).cuda()

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=6, eta_min=0.000006, last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

train_loader = get_loader(train_images, train_images_gt_object, batchsize=args.batchsize,
                          trainsize=args.trainsize, num_workers=12)
                          
total_step = len(train_loader)

print(f'\n', '*' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch_Size: {}\n"
                    "Image_Size: {}\nTraining_Epochs: {}\nTraining_Save: {}\nTotal_Num_Of_Images: {}\n".
                    format(args.train_img_dir, args.train_gt_dir, args.lr, args.batchsize, 
                    args.trainsize, args.epoch, args.save_model, len(train_images)), '*' * 30, f'\n')

# initialize GradScaler instance since we are using Automatic Mixed Precision
scaler = torch.cuda.amp.GradScaler()
lrs = []

# total of args.epoch for training
for epoch_iter in tqdm(range(1, args.epoch + 1)):
    model.train(True)
    trainer(train_loader=train_loader, model=model,
            optimizer=optimizer, epoch=epoch_iter,
           opt=args, total_step=total_step, scaler=scaler)
    if epoch_iter >= 15:
        scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

plt.plot(lrs)
plt.title("Learning Rate Strategy")
plt.xlabel("Epochs")
plt.ylabel("LR Value")
plt.show()