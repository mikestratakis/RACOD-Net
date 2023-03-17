import torch
import os
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from os import path
from functools import partial
from src.Model.RACOD import RACOD
from src.Datasets.datasets import dataset_creation
from src.Test_Functions.test_dataloader import test_dataset
from Evaluation_Tools.evaluation_metrics import MAE

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=456, help='the image input size')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='init learning rate, try `lr=0.000006`')
parser.add_argument('--model_path', type=str,
                    default='Produced_Weights/RACOD/COD/RACOD_checkpoint.pt')
parser.add_argument('--test_save', type=str,
                    default='Results/Test/')

testing_args = parser.parse_args("")
save_path = testing_args.test_save
os.makedirs(save_path, exist_ok=True)

_, _, test_images, test_images_gt_object, _ = dataset_creation(show=False)

model = RACOD(
    img_size=testing_args.testsize,
    embed_dims=[64, 128, 320, 512],
    num_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    depths=[3, 8, 27, 3],
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0,
    drop_path_rate=0.0,
    decoder_channels=768,
    num_classes=1
).cuda()

print(f'\n', '*' * 30, "\nTesting_Save_Directory: {}\nTotal_Num_Of_Images_To_Be_Saved: {}\n".
                    format(testing_args.test_save, len(test_images)), '*' * 30, f'\n')

#LOAD_PRETRAINED_WEIGHTS = False
LOAD_PRETRAINED_WEIGHTS = True

if LOAD_PRETRAINED_WEIGHTS:
    optimizer = torch.optim.Adam(model.parameters(), testing_args.lr, weight_decay=testing_args.lr)
    scaler = torch.cuda.amp.GradScaler()
    checkpoint = torch.load(testing_args.model_path)
    # load state_dict from trained_weights
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    print('[INFO] Parameters Of Model Loaded From Pretrained Weights')
    model.eval()
    print('[INFO] Model Ready For Testing \n')
else:
    model.eval()
    print('[INFO] Model Ready For Testing \n')

for i in tqdm(range(len(test_images))):
    _MAE = MAE()
    test_loader = test_dataset(test_images[i], test_images_gt_object[i], testsize=testing_args.testsize)
    image, gt = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    image = image.cuda()
    # inference
    _, output_mask = model(image)
    # reshape and normalize
    output_mask = F.interpolate(output_mask, size=gt.shape, mode='bilinear', align_corners=False)
    output_mask = torch.sigmoid(output_mask)
    output_mask = np.uint8(output_mask.squeeze().detach().cpu().numpy()*255)
    output_mask = output_mask / (np.linalg.norm(output_mask) +  1e-8)
    _MAE.step(pred=output_mask, gt=gt)
    temp = _MAE.get_results()['mae']
    #print(f'[Test] Image_Name: {test_images[i][21:]} ({i}/{len(test_images)}), MAE: {temp:.3f}')
    fig, ax_main = plt.subplots(1, 3, figsize = (14, 8))
    ax_main[0].title.set_text(f"Prediction Output with MaE => {temp:.3f}")
    ax_main[0].imshow(output_mask, cmap='gist_gray')
    ax_main[1].title.set_text(f"Ground Truth")
    ax_main[1].imshow(gt, cmap='gist_gray')
    ax_main[2].title.set_text(f"Image: {test_images[i][23:]}")
    ax_main[2].imshow(Image.open(test_images[i]))
    fig.savefig(path.join(save_path, "{0}.{1}".format(test_images[i][23:-4].replace("/","_"), 'jpg')))
    plt.close(fig)

print('[INFO] Testing Done')