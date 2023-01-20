import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from src.Test_Functions.test_dataloader import test_dataset
from src.Model.RACOD import RACOD
from src.Datasets.datasets_polyp import dataset_creation
from functools import partial
from Evaluation_Tools.evaluation_metrics import MAE, WeightedFmeasure, Emeasure, Smeasure

parser = argparse.ArgumentParser()
parser.add_argument('--evalsize', type=int, default=408, help='the image input size')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='init learning rate, try `lr=0.000006`')
parser.add_argument('--model_path', type=str,
                    default='Produced_Weights/RACOD/RACOD_checkpoint.pt')

evaluation_args = parser.parse_args("")

dictionary_of_evaluation = dataset_creation()

model = RACOD(
    img_size=352,
    embed_dims=[64, 128, 320, 512],
    num_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    depths=[3, 8, 27, 3],
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0,
    drop_path_rate=0.1,
    decoder_channels=768,
    num_classes=1
).cuda()

#LOAD_PRETRAINED_WEIGHTS = False
LOAD_PRETRAINED_WEIGHTS = True

if LOAD_PRETRAINED_WEIGHTS:
    optimizer = torch.optim.Adam(model.parameters(), evaluation_args.lr, weight_decay=evaluation_args.lr)
    scaler = torch.cuda.amp.GradScaler()
    checkpoint = torch.load(evaluation_args.model_path)
    # load state_dict from trained_weights
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    print('[INFO] Parameters Of Model Loaded From Pretrained Weights')
    model.eval()
    print('[INFO] Model Ready For Evaluation \n')
else:
    model.eval()
    print('[INFO] Model Ready For Evaluation \n')


for i in range(0, len(dictionary_of_evaluation), 2):
    dataset = list(dictionary_of_evaluation.items())[i:i+2]
    num_holder = len(dataset[0][0]) - len('evaluate_test_images')
    dataset_name = dataset[0][0][:num_holder - 1].upper()
    # define metric functions
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    _MAE = MAE()

    for i in tqdm(range(len(dataset[0][1]))):
        test_loader = test_dataset(dataset[0][1][i], dataset[1][1][i], testsize=evaluation_args.evalsize)
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
        # metrics
        WFM.step(pred=output_mask, gt=gt)
        SM.step(pred=output_mask, gt=gt)
        EM.step(pred=output_mask, gt=gt)
        _MAE.step(pred=output_mask, gt=gt)
        
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = _MAE.get_results()['mae']
    e_measure_adp = em['adp'].mean()
    e_measure_mean = em['curve'].mean()
    
    print(f'[Testing Done In {dataset_name} Dataset] Mean Absolute Error => {mae:.3f}')
    print(f'[Testing Done In {dataset_name} Dataset] Weighted F-measure => {wfm:.3f}')
    print(f'[Testing Done In {dataset_name} Dataset] S-measure => {sm:.3f}')
    print(f'[Testing Done In {dataset_name} Dataset] E-measure Adp => {e_measure_adp:.3f}')
    print(f'[Testing Done In {dataset_name} Dataset] E-measure Mean => {e_measure_mean:.3f}')
    print(65*'*')


