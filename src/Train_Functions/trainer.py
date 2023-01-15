import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

def structure_loss(pred, mask):
    
    # BCE loss
    k = nn.Softmax2d()
    weit = torch.abs(pred - mask)
    weit = k(weit)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # IOU loss
    smooth = 1
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + smooth) / (union - inter + smooth)

    return (wbce + wiou).mean()

def trainer(train_loader, model, optimizer, epoch, opt, total_step, scaler):
    all_dataset_loss_top = 0
    all_dataset_loss_top_bottom = 0
    iterations = 0
    for step, data_pack in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        images, gts = data_pack
        # Variables are just wrappers for the tensors so we can easily auto compute the gradients.
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output_mask_top, output_mask_top_bottom = model(images)
            assert output_mask_top.dtype is torch.float16, "float16 Error From Model During Training"
            assert output_mask_top_bottom.dtype is torch.float16, "float16 Error From Model During Training"
            loss_top = structure_loss(output_mask_top, gts)
            loss_top_bottom = structure_loss(output_mask_top_bottom, gts)
            loss_total = loss_top + loss_top_bottom
            
            assert loss_total.dtype is torch.float32, "float32 Error From Optimizer During Training"
        
        scaler.scale(loss_total).backward()
    
        #scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        scaler.step(optimizer)
        scaler.update()
        
        all_dataset_loss_top += loss_top.data
        all_dataset_loss_top_bottom += loss_top_bottom.data
        iterations += 1
        
        if step % 10 == 0 or step == total_step:
            print('[Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_Top: {:.4f} Loss_Top_Bottom: {:0.4f}]'
                  .format(epoch, opt.epoch, step, total_step, loss_top.data, loss_top_bottom.data))
            print(f'So far we have {all_dataset_loss_top/iterations:.4f} --- {all_dataset_loss_top_bottom/iterations:.4f}')

    average_loss_num_top = all_dataset_loss_top/iterations
    average_loss_num_top_bottom = all_dataset_loss_top_bottom/iterations
    print(f'[End of Epoch: {epoch}] => [Avr.Loss From Top Features => {average_loss_num_top:.4f}]')
    print(f'[End of Epoch: {epoch}] => [Avr.Loss From All  Features => {average_loss_num_top_bottom:.4f}]')
    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    # save our model+opt+scale -> state
    checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict()
    }
    torch.save(checkpoint, save_path + 'RACOD_checkpoint.pt')