import cv2
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms


# reverse normalize and convert to PIL
# if normalize and want to display normalize image
# c h w -> c h w


def visualizer(idx, image_path, save_path, mask, rec_diff, seg_target, heat_map=True, size=(256, 256)):
    save_path = save_path + '/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = Image.open(image_path)
    image_pil = image.resize(size)  # PIL

    mask_np = np.repeat(mask[..., None], 3, axis=2)
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

    B, C, H, W = rec_diff.shape  # 1, 576, 64, 64
    asm_rec = torch.sum(rec_diff, dim=1)  # 1, 64, 64 anomaly score map
    asm_rec = nn.functional.interpolate(asm_rec.unsqueeze(0),
                                        size=size[0],
                                        mode='bilinear',
                                        align_corners=True)  # 1, 1, 64, 64
    asm_rec = (asm_rec[0] - torch.min(asm_rec[0])) / (torch.max(asm_rec[0]) - torch.min(asm_rec[0]))
    asm_rec = asm_rec.data.cpu().numpy().astype(np.float32)  # 1, 64, 64
    asm_rec = asm_rec.repeat(3, axis=0).transpose(1, 2, 0)

    img_np = np.array(image_pil).astype(np.uint8)

    asm_rec = (asm_rec * 255).astype(np.uint8)
    if heat_map:
        asm_rec = cv2.applyColorMap(asm_rec, cv2.COLORMAP_JET)
        asm_rec = cv2.cvtColor(asm_rec, cv2.COLOR_BGR2RGB)
        # asm_rec = cv2.addWeighted(img_np, 0.7, asm_rec, 0.3, 0)
    asm_rec = Image.fromarray(asm_rec)

    asm_seg = ((np.repeat(seg_target[..., None], 3, axis=2)) * 255).astype(np.uint8)
    if heat_map:
        asm_seg = cv2.applyColorMap(asm_seg, cv2.COLORMAP_JET)
        asm_seg = cv2.cvtColor(asm_seg, cv2.COLOR_BGR2RGB)
        # asm_seg = cv2.addWeighted(img_np, 0.7, asm_seg, 0.3, 0)
    asm_seg = Image.fromarray(asm_seg)

    asm = Image.new('RGB', (256 * 4 + 20, 256), (255, 255, 255))
    asm.paste(image_pil, (0, 0))
    asm.paste(mask_pil, (256 + 5, 0))
    asm.paste(asm_rec, (256 * 2 + 10, 0))
    asm.paste(asm_seg, (256 * 3 + 15, 0))
    asm.save(save_path + str(idx) + '.png')


def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.256, 0.225], dtype=torch.float32)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())
    unnormalize_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return unnormalize_trans(img)


def visualize_and_save(idx, save_path, image_path, mask, score):
    save_path = save_path + 'images/'

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(256, 256)),
                                    transforms.ToTensor()])
    image = transform(image)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    toPIL = transforms.ToPILImage()
    '''
        save_path: str
        img: tensor, [1,3,h,w], normalize, 0~1, torch.float32
        mask: array, [h, w], 01, np.float32
        score: tensor, [1, 1, h, w], torch.float32
    '''

    img_pil = toPIL(image)
    img_pil = img_pil.resize((256, 256), Image.ANTIALIAS)

    mask_pil = toPIL((mask * 255).astype(np.uint8))

    score_nor = (score[0] - torch.min(score[0])) / (torch.max(score[0]) - torch.min(score[0]))
    score_nor = score_nor.data.cpu().numpy().astype(np.float32)
    # (1,h,w)->(3,h,w)->(h,w,3)
    score_nor = score_nor.repeat(3, axis=0).transpose(1, 2, 0)
    score_nor = (score_nor * 255).astype(np.uint8)
    score_nor_heatmap = cv2.applyColorMap(score_nor, cv2.COLORMAP_JET)
    img_np = np.array(img_pil).astype(np.uint8)
    heatmap_img = cv2.addWeighted(img_np, 0.7, score_nor_heatmap, 0.3, 0)

    score_nor_heatmap = cv2.cvtColor(score_nor_heatmap, cv2.COLOR_BGR2RGB)
    score_nor_heatmap = Image.fromarray(score_nor_heatmap)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap_img)

    image_combi = Image.new('RGB', (256 * 4 + 20, 256), (255, 255, 255))
    image_combi.paste(img_pil, (0, 0))
    image_combi.paste(mask_pil, (256 + 5, 0))
    image_combi.paste(score_nor_heatmap, (256 * 2 + 10, 0))
    image_combi.paste(heatmap_img, (256 * 3 + 15, 0))
    image_combi.save(save_path + str(idx) + '.jpg')