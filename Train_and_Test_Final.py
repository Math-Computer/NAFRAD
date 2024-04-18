import os
import time
import datetime
import warnings
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils import *

warnings.filterwarnings('ignore')

torch.cuda.set_device(7)

if __name__ == '__main__':

    # * Statistic all results in an Excel
    DataExcel = {
        'category': ['carpet', 'grid', 'leather', 'tile', 'wood', 'Avg. Tex.',
                     'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                     'screw', 'toothbrush', 'transistor', 'zipper', 'Avg. Obj.', 'Avg.'],
        'image_aucroc': list(range(18)),
        'pixel_aucroc': list(range(18)),
        'pixel_ap': list(range(18)),
        'IoU(0.5)': list(range(18))
    }

    # * MVTec AD class names
    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    objects = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    no_rotation_category = ['capsule', 'metal_nut', 'pill', 'toothbrush', 'transistor']
    slight_rotation_category = ['cable', 'zipper']
    rotation_category = textures + ['bottle', 'hazelnut', 'screw']
    RES_Class = ['leather', 'tile', 'bottle', 'cable', 'metal_nut', 'pill', 'transistor', 'zipper']

    # * Train & test class names
    data_names = textures + objects

    # * Data paths
    mvtec_dir = './data/MVTecAD'
    dtd_dir = './data/dtd/images/'

    # * Save paths
    result_root = './temp/results/'
    log_root = './temp/logs/'

    # * Setups
    train_image_size = 256  # default 256
    test_image_size = 256  # default 256
    feature_size = 64  # default 64
    train_batch_size = 8  # default 8
    test_batch_size = 1  # default 1
    epochs = 700  # default 700
    feature_start = 0  # default 0
    feature_end = 2  # default 2
    log_val = 1  # default 1

    # * Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for name in data_names:

        if name in no_rotation_category:
            rotate_degree = 0
        if name in slight_rotation_category:
            rotate_degree = (-5, 5)
        if name in rotation_category:
            rotate_degree = (-90, 90)

        if name in RES_Class:
            rec_in_channels = 195
        else:
            rec_in_channels = 579

        class_path = os.path.join(mvtec_dir, name)  # ./data/MVTecAD/carpet

        result_path = result_root + name  # ./temp/results/carpet
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        log_path = log_root + name  # ./temp/logs/carpet
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # * Load training data
        train_dataset = MVTecDRAEMTrainDataset(class_path, dtd_dir, (256, 256), rotate_degree)
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

        # * Load test data
        test_dataset = MVTecDRAEMTestDataset(class_path, (256, 256))
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

        # * Define models
        if name in RES_Class:
            feature_extractor = Feature_Extractor_Resnet50(weights='IMAGENET1K_V1').to(device)
        else:
            feature_extractor = FeatureExtractionSwin(weights='swin-large_in21k-pre-3rdparty_in1k', device=device)
        feature_aggregator = FeatureAggregation(feature_size=feature_size).to(device)
        feature_reconstructor = EncoderDecoder(rec_in_channels).to(device)
        feature_segmentor = DiscriminativeSubNetwork(in_channels=2*rec_in_channels, out_channels=1).to(device)

        # * Define loss functions
        L2_loss = nn.MSELoss(reduction='mean')
        CE_loss = nn.CrossEntropyLoss()
        compute_energy = nn.MSELoss(reduction='none')

        # * Define optimizer
        optimizer = optim.Adam([
            {'params': feature_reconstructor.parameters(), 'lr': 1e-4, 'weight_decay': 1e-8},
            {'params': feature_segmentor.parameters(), 'lr': 1e-4, 'weight_decay': 1e-8}
        ])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400, 600], gamma=0.1, last_epoch=-1)

        writer = SummaryWriter(log_dir=log_path)

        iter_num = 0
        iaucroc = []
        paucroc = []
        pap = []
        piou = []
        max_ap = 0.0

        # * Start training
        for epoch in range(1, epochs + 1):

            feature_extractor.eval()
            feature_reconstructor.train()
            feature_segmentor.train()

            seg_loss_epoch = 0.0
            rec_loss_epoch = 0.0
            time_start = time.time()

            for i, data in enumerate(train_loader):
                # * Load image and mask
                normal_image = data['normal_image'].to(device)  # torch.float32 [b, 3, 256, 256]
                pseudo_image = data['pseudo_image'].to(device)  # torch.float32 [b, 3, 256, 256]
                mask = data['mask'].to(device)  # torch.int64 [b, 256, 256]

                # * 1 Extract feature of normal image
                normal_feature_maps = [normal_image] + feature_extractor(normal_image)[feature_start:feature_end]
                normal_feature = feature_aggregator(normal_feature_maps)  # [b, 576, 64, 64]

                # * 2 Extract feature of pseudo image
                pseudo_feature_maps = [pseudo_image] + feature_extractor(pseudo_image)[feature_start:feature_end]
                pseudo_feature = feature_aggregator(pseudo_feature_maps)

                # * Reconstruct the feature of pseudo or normal
                feature_reconstruction = feature_reconstructor(pseudo_feature)

                # * Fuse the reconstruction and feature of pseudo
                feature_diff = compute_energy(feature_reconstruction, pseudo_feature)  # L2 distance
                reconstruction_diff_concat = torch.cat((feature_diff, pseudo_feature), dim=1)  # Concat

                # * Segment the difference
                pseudo_predict = feature_segmentor(reconstruction_diff_concat)  # [b, 1, 64, 64]
                pseudo_predict = torch.sigmoid(pseudo_predict)  # [b, 1, 64, 64]

                # * Resize mask to the size of feature
                mask = nn.functional.interpolate(mask[:, None, :, :].float(),
                                                 size=feature_size,
                                                 mode='bilinear',
                                                 align_corners=False)  # [b, 1, 64, 64]
                mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))  # [b, 1, 64, 64]

                # * Compute reconstruction loss
                loss_rec_normal = L2_loss(feature_reconstruction, normal_feature)

                # * Compute segmentation loss
                loss_seg_focal = focal_loss(pseudo_predict, mask.long())
                loss_seg_l1 = l1_loss(pseudo_predict, mask.long())
                loss_seg_pseudo = loss_seg_focal + loss_seg_l1
                total_loss = loss_seg_pseudo + loss_rec_normal

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                seg_loss_epoch += loss_seg_pseudo.data.cpu().numpy()
                rec_loss_epoch += loss_rec_normal.data.cpu().numpy()

                writer.add_scalar('Seg_loss', loss_seg_pseudo, iter_num)
                writer.add_scalar('Rec_loss', loss_rec_normal, iter_num)

                iter_num += 1

            scheduler.step()

            data_num = len(train_loader)
            epoch_log = '{} Epoch[{}/{}]  seg_loss: {:.4f} rec_loss: {:.4f} Time: {:.4f}s.'.format(
                name, epoch, epochs, seg_loss_epoch / data_num, rec_loss_epoch / data_num, time.time() - time_start)
            print(epoch_log)

            with open(result_path + '/loss_log.txt', mode='a') as f:
                f.write(epoch_log)
                f.write('\n')

            # * Start testing
            display_image_path = []
            display_mask = []
            display_feature_diff = []
            display_segment_predict = []

            if epoch % log_val == 0 and epoch >= 680:

                feature_extractor.eval()
                feature_reconstructor.eval()
                feature_segmentor.eval()

                masks = []
                scores = []
                scores_I = []
                ious = []

                for i, data in enumerate(test_loader):
                    # * Load test image and mask
                    image_path = data['image_path'][0]  # ! original image size is not 256*256
                    image = data['image'].to(device)
                    mask = data['mask'].squeeze().numpy()  # [256, 256]

                    # * Extract feature
                    feature_maps = [image] + feature_extractor(image)[feature_start:feature_end]
                    # * Aggregate feature
                    feature = feature_aggregator(feature_maps)
                    # * Reconstruct feature
                    feature_reconstruction = feature_reconstructor(feature)

                    # * Fuse the difference of reconstruction and feature
                    feature_diff = compute_energy(feature_reconstruction, feature)
                    reconstruction_diff_concat = torch.cat((feature_diff, feature), dim=1)

                    # * Segment the difference
                    feature_predict = feature_segmentor(reconstruction_diff_concat)  # [1, 1, 64, 64]
                    feature_predict = torch.sigmoid(feature_predict)

                    # * Resize the predict
                    feature_predict = nn.functional.interpolate(feature_predict,
                                                                size=test_image_size,
                                                                mode='bilinear',
                                                                align_corners=False)  # [1, 1, 256, 256]
                    segment_predict = feature_predict[0, 0, :, :].detach().cpu().numpy()  # [256, 256]

                    display_image_path.append(image_path)
                    display_mask.append(mask)
                    display_feature_diff.append(feature_diff.detach().cpu())  # cuda will locate the memory
                    display_segment_predict.append(segment_predict)

                    masks.append(mask)
                    scores.append(segment_predict)

                # * Calculate the metrics
                masks = np.array(masks).astype(np.int32)
                scores = np.array(scores)

                # * Calculate Image-level metrics
                scores_ = torch.from_numpy(scores)
                scores_ = scores_.unsqueeze(0)
                scores_ = nn.functional.avg_pool2d(scores_, 32, stride=1, padding=32 // 2).squeeze().numpy()
                image_pred = scores_.max(1).max(1)
                image_label = masks.any(axis=1).any(axis=1)
                image_auc = roc_auc_score(image_label, image_pred)
                image_ap = average_precision_score(image_label, image_pred)

                # * Calculate Pixel-level metrics
                pixel_auc = roc_auc_score(masks.ravel(), scores.ravel())
                pixel_ap = average_precision_score(masks.ravel(), scores.ravel())

                bn_scores = (scores >= 0.5).astype(np.int32)
                iou01 = segment_iou_score(bn_scores, masks)
                iou = 0.0 if len(iou01) == 1 else iou01[1]

                # * Log metrics
                writer.add_scalar('image_auc', image_auc, epoch)
                writer.add_scalar('pixel_auc', pixel_auc, epoch)
                writer.add_scalar('pixel_ap', pixel_ap, epoch)

                score_log = 'epoch[{}/{}] image_aucroc: {:.4f}, pixel_aucroc: {:.4f}, pixel_ap: {:.4f}, IoU: {:.4f}'\
                            .format(epoch, epochs, image_auc, pixel_auc, pixel_ap, iou)
                print(score_log)

                with open(result_path + '/score_log.txt', mode='a') as f:
                    f.write(score_log)
                    f.write('\n')

                # * Visualize the results
                if pixel_ap > max_ap:
                    print('Start Visualization!')
                    max_ap = pixel_ap
                    for i in range(len(display_image_path)):

                        # * Visualize the segmentation
                        visualizer(
                            idx=i,
                            image_path=display_image_path[i],
                            save_path=result_path,
                            mask=display_mask[i],
                            rec_diff=display_feature_diff[i],
                            seg_target=display_segment_predict[i],
                            heat_map=False)

                        # * Visualize the reconstruction
                        score = display_feature_diff[i]
                        score = torch.sum(score, dim=1).unsqueeze(1)
                        score = nn.functional.interpolate(score, size=256, mode='bilinear',
                                                          align_corners=True)  # (1, 1, 256, 256)
                        visualize_and_save(i, result_path, display_image_path[i], display_mask[i], score)

                iaucroc.append(image_auc)
                paucroc.append(pixel_auc)
                pap.append(pixel_ap)
                piou.append(iou)

        class_idx = DataExcel['category'].index(name)
        DataExcel['image_aucroc'][class_idx] = np.mean(iaucroc)
        DataExcel['pixel_aucroc'][class_idx] = np.mean(paucroc)
        DataExcel['pixel_ap'][class_idx] = np.mean(pap)
        DataExcel['IoU(0.5)'][class_idx] = np.mean(piou)

    avg_tex_idx = DataExcel['category'].index('Avg. Tex.')
    avg_obj_idx = DataExcel['category'].index('Avg. Obj.')
    avg_idx = DataExcel['category'].index('Avg.')

    DataExcel['image_aucroc'][avg_tex_idx] = np.mean(DataExcel['image_aucroc'][:5])
    DataExcel['image_aucroc'][avg_obj_idx] = np.mean(DataExcel['image_aucroc'][6:16])
    DataExcel['image_aucroc'][avg_idx] = np.mean(DataExcel['image_aucroc'][:5] + DataExcel['image_aucroc'][6:16])

    DataExcel['pixel_aucroc'][avg_tex_idx] = np.mean(DataExcel['pixel_aucroc'][:5])
    DataExcel['pixel_aucroc'][avg_obj_idx] = np.mean(DataExcel['pixel_aucroc'][6:16])
    DataExcel['pixel_aucroc'][avg_idx] = np.mean(DataExcel['pixel_aucroc'][:5] + DataExcel['pixel_aucroc'][6:16])

    DataExcel['pixel_ap'][avg_tex_idx] = np.mean(DataExcel['pixel_ap'][:5])
    DataExcel['pixel_ap'][avg_obj_idx] = np.mean(DataExcel['pixel_ap'][6:16])
    DataExcel['pixel_ap'][avg_idx] = np.mean(DataExcel['pixel_ap'][:5] + DataExcel['pixel_ap'][6:16])

    DataExcel['IoU(0.5)'][avg_tex_idx] = np.mean(DataExcel['IoU(0.5)'][:5])
    DataExcel['IoU(0.5)'][avg_obj_idx] = np.mean(DataExcel['IoU(0.5)'][6:16])
    DataExcel['IoU(0.5)'][avg_idx] = np.mean(DataExcel['IoU(0.5)'][:5] + DataExcel['IoU(0.5)'][6:16])

    df = pd.DataFrame(DataExcel)
    df.to_excel(result_root + str(datetime.date.today()) + '.xlsx', index=False)
