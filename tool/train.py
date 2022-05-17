import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from dataloader.ABCDataset import ABCDataset
from tensorboardX import SummaryWriter

from util import config
from util.s3dis import S3DIS
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn
from util import transform as t
from util.loss_utils import compute_embedding_loss, compute_normal_loss, \
        compute_param_loss, compute_nnl_loss, compute_miou, compute_type_miou_abc, npy
from util.abc_utils import mean_shift, compute_entropy, construction_affinity_matrix_type, \
        construction_affinity_matrix_normal, mean_shift_gpu


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Primitive Segmentation')
    parser.add_argument('--config', type=str, default='config/ABC/ABC_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/ABC/ABC_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1: # single GPU training
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    elif args.data_name == 'ABC':
        from dataloader.ABCDataset import ABCDataset
        ABCDataset(args.data_root, args.train_dataset, opt=args, skip=args.train_skip, fold=args.train_fold)
        ABCDataset(args.data_root, args.test_dataset, opt=args, skip=args.test_skip)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    elif args.arch == 'pointtransformer_primitive_seg_repro':
        from model.pointtransformer.pointtransformer_primitive_seg import PointTransformerSeg as Model
        # from model.pointtransformer.dgcnn_transformer import PointTransformer_PrimSeg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    # model = Model(c=args.fea_dim, k=args.classes)
    model = Model(c=args.fea_dim, args=args)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.2),int(args.epochs*0.4), int(args.epochs*0.6),int(args.epochs*0.8)], gamma=0.1) # decay lr by 10% when epoch is 60% and 80% of epochs

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        # logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    # train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    train_data = ABCDataset(args.data_root, args.train_dataset, opt=args, skip=args.train_skip, fold=args.train_fold)
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = None
    if args.evaluate:
        # val_transform = None
        # val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        val_data = ABCDataset(args.data_root, args.test_dataset, opt=args, skip=args.test_skip)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.is_test == True:
        miou = validate(val_loader, model, 1)
        if main_process():
            logger.info('Test IoU: {:.4f}'.format(miou))
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
            train(train_loader, model, optimizer, epoch)
            scheduler.step()
            epoch_log = epoch + 1
            # if main_process():
            #     writer.add_scalar('loss_train', loss_train, epoch_log)
            #     writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            #     writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            #     writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            torch.cuda.empty_cache()

            is_best = False
            if args.evaluate and (epoch_log % args.eval_freq == 0):
                if args.data_name == 'shapenet':
                    raise NotImplementedError()
                else:
                    # loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
                    miou = validate(val_loader, model, epoch_log)

                if main_process():
                    # writer.add_scalar('loss_val', loss_val, epoch_log)
                    # writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                    # writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                    # writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                    is_best = miou > best_iou
                    best_iou = max(best_iou, miou)

            if (epoch_log % args.save_freq == 0) and main_process():
                filename = args.save_path + '/model/model_last.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
                if is_best:
                    logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                    shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
            
            torch.cuda.empty_cache()

        if main_process():
            writer.close()
            logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))

def process_batch(batch_data_label, model, args, postprocess=False):

    inputs_xyz_th = (batch_data_label['gt_pc']).float().cuda(non_blocking=True).permute(0,2,1)
    inputs_n_th = (batch_data_label['gt_normal']).float().cuda(non_blocking=True).permute(0,2,1)
    inputs_xyz = inputs_xyz_th.transpose(1, 2).contiguous()
    inputs_n = inputs_n_th.transpose(1, 2).contiguous()

    batch_size, N, _ = inputs_xyz.shape
    l = np.arange(N)
    if postprocess:
        np.random.seed(1234)
    np.random.shuffle(l)
    random_index = torch.from_numpy(l[:7000])
    
    points = torch.cat([inputs_xyz, inputs_n, inputs_xyz], dim=-1)
    subidx = random_index.to(points.device).long().view(1,1,-1).repeat(batch_size,1,1)
    points = torch.gather(points.permute(0, 2, 1), -1, subidx.repeat(1,9,1)) # [b, 6, n_sub]
    inputs_xyz = torch.gather(inputs_xyz.permute(0, 2, 1), -1, subidx.repeat(1,3,1)) # [b, 6, n_sub]
    
    affinity_feat, type_per_point, param_per_point = model(points.transpose(1, 2).contiguous())

    sub_idx = subidx.squeeze(1)
    inputs_xyz_sub = torch.gather(inputs_xyz_th, -1, sub_idx.unsqueeze(1).repeat(1,3,1))
    N_gt = (batch_data_label['gt_normal']).float().cuda(non_blocking=True)
    N_gt = torch.gather(N_gt, 1, sub_idx.unsqueeze(-1).repeat(1,1,3))
    I_gt = torch.gather(batch_data_label['I_gt'], -1, sub_idx)
    T_gt = torch.gather(batch_data_label['T_gt'], -1, sub_idx)

    loss_dict = {}
        
    if 'f' in args.loss_class:
        # network feature loss
        feat_loss, pull_loss, push_loss = compute_embedding_loss(affinity_feat, I_gt)
        loss_dict['feat_loss'] = feat_loss
    # if 'n' in args.loss_class:
    #     # normal angle loss
    #     normal_loss = compute_normal_loss(normal_per_point, N_gt)
    #     loss_dict['normal_loss'] = args.normal_weight * normal_loss
    if 'p' in args.loss_class:
        T_param_gt = torch.gather(batch_data_label['T_param'], 1, sub_idx.unsqueeze(-1).repeat(1,1,22))
        # parameter loss
        param_loss = compute_param_loss(param_per_point, T_gt, T_param_gt)
        loss_dict['param_loss'] = args.param_weight * param_loss
    if 'r' in args.loss_class:
        # primitive nnl loss
        type_loss = compute_nnl_loss(type_per_point, T_gt)
        loss_dict['nnl_loss'] = args.type_weight * type_loss

    total_loss = 0
    for key in loss_dict.keys():
        if 'loss' in key:
            total_loss += loss_dict[key]

    # try:
    if postprocess:
        
        affinity_matrix = construction_affinity_matrix_type(inputs_xyz_sub, type_per_point, param_per_point, args.sigma)
        
        affinity_matrix_normal = construction_affinity_matrix_normal(inputs_xyz_sub, N_gt, sigma=args.normal_sigma, knn=args.edge_knn) 

        obj_idx = batch_data_label['index'][0]
            
        spec_embedding_list = []
        weight_ent = []

        # use network feature
        feat_ent = args.feat_ent_weight - float(npy(compute_entropy(affinity_feat)))
        weight_ent.append(feat_ent)
        spec_embedding_list.append(affinity_feat)
        
        # use geometry distance feature
        topk = args.topK            
        # 求对称正定义广义特征值问题的k个最大或最小特征值以及对应特征向量
        affinity_matrix = affinity_matrix.to(points.device)
        e, v = torch.lobpcg(affinity_matrix, k=topk, niter=10)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)

        dis_ent = args.dis_ent_weight - float(npy(compute_entropy(v)))
        
        weight_ent.append(dis_ent)
        spec_embedding_list.append(v)
            
        # use edge feature
        edge_topk = args.edge_topK
        affinity_matrix_normal = affinity_matrix_normal.to(points.device)
        e, v = torch.lobpcg(affinity_matrix_normal, k=edge_topk, niter=10)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
        
        edge_ent = args.edge_ent_weight - float(npy(compute_entropy(v)))
        
        weight_ent.append(edge_ent)
        spec_embedding_list.append(v)

        
        
        # combine features
        weighted_list = []
        norm_weight_ent = weight_ent / np.linalg.norm(weight_ent)
        for i in range(len(spec_embedding_list)):
            weighted_list.append(spec_embedding_list[i] * weight_ent[i])

        spectral_embedding = torch.cat(weighted_list, dim=-1)
        
        spec_cluster_pred = mean_shift_gpu(spectral_embedding, bandwidth=args.bandwidth)
        cluster_pred = spec_cluster_pred
        miou = compute_miou(spec_cluster_pred, I_gt)
        loss_dict['miou'] = miou
        miou = compute_type_miou_abc(type_per_point, T_gt, cluster_pred, I_gt)
        loss_dict['type_miou'] = miou
    # except:
    #     import ipdb
    #     ipdb.set_trace()

    return total_loss, loss_dict

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    param_loss_meter = AverageMeter()
    nnl_loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for batch_idx, batch_data_label in enumerate(train_loader):
        data_time.update(time.time() - end)
        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        optimizer.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
            total_loss, loss_dict = process_batch(batch_data_label, model, args)

            total_loss.backward()
        
        optimizer.step()

        # All Reduce loss
        feat_loss = loss_dict['feat_loss']
        param_loss = loss_dict['param_loss']
        nnl_loss = loss_dict['nnl_loss']
        if args.multiprocessing_distributed:
            dist.all_reduce(feat_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(param_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(nnl_loss.div_(torch.cuda.device_count()))

        feat_loss_meter.update(npy(feat_loss).item())
        param_loss_meter.update(npy(param_loss).item())
        nnl_loss_meter.update(npy(nnl_loss).item())
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + batch_idx + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Feat_Loss {feat_loss_meter.val:.4f} '
                        'Param_Loss {param_loss_meter.val:.4f} '
                        'Primitive_Loss {nnl_loss_meter.val:.4f}.'.format(epoch+1, args.epochs, batch_idx + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          feat_loss_meter=feat_loss_meter,
                                                          param_loss_meter=param_loss_meter,
                                                          nnl_loss_meter=nnl_loss_meter))
        if main_process():
            writer.add_scalar('feat_loss_train_batch', feat_loss_meter.val, current_iter)
            writer.add_scalar('param_loss_train_batch', param_loss_meter.val, current_iter)
            writer.add_scalar('nnl_loss_train_batch', nnl_loss_meter.val, current_iter)
            writer.add_scalar('data_time', data_time.val, current_iter)
            writer.add_scalar('batch_time', batch_time.val, current_iter)
        


    # for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
    #     data_time.update(time.time() - end)
    #     coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
    #     output = model([coord, feat, offset])
    #     if target.shape[-1] == 1:
    #         target = target[:, 0]  # for cls
    #     loss = criterion(output, target)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     output = output.max(1)[1]
    #     n = coord.size(0)
    #     if args.multiprocessing_distributed:
    #         loss *= n
    #         count = target.new_tensor([n], dtype=torch.long)
    #         dist.all_reduce(loss), dist.all_reduce(count)
    #         n = count.item()
    #         loss /= n
    #     intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
    #     if args.multiprocessing_distributed:
    #         dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
    #     intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    #     intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    #     accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    #     loss_meter.update(loss.item(), n)
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     # calculate remain time
    #     current_iter = epoch * len(train_loader) + i + 1
    #     remain_iter = max_iter - current_iter
    #     remain_time = remain_iter * batch_time.avg
    #     t_m, t_s = divmod(remain_time, 60)
    #     t_h, t_m = divmod(t_m, 60)
    #     remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

    #     if (i + 1) % args.print_freq == 0 and main_process():
    #         logger.info('Epoch: [{}/{}][{}/{}] '
    #                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
    #                     'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
    #                     'Remain {remain_time} '
    #                     'Loss {loss_meter.val:.4f} '
    #                     'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
    #                                                       batch_time=batch_time, data_time=data_time,
    #                                                       remain_time=remain_time,
    #                                                       loss_meter=loss_meter,
    #                                                       accuracy=accuracy))
    #     if main_process():
    #         writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
    #         writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
    #         writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
    #         writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    # if main_process():
    #     logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    # return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, epoch_log):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    param_loss_meter = AverageMeter()
    nnl_loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    type_miou_meter = AverageMeter()

    model.eval()
    end = time.time()
    stat_dict = {}
    cnt = 0

    for batch_idx, batch_data_label in enumerate(val_loader):
        data_time.update(time.time() - end)
        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        with torch.no_grad():
            total_loss, loss_dict = process_batch(batch_data_label, model, args, postprocess=True)
        
        if args.multiprocessing_distributed:
            dist.all_reduce(loss_dict['feat_loss'].div_(torch.cuda.device_count()))
            dist.all_reduce(loss_dict['param_loss'].div_(torch.cuda.device_count()))
            dist.all_reduce(loss_dict['nnl_loss'].div_(torch.cuda.device_count()))
            dist.all_reduce(loss_dict['miou'].div_(torch.cuda.device_count()))
            dist.all_reduce(loss_dict['type_miou'].div_(torch.cuda.device_count()))
        
        # Accumulate statistics and print out
        for key in loss_dict:
            if key not in stat_dict: stat_dict[key] = 0
            stat_dict[key] += loss_dict[key].item()
        cnt += len(batch_data_label['index'])
    
    feat_loss_meter.update(stat_dict['feat_loss']/cnt)
    param_loss_meter.update(stat_dict['param_loss']/cnt)
    nnl_loss_meter.update(stat_dict['nnl_loss']/cnt)
    miou_meter.update(stat_dict['miou']/cnt)
    type_miou_meter.update(stat_dict['type_miou']/cnt)
    if main_process():
        logger.info('Feat_Loss {feat_loss_meter.val:.4f} '
                    'Param_Loss {param_loss_meter.val:.4f} '
                    'Primitive_Loss {nnl_loss_meter.val:.4f}.'.format(feat_loss_meter=feat_loss_meter,
                                                        param_loss_meter=param_loss_meter,
                                                        nnl_loss_meter=nnl_loss_meter))
        logger.info('Val result: mIoU/Type mIou {miou_meter.val:.4f}/{type_miou_meter.val:.4f}.'.format(miou_meter=miou_meter, type_miou_meter=type_miou_meter))

        writer.add_scalar('feat_loss_val', feat_loss_meter.val, epoch_log)
        writer.add_scalar('param_loss_val', param_loss_meter.val, epoch_log)
        writer.add_scalar('nnl_loss_val', nnl_loss_meter.val, epoch_log)
        writer.add_scalar('miou', miou_meter.val, epoch_log)
        writer.add_scalar('type_miou', type_miou_meter.val, epoch_log)

    return miou_meter.val

    # for i, (coord, feat, target, offset) in enumerate(val_loader):
    #     data_time.update(time.time() - end)
    #     coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
    #     if target.shape[-1] == 1:
    #         target = target[:, 0]  # for cls
    #     with torch.no_grad():
    #         output = model([coord, feat, offset])
    #     loss = criterion(output, target)

    #     output = output.max(1)[1]
    #     n = coord.size(0)
    #     if args.multiprocessing_distributed:
    #         loss *= n
    #         count = target.new_tensor([n], dtype=torch.long)
    #         dist.all_reduce(loss), dist.all_reduce(count)
    #         n = count.item()
    #         loss /= n

    #     intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
    #     if args.multiprocessing_distributed:
    #         dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
    #     intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    #     intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    #     accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
    #     loss_meter.update(loss.item(), n)
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #     if (i + 1) % args.print_freq == 0 and main_process():
    #         logger.info('Test: [{}/{}] '
    #                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
    #                     'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
    #                     'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
    #                     'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
    #                                                       data_time=data_time,
    #                                                       batch_time=batch_time,
    #                                                       loss_meter=loss_meter,
    #                                                       accuracy=accuracy))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # if main_process():
    #     logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    #     for i in range(args.classes):
    #         logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    #     logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    # return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
