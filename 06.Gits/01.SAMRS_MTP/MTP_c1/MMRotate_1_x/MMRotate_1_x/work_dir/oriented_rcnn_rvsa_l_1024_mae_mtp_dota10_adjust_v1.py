angle_version = 'le90'
backend_args = None
checkpoint = '/mnt/hdd/eric/.tmp_ipy/00.Weights/MTP/diorr-rvsa-l-mae-mtp-epoch_12.pth'
data_root = '/mnt/hdd/eric/.tmp_ipy/00.Data/ShipRS_dataset/ShipRSImageNet_V1/DOTA_Format_MMrotate_1x/'
dataset_type = 'ShipDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
device = 'cuda:0'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gpu_ids = range(0, 1)
load_from = '/mnt/hdd/eric/.tmp_ipy/00.Weights/MTP/diorr-rvsa-l-mae-mtp-epoch_12.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depth=24,
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dim=1024,
        img_size=800,
        interval=6,
        mlp_ratio=4,
        num_heads=16,
        out_indices=[
            7,
            11,
            15,
            23,
        ],
        patch_size=16,
        pretrained=None,
        qk_scale=None,
        qkv_bias=True,
        type='RVSA_MTP_branches',
        use_abs_pos_emb=True,
        use_checkpoint=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                angle_version='le90',
                edge_swap=True,
                norm_factor=None,
                proj_xy=True,
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.1,
                ),
                type='DeltaXYWHTRBBoxCoder'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(
                beta=1.0, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0,
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False),
            num_classes=1,
            predict_box_type='rbox',
            reg_class_agnostic=True,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            roi_feat_size=7,
            type='mmdet.Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(
                clockwise=True,
                out_size=7,
                sample_num=2,
                type='RoIAlignRotated'),
            type='RotatedSingleRoIExtractor'),
        test_cfg=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            nms_pre=2000,
            score_thr=0.05),
        train_cfg=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='mmdet.RandomSampler')),
        type='mmdet.StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='mmdet.AnchorGenerator',
            use_box_type=True),
        bbox_coder=dict(
            angle_version='le90',
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            type='MidpointOffsetCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        type='OrientedRPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            nms_pre=2000,
            score_thr=0.05),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='mmdet.RandomSampler')),
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D'),
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='mmdet.RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms'),
            nms_pre=2000)),
    type='mmdet.FasterRCNN')
model_wrapper = dict(
    detect_anomalous_params=False,
    find_unused_parameters=False,
    type='MMDistributedDataParallel')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
resume = False
seed = 0
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='val',
        data_prefix=dict(img='images_png', img_path='images_png'),
        data_root=
        '/mnt/hdd/eric/.tmp_ipy/00.Data/ShipRS_dataset/ShipRSImageNet_V1/DOTA_Format_MMrotate_1x/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='ShipDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='train',
        data_prefix=dict(img='images_png', img_path='images_png'),
        data_root=
        '/mnt/hdd/eric/.tmp_ipy/00.Data/ShipRS_dataset/ShipRSImageNet_V1/DOTA_Format_MMrotate_1x/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='ShipDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='val',
        data_prefix=dict(img='images_png', img_path='images_png'),
        data_root=
        '/mnt/hdd/eric/.tmp_ipy/00.Data/ShipRS_dataset/ShipRSImageNet_V1/DOTA_Format_MMrotate_1x/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='mmdet.Pad'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='ShipDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            800,
            800,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/06.Gits/01.SAMRS_MTP/MTP/MMRotate_1_x/work_dir'
