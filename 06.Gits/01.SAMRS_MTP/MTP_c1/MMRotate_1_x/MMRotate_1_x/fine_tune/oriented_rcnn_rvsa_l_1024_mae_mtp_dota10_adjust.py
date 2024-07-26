
################################### default time #############################################

default_scope = 'mmrotate'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=8),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

#################################### schedule ################################################

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
    type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor_ViT', 
    paramwise_cfg=dict(
        num_layers=24, 
        layer_decay_rate=0.9,
        )
        )

#################################### dataset ################################################

# dataset settings
dataset_type = 'DOTADataset'
data_root = '/kernel/cv/diwang22/split_ms_dotav1'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(type='DOTAMetric', metric='mAP')

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='DOTAMetric',
    outfile_prefix = '/data/diwang22/code/RS-MTP-workdir/finetune/Rotated_Detection/dotav1_ms/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/test/submit/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10',
    format_only=True,
    merge_patches=True)

###############

model_wrapper=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=False,
        detect_anomalous_params=False)

angle_version = 'le90'

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
            num_classes=20,
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