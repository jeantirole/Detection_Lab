import mmcv
import mmrotate
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
# Check mmcv installation
print(mmcv.__version__) #noqa E1101
# Check MMRotate installation
print(mmrotate.__version__)
# Check MMDetection installation
print(mmdet.__version__)

print(get_compiling_cuda_version())
print(get_compiler_version())

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmdet.apis import inference_detector, init_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

# register all modules in mmdet into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules_mmdet(init_default_scope=False)
register_all_modules(init_default_scope=False)

# Choose to use a config and initialize the detector
config = '/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/06.Gits/01.SAMRS_MTP/MTP/MMRotate_1_x/mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'
# Setup a checkpoint file to load
#checkpoint = '/mnt/hdd/eric/.tmp_ipy/15.Lab_Detection/06.Gits/01.SAMRS_MTP/MTP/MMRotate_1_x/rotated_rtmdet_l-3x-dota-23992372.pth'
# Set work_dir
work_dir = '../work_dirs/rotated_rtmdet_l-3x-dota'
# Set the device to be used for evaluation
device='cuda:0'
# Set inference path
img = './demo.jpg'

cfg = Config.fromfile(config)
#cfg.load_from = checkpoint
cfg.work_dir = work_dir

# register all modules in mmrotate into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
#model = init_detector(cfg, checkpoint, palette="dota", device=device)
#model

from mmrotate.registry import MODELS
from mmrotate.testing import get_detector_cfg
#from torchinfo import summary

cfg_file=config
input_size=(1,3,1024,1024)
model=get_detector_cfg(cfg_file)
detector=MODELS.build(model)
#summary(detector, input_size)

# # Let's take a look at the dataset image
# import matplotlib.pyplot as plt
# img = mmcv.imread('ssdd_tiny/images/000631.png')
# plt.figure(figsize=(15, 10))
# plt.imshow(mmcv.bgr2rgb(img))
# plt.show()


from mmrotate.registry import DATASETS
from mmrotate.datasets.dota import DOTADataset

@DATASETS.register_module()
class TinyDataset(DOTADataset):
    """SAR ship dataset for detection."""
    
    METAINFO = {
            'classes':('ship',),
                # palette is a list of color tuples, which is used for visualization.
            'palette': [(165, 42, 42),]
        }
    

cfg = Config.fromfile(config)

import random
import numpy as np
import torch


# Modify dataset type and path
cfg.data_root = 'ssdd_tiny/'
cfg.dataset_type = 'TinyDataset'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
#cfg.load_from = 'rotated_rtmdet_tiny-3x-dota-9d821076.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../work_dirs/tutorial_exps'

cfg.optim_wrapper.optimizer.lr = 0.001

cfg.train_cfg.val_interval = 3
# Change the evaluation metric since we use customized dataset.
cfg.val_evaluator.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.val_evaluator.interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 1
# cfg.max_epochs=36
cfg.train_cfg.max_epochs=12

# Set seed thus the results are more reproducible
cfg.seed = 0
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

cfg.gpu_ids = range(1)
cfg.device='cuda'


# modify pipeline mainly for resize scale (512,512)
cfg.train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(
        type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
cfg.val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
cfg.test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(512, 512),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.ann_file = 'val'
cfg.val_dataloader.dataset.data_prefix=dict(img_path='images', img='images')
# cfg.val_dataloader.dataset.img_prefix = 'images'
cfg.val_dataloader.dataset.data_root = 'ssdd_tiny/'

cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.ann_file = 'train'
cfg.train_dataloader.dataset.data_prefix=dict(img_path='images', img='images')
# cfg.train_dataloader.dataset.img_prefix = 'images'
cfg.train_dataloader.dataset.data_root = 'ssdd_tiny/'

cfg.test_dataloader.dataset.type = cfg.dataset_type
cfg.test_dataloader.dataset.data_prefix=dict(img_path='images', img='images')
cfg.test_dataloader.dataset.ann_file = 'val'
# cfg.test_dataloader.dataset.img_prefix = 'images'
cfg.test_dataloader.dataset.data_root = 'ssdd_tiny/'

cfg.train_dataloader.dataset.pipeline=cfg.train_pipeline
cfg.val_dataloader.dataset.pipeline=cfg.val_pipeline
cfg.test_dataloader.dataset.pipeline=cfg.test_pipeline

cfg.val_evaluator = dict(type='DOTAMetric', metric='mAP')
cfg.test_evaluator = cfg.val_evaluator
# We can also use tensorboard to log the training process
#cfg.vis_backends=[dict(type='TensorboardVisBackend')]
#cfg.visualizer = dict(type='mmrotate.RotLocalVisualizer',vis_backends=vis_backends)

# We can initialize the logger for training and have a look
# at the final config used for training
cfg.dump('./tiny_cfg.py')
print(f'Config:\n{cfg.pretty_text}')



# register all modules in mmdet into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules_mmdet(init_default_scope=False)
register_all_modules(init_default_scope=False)

# register all modules in mmrotate into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(cfg, None, palette="ssdd", device=cfg.device)


runner = Runner.from_cfg(cfg)
runner.train()


