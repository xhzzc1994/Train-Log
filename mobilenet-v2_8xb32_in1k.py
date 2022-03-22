# mobilenet-v2_8×b32_in1k配置文件
#
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',  # 模型
    '../_base_/datasets/imagenet_bs32_pil_resize.py',  # 数据
    # '../_base_/schedules/imagenet_bs256_epochstep.py',  # 训练策略
    '../_base_/schedules/imagenet_bs256.py',  # 使用基本策略
    '../_base_/default_runtime.py'  # 默认运行设置
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset setting
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,  # 构建 dataloader 时，每个 GPU 的 Batch Size
    workers_per_gpu=2,  # 构建 dataloader 时，每个 GPU 的 线程数

    # 配置训练集
    train=dict(
        # type=dataset_type,
        data_prefix='/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/train',
        classes="/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/classes.txt",
        # pipeline=train_pipeline
    ),
    val=dict(
        # type=dataset_type,
        data_prefix='/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/val',
        ann_file='/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/val.txt',
        classes="/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/classes.txt",
        # pipeline=test_pipeline
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        # type=dataset_type,
        data_prefix='/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/test',
        ann_file='/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/test.txt',
        classes="/algdata01/xianxian.wang/zzc/classification/datasets/mini-imagenet/classes.txt",
        # pipeline=test_pipeline
    ))
# evaluation = dict(interval=1, metric='accuracy')
# evaluation hook 的配置
evaluation = dict(
    interval=1,
    metric_options={'top_k': (1, 5)}
    )  # 设置topk的值

# optimizer schedules 训练策略的配置
optimizer = dict(
    type='SGD', 
    lr=0.045, 
    momentum=0.9, 
    weight_decay=0.00004
    )
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', gamma=0.98, step=1)
lr_config = dict(policy='step', gamma=0.98, step=[30, 60, 90])  # 训练大型数据集时
# max epochs
runner = dict(type='EpochBasedRunner', max_epochs=300)
