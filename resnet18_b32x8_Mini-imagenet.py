_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,  # 干网网络深度， ResNet 一般有18, 34, 50, 101, 152 可以选择
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,  # 输入通道数，这与 neck 的输出通道一致
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# =================== dataset settings ===================
dataset_type = 'ImageNet'  # 数据集名称
img_norm_cfg = dict(  # 图像归一化设置，用来归一化输入图像
    mean=[123.675, 116.28, 103.53],  # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],  # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True  # 是否翻转通道，使用cv2\mmcv读取图片默认为RGB通道通道顺序，这里 Normalize 均值方差数组的数值是以 RGB 通道顺序， 因此需要反转通道顺序。
)
# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 读取图片
    dict(type='RandomResizedCrop', size=224),  # 随即缩放抠图
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),  # 以概率为0.5随机水平翻转图片
    dict(type='Normalize', **img_norm_cfg),  # 进行归一化
    dict(type='ImageToTensor', keys=['img']),  # 将 image 转换为 torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),  # 将 gt_label 转换为 torch.Tensor
    dict(type='Collect', keys=['img', 'gt_label'])  # 决定数据中哪些键应该传递给检测器的流程
]
# 测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])  # test 时不传递 gt_label
]
data = dict(
    samples_per_gpu=128,  # 单个GPU的 batch size
    workers_per_gpu=2,  # 单个GPU的线程数

    # 配置训练集
    train=dict(
        # type=dataset_type,  # 数据集名称
        # ############# cats_dogs_dataset #############
        # data_prefix='data/cats_dogs_dataset/training_set/training_set',
        # classes="data/cats_dogs_dataset/classes.txt",

        data_prefix='data/mini-imagenet/train',  # 数据集目录，当不存在 ann_file 时，类别信息从文件夹自动获取
        classes="data/mini-imagenet/classes.txt",

        # pipeline=train_pipeline  # 数据集需要经过的 数据流水线
    ),
    val=dict(
        # type=dataset_type,
        data_prefix='data/mini-imagenet/val',
        ann_file='data/mini-imagenet/val.txt',  # 标注文件路径，存在 ann_file 时，不通过文件夹自动获取类别信息
        classes="data/mini-imagenet/classes.txt",

        # pipeline=test_pipeline
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        # type=dataset_type,
        data_prefix='data/mini-imagenet/test',
        ann_file='data/mini-imagenet/test.txt',
        classes="data/mini-imagenet/classes.txt",

        # pipeline=test_pipeline
    ))

# evaluation = dict(  # evaluation hook 的配置
#     interval=1,  # 验证期间的间隔，单位为 epoch 或者 iter， 取决于 runner 类型。
#     metric='accuracy'  # 验证期间使用的指标。
# )
evaluation = dict(
    interval=1,
    metric_options={'top_k': (1, 5)}  # 设置使用topk的验证方式
)

# =================== optimizer schedules 训练策略的配置 ===================
# 继承自 '../_base_/schedules/imagenet_bs256.py' 中
"""
用于构建优化器的配置文件，支持pytorch中所有的优化器，同时其参数与pytroch里的参数是一致的
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
    SGD：使用的优化方式
    lr：当 lr=0.1 时，是从头训练所使用的学习率，如果是 fineturn(微调) 的话不需要使用这么大的学习率
    momentum：动量不需要改
"""
optimizer = dict(
    type='SGD',  # 优化器类型
    lr=0.045,  # 优化器的学习率
    momentum=0.9,  # 动量
    weight_decay=0.00004  # 权重衰减系数
)
# optimizer hook的配置文件
optimizer_config = dict(grad_clip=None)
# learning policy
"""
lr_config = dict(policy='step', gamma=0.98, step=1)
    学习率下降策略，对于fineturn小的数据集来说，可以设置为 setp=1，即训练一轮就做一个下降
    当训练的是ImageNet等大型数据集时，需要将学习率下降策略设置为： step=[30, 60, 90]
"""
# lr_config = dict(policy='step', gamma=0.98, step=1)
lr_config = dict(
    policy='step',  # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。
    gamma=0.98,  #
    step=[30, 60, 90]  # 在 epoch 为30， 60， 90时，lr进行衰减
)
# max epochs
runner = dict(
    type='EpochBasedRunner',  # 设置使用的 runner 的类别，如 IterBasedRunner 或 EpochBasedRunner。
    max_epochs=300  # runner的总回合数，对于 IterBasedRunner 则使用 max_iters
)

# =================== 运行配置：继承自 default_runtime.py ===================
# 本部分主要包括保存权重策略、日志配置、训练参数、断点权重路径和工作目录等等。
# checkpoint saving
checkpoint_config = dict(interval=1)  # 保存的间隔为 1，单位会根据runner不同而变动，可以为epoch或者iter
# yapf:disable
log_config = dict(
    interval=100,  # 打印日志的间隔，单位 iters
    hooks=[
        dict(type='TextLoggerHook'),  # 记录训练过程的文本记录器
        dict(type='TensorboardLoggerHook')  # 支持Tensorboard日志
    ])
# yapf:enable

dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数
log_level = 'INFO'  # 日志的输出级别
load_from = None  # 使用的 checkpoint 的路径，设置的预训练模型；
resume_from = None  # 从给定路径里恢复检查点(checkpoints),训练模式将从检查点保存的轮次开始恢复训练
workflow = [('train', 1)]  # runner的工作流程，[('train', 1)]，表示只有一个工作流且工作流只执行一次
