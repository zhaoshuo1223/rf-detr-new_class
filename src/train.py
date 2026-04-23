import os
import torch
# from rfdetr import RFDETRSegPreview
from rfdetr import RFDETRMedium,RFDETRSmall,RFDETRNano,RFDETRBase,RFDETRLarge,RFDETRSegMedium
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL
# 抑制 TensorFlow oneDNN 警告信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


AUG_INDUSTRIAL = {
    "HorizontalFlip": {"p": 0.3},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},
}



if __name__ == '__main__':

    if torch.cuda.is_available():
       deive = torch.device('cuda:0')
    else:
       deive = torch.device('cpu')

    # 初始化分割模型（在这里传入模型配置参数）
    model = RFDETRNano(
        num_windows=2,      # 必须在模型初始化时指定，与训练时保持一致
        patch_size=16,      # 必须在模型初始化时指定，与训练时保持一致
        resolution=128,     # 必须在模型初始化时指定，与训练时保持一致
        freeze_encoder=True,
        num_classes=1,
        positional_encoding_size = 128//16,
        segmentation_head=False,
    )

    # 启动训练
    model.train(
        # dataset_file = 'coco',
        dataset_dir = r'D:\aotto\dingweixiao\aaa_coco_data', 
        num_windows = 2,
        num_classes=1,
        epochs = 100,           # 训练轮次
        patch_size = 16,
        resolution = 128,       # 输入图像分辨率
        batch_size = 8,         # 批次大小（建议根据硬件调整）
        num_workers = 2,        # 线程数
        grad_accum_steps = 4,   # 梯度累积步骤
        gradient_checkpointing=False,
        validation_interval=10,        #几个伦茨进行一次验证
        checkpoint_interval = 10,    #几个伦茨进行一次保存
        lr = 1e-4,              # 学习率
        # optimizer = 'SGD',      # 优化器
        amp = True,             # 是否使用混合精度训练
        # use_ema = True,         # 是否使用EMA
        deivce = deive,         # 训练设备
        #early_stopping = False,
        #early_stopping_patience: int = 10 
        #resume = r"D:\aotto\chengxingliao\model\576m\eval\latest.pth",         # 继续训练
        distributed = False,    # 是否使用分布式训练
        output_dir = r'D:\aotto\dingweixiao\MODEL\test',
        aug_config=AUG_INDUSTRIAL,
        multi_scale = True,
        early_stopping=True,
        early_stopping_patience=10,   #早停，10个伦茨后停止
        early_stopping_min_delta=0.002,
        progress_bar = "rich"
        # persistent_workers = True,   #数据加载器，True为持久化
        #expanded_scales = True,
    )
