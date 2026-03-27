import os
import torch
from rfdetr import RFDETRSegPreview
from rfdetr import RFDETRMedium,RFDETRSmall,RFDETRNano,RFDETRBase,RFDETRLarge

# 抑制 TensorFlow oneDNN 警告信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':

    if torch.cuda.is_available():
       deive = torch.device('cuda:0')
    else:
       deive = torch.device('cpu')

    # 初始化分割模型（在这里传入模型配置参数）
    model = RFDETRMedium(
        num_windows=2,      # 必须在模型初始化时指定，与训练时保持一致
        patch_size=16,      # 必须在模型初始化时指定，与训练时保持一致
        resolution=384,     # 必须在模型初始化时指定，与训练时保持一致
        segmentation_head=False
    )

    # 启动训练
    model.train(
        dataset_dir = r'C:\Users\zhaoshuo\Desktop\chengxingliao\reyi\coco',
        #dataset_file = "liao",
        #background_category_ids = [0],        #背景标注设置，设置所有的背景类别为0.          
        num_windows = 2,
        epochs = 50,           # 训练轮次
        patch_size = 16,
        resolution = 384,       # 输入图像分辨率
        batch_size = 1,         # 批次大小（建议根据硬件调整）
        num_workers = 2,        # 线程数
        grad_accum_steps = 6,   # 梯度累积步骤
        gradient_checkpointing=False,
        validation_interval=1,        #几个伦茨进行一次验证
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
        output_dir = r'D:\aotto\chengxingliao\model\384m'
    )
