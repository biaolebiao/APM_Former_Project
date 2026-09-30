import torch
import logging
from monai.networks.nets import resnet18, DenseNet121, SEResNet50, EfficientNetBN, ViT

def build_selected_model(model_type, device, img_size=(96, 96, 96)):
    """
    根据传入的 model_type 动态初始化并返回模型对象及对应的解冻轮次
    """
    logger = logging.getLogger("Train")
    logger.info("=" * 60)
    logger.info(f"🚀 当前正在初始化的模型架构: {model_type}")
    logger.info("=" * 60)
    
    # 彻底废弃分阶段冻结策略，所有模型均从 Epoch 0 开始端到端联合训练
    unfreeze_epoch = 0  

    PRETRAINED_SWIN_PATH = "checkpoints/model_swinvit.pt"
    
    if model_type == "APM_Former":
        # 对应论文 Table II 中的 Ours / Variant 2: AP(+) + DCN(+)
        from models.apm_former import APM_Former_ImageOnly
        model = APM_Former_ImageOnly(
            feature_size=24, num_classes=2, 
            pretrained_swin_path=PRETRAINED_SWIN_PATH,
            use_anatomy_prior=True, use_dcn_alignment=True
        ).to(device)

    elif model_type == "APM_Former_Variant1":
        # 对应论文 Table II 中的 Variant 1: AP(+) + DCN(-)
        # 仅生成先验注意力掩码，不执行 DCN 形变对齐
        from models.apm_former import APM_Former_ImageOnly
        model = APM_Former_ImageOnly(
            feature_size=24, num_classes=2, 
            pretrained_swin_path=PRETRAINED_SWIN_PATH,
            use_anatomy_prior=True, use_dcn_alignment=False 
        ).to(device)

    elif model_type == "Pure_Swin":
        # 对应论文 Table II 中的 Baseline: AP(-) + DCN(-)
        # 退化为纯净的 Swin Transformer 架构
        from models.apm_former import APM_Former_ImageOnly
        model = APM_Former_ImageOnly(
            feature_size=24, num_classes=2, 
            pretrained_swin_path=PRETRAINED_SWIN_PATH,
            use_anatomy_prior=False, use_dcn_alignment=False
        ).to(device)

    elif model_type == "ResNet18":
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)

    elif model_type == "DenseNet121":
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    elif model_type == "SEResNet50":
        model = SEResNet50(spatial_dims=3, in_channels=1, num_classes=2).to(device)

    elif model_type == "EfficientNet3D":
        model = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2).to(device)

    elif model_type == "ViT":
        model = ViT(in_channels=1, img_size=img_size, patch_size=(16, 16, 16), num_classes=2, spatial_dims=3, classification=True, post_activation="Tanh").to(device)

    else:
        raise ValueError(f"❌ 不支持的模型类型: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📦 模型 [{model_type}] 总参数量: {total_params / 1e6:.2f} M")
    
    return model, unfreeze_epoch