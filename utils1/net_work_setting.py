# decoder_config.py
import os

script_directory = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在文件夹的绝对路径
parent_directory = os.path.dirname(script_directory)  # 获取上级目录的路径
print(parent_directory)
h5_path = os.path.join(parent_directory, 'train_data')
save_base_dir = os.path.join(parent_directory, "decoder", "cache")
resume_from = os.path.join(save_base_dir, )

decoder_cfg = {
    # ============================================================
    # Checkpoint / path settings
    # ============================================================
    "paths": {
        "lewm_ckpt_path": r"C:\Users\wangh\Desktop\world_model\lewd2\cache\20260522_180443\lewm_best.ckpt",
        "h5_path": r"C:\Users\wangh\Desktop\world_model\le-wm\train_data",
        "save_base_dir": r"C:\Users\wangh\Desktop\world_model\le-wm\decoder\cache",
        "output_name": "visualization_decoder",
        "resume_from": r"C:\Users\wangh\Desktop\world_model\le-wm\decoder\cache\20260524_121226\visualization_decoder_best.pt",
        # "resume_from": r"C:\Users\wangh\Desktop\world_model\le-wm\decoder_ckpts\visualization_decoder_last.pt",
    },

    # ============================================================
    # Decoder network settings
    # ============================================================
    "model": {
        "latent_dim": 192,
        "hidden_dim": 256,
        "image_size": 224,
        "patch_size": 16,
        "channels": 3,
        "depth": 4,
        "heads": 8,
        "dim_head": 64,
        "mlp_dim": None,
        "dropout": 0.0,
        "num_latent_tokens": 16,
    },

    # ============================================================
    # Training settings
    # ============================================================
    "train": {
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "save_every_steps": None,
        # "save_every_steps": 500,
    },

    # ============================================================
    # Loss settings
    # ============================================================
    "loss": {
        "use_foreground_weight": True,
        "foreground_weight": 10.0,
        "background_weight": 1.0,
        "foreground_threshold": 0.15,
        "l2_weight": 0.1,

        # ImageNet white background in normalized space:
        # white_norm = (1 - mean) / std
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    },

    # ============================================================
    # Inference / visualization settings
    # ============================================================
    "infer": {
        "decoder_ckpt_path": r"C:\Users\wangh\Desktop\world_model\le-wm\decoder_ckpts\visualization_decoder_best.pt",
        "save_dir": r"C:\Users\wangh\Desktop\world_model\le-wm\decoder_infer_result",
        "infer_index": 0,
        "max_save_images": 8,
        "unnormalize_imagenet": True,
    },
}