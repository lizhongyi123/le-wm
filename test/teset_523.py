# decoder_config.py
import os

script_directory = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在文件夹的绝对路径
parent_directory = os.path.dirname(script_directory)  # 获取上级目录的路径
print(parent_directory)
h5_path = os.path.join(parent_directory, 'train_data')
save_base_dir = os.path.join(parent_directory, "decoder", "cache")
print(save_base_dir)

save_base_dir = os.path.join(parent_directory, "decoder\cache",)
print(save_base_dir)
