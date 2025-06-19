import os
import random

# 设置数据集目录和保存文件名的路径
data_dir = '/home/admin1/brats/2021/SCnet/Dataset/BraTS2021_2/TrainingData/'  # 将路径替换为实际数据集的路径
save_dir = '/home/admin1/brats/2021/SCnet/Dataset/BraTS2021_2/'     # 将路径替换为保存划分文件名的路径

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 获取所有文件名列表
file_names = os.listdir(data_dir)
random.shuffle(file_names)  # 随机打乱文件名顺序

# 计算划分的索引
total_files = len(file_names)
train_split = int(0.8 * total_files)
valid_split = int(0.1 * total_files)

# 划分数据集
train_files = file_names[:train_split]
valid_files = file_names[train_split : train_split + valid_split]
test_files = file_names[train_split + valid_split:]

# 保存文件名到相应的文本文件
def save_file_names(file_list, file_path):
    with open(file_path, 'w') as f:
        for file_name in file_list:
            f.write(file_name + '\n')

save_file_names(train_files, os.path.join(save_dir, 'train.txt'))
save_file_names(valid_files, os.path.join(save_dir, 'valid.txt'))
save_file_names(test_files, os.path.join(save_dir, 'test.txt'))

print("数据集划分后的文件名已保存。")
