import os
import numpy as np
import nibabel as nib

# 指定文件夹路径
root_folder = "/home/admin1/brats/2021/SCnet/Dataset/BraTS2021_2/TrainingData/"

# 遍历每个文件夹
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
    
    # 获取标签文件路径列表
    label_files = [filename for filename in os.listdir(folder_path) if filename.endswith("seg.nii.gz")]
    
    # 遍历处理每个标签文件
    for label_file in label_files:
        label_file_path = os.path.join(folder_path, label_file)
        
        # 加载标签图像
        label_img = nib.load(label_file_path).get_fdata()
        
        # 将类别 1、2 和 4 合并为类别 1
        label_img[(label_img == 1) | (label_img == 2) | (label_img == 4)] = 1
        
        # 保存处理后的标签图像，直接覆盖原始文件
        nib.save(nib.Nifti1Image(label_img, affine=None), label_file_path)
        
print("标签合并完成。")
