import cv2 as cv
import os
import time
import mmcv
import shutil
from predict import predict_single_person
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
input_video = 'dataset/moving.mp4'
output_path = 'output_pred.mp4'

temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))

imgs = mmcv.VideoReader(input_video)
weights_path = "/home/jvm/HRNet/multi_train/model_5.pth"
#prog_bar = mmcv.ProgressBar(len(imgs))

for frame_id, img in enumerate(imgs):
    
    ## 处理单帧画面
    img = predict_single_person(img,weights_path,device)
    # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
    img.save(f'{temp_out_dir}/{frame_id:06d}.jpg', "BMP")
    
    #prog_bar.update() # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)
print('视频已生成', output_path)