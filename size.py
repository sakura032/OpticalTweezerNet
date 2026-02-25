# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2019

@author: admin
"""
from tensorflow import keras
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.pyplot.switch_backend('agg')
#import spectra_process.subpys as subpys
#import scipy.optimize as optimize
import os
#import time


def run_evaluation():

    # Set default decvice: GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # if set '-1', python runs on CPU, '0' uses 1060 6GB
    # In[] CNN preprocessing: 2-2
    source_data_path = './data/EV/'
    save_train_model_path = './TweezerNet/EV/'

    # load experimental dataset # 加载数据
    try:
        X = np.load(source_data_path+'X_train.npy')  #X_mean/X_std：训练时保存的标准化均值/方差
        Y = np.load(source_data_path+'Y_train.npy')
    except FileNotFoundError as e:
        print(f"数据文件缺失：{e}")
        exit(1)
    #load Resnet or CNN model
    print('1. Finish loading original test dataset!') 
        
    # 新增加验证集切分 --- （仅保留最后10%作为模型未见过的验证数据）
    #X = np.real(X).astype('float32')
    validation_ratio = 0.1
    split_marker = np.int64(np.round((1 - validation_ratio) * Y.shape[0]))
    #X, Y = X[split_marker:, :], Y[split_marker:]  # 直接覆盖变量，只取验证集
    # 只验证模型“没见过”的数据
    X_val = np.real(X[split_marker:, :]).astype('float32')  # 重命名为X_val，避免覆盖原始变量
    Y_val = Y[split_marker:].astype('float32').flatten() #必须确保都是一维
    print('2. Finish splitting validation/test dataset (1/10 of training data)!')  # 对应“切分测试集”步骤

    #  标准化均值/方差
    X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
    X_std = np.load(save_train_model_path+'X_scale_std.npy')
    print('3. Finish loading scale mean and std!')  # 对应“加载标准化参数”步骤

    # 1.加载ResNet/CNN模型
    try:
        model = keras.models.load_model(save_train_model_path+'regression_model.h5')
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit(1)
    print('4. Finish loading the model!')  

        
    #3.模型预测：
    #    - 重塑X维度（适配Conv1D输入）
    #    - YPredict：模型输出的预测尺寸
    X_val_norm = (X_val - X_mean) / X_std #数据标准化：和训练数据用相同的均值/方差（关键！否则预测偏差大）
    X_val_reshaped = np.reshape(X_val_norm, (X_val_norm.shape[0], X_val_norm.shape[1], 1)) # 重塑维度适配Conv1D输入（(样本数, 序列长度, 通道数)）
    YPredict = model.predict(X_val_reshaped, verbose=0)  # verbose=0关闭预测进度条
    # 关键：将YPredict从二维(N,1)转为一维(N,)，解决维度不匹配
    YPredict_flat = YPredict.squeeze(axis=1)
    print('5. Finish predicting!') 
    print(f" 预测结果维度：{YPredict_flat.shape}(原始：{YPredict.shape})")

    # 计算预测误差：
    #rmse = np.sqrt(((Y-YPredict)**2).mean()) 
    rmse = np.sqrt(((Y_val - YPredict_flat) ** 2).mean())   # RMSE（均方根误差）：衡量预测精度（越小越好）
    print(f"模型预测RMSE均方根误差: {rmse:.4f} nm")
    #r1= np.sqrt(((Y-YPredict)**2).mean())  
    #rmse=r1  #冗余赋值（r1=rmse）：不影响功能，可删除

    # 线性拟合（x=真实尺寸，y=预测尺寸）
    fontsize_val = 19
    x = Y_val.flatten()
    y = YPredict_flat.flatten()
    a, b = np.polyfit(x, y, 1)  # 拟合直线：y = a*x + b    

    # 绘图：
    plt.figure(figsize=(8, 8)) # 创建画布
    #print (rmse)
    plt.scatter(x, y, alpha=0.6, s=20)  # alpha=0.6解决点重叠，s=20调小点大小 scatter：真实值vs预测值散点图
    #print(x)
    #print(y)
    plt.plot(x,a*x+b, color='red',linewidth=4)   #plot：红色拟合直线（直观展示预测偏差）
    #保存预测值CSV
    dataframe= pd.DataFrame({
        'True_Size(nm)': Y_val,
        'Predicted_Size(nm)': YPredict_flat.flatten()
    })
    dataframe.to_csv('./output/train_status/binary_train/Res_predicted.csv')  #保存预测结果：将预测值写入CSV（后续分析用）

    #设置坐标轴标签（X=真实尺寸，Y=预测尺寸）
    plt.xlabel('True size (nm)', fontsize=fontsize_val)
    plt.ylabel('Predicted size (nm)', fontsize=fontsize_val)
    plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val-2)
    plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val-2)
    plt.text(0.15, 0.85, f'RMSE = {rmse:.4f}', fontsize=fontsize_val, transform=plt.gca().transAxes)  # transform=plt.gca().transAxes：基于画布的相对坐标（0-1）
    #plt.text(0.85, 0.05, 'RMSE = %f'%r1, fontsize=fontsize_val)  
    plt.title('Network vs. Actual size  ResNet-50 NTA prediction', fontsize=fontsize_val)
    plt.tight_layout()  # 自动调整布局，避免标签被截断

    # 注释代码：加载学习率日志（未使用）
    #CSV_Path = './output/train_status/binary_train/'
    #lr = pd.read_csv(CSV_Path+'run-train-tag-epoch_lr.csv').values


    # 1保存/显示图片：
    #     - savefig：保存高清图片（dpi=300）到指定路径
    #     - show：GUI环境下显示图片（agg后端无效果，仅保存即可）
    plt.savefig('./output/prediction/ResNet_20mn.png', dpi=300, bbox_inches='tight')
    plt.show()  


if __name__ == "__main__":
    run_evaluation()

