import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, Add, \
                         Activation, BatchNormalization, \
                         AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
import pandas as pd
import os
#from sklearn.metrics import mean_squared_log_error


#Architecture based off of the paper: 'Deep Residual Learning for Image Recognition'.

# 模块2：GPU训练环境配置 Set default decvice: GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 建议加上这行，防止 TF 2.16.1 预占所有显存导致 OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# 在模块 2 之后添加
#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)
#增强版
#from tensorflow.keras import mixed_precision
#try:
#    policy = mixed_precision.Policy('mixed_float16')
#    mixed_precision.set_global_policy(policy)
#    print(f'已开启混合精度加速！计算类型: {policy.compute_dtype}, 变量类型: {policy.variable_dtype}')
#except Exception as e:
#    print(f"混合精度配置失败，将使用标准精度运行: {e}")
    
#模块3：恒等残差块 idn_block（ResNet的核心积木1）
def idn_block(input_tensor):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    #F1, F2, F3 = filters
    
    # Save the input value to later add back to the main path. # 保存输入用于快捷连接
    X_shortcut = input_tensor  
    
    # First component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    # Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])    # 论文核心：快捷连接（Shortcut connection）
    X = Activation('relu')(X)
   
    return X
    

#模块 4：卷积残差块 conv_block（ResNet 的核心积木 2）
def conv_block(input_tensor):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    #F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = input_tensor

    # First component of main path 
    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv1D(filters=64, kernel_size=5, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

#模块 5：学习率衰减函数 step_decay 对应论文里的优化器设置
def step_decay(epoch):
    lr = 5e-5 # 1e-3
    drop_factor = 0.1
    drop_period = 20 # 20
    lrate = lr*np.power(drop_factor, np.floor((1+epoch)/drop_period))   # 将 np.math.floor 改为 np.floor
#    decay_rate.append(lrate)
    return lrate
#新版本 numpy 里np.math.pow已经被弃用，改成np.power就不会有警告了
    

#模块 6：ResNet50 主网络构建函数
def ResNet50(input_shape, output_shape=1, dropout_rate=0.8, learning_rate=5e-5):
    """
    Implementation of the popular ResNet50 the following architecture:
    
    """
    
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Stage 1 预处理
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # 原来的X = MaxPooling1D((1, 3), strides=(1, 2))(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = conv_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 3
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 4 最深的一层
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 5
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)

    # AVGPOOL 
    # 原来的X = AveragePooling1D((1,2), name='avg_pool')(X)
    X = AveragePooling1D(2, name='avg_pool')(X)
    
    # Flatten
    X = Flatten()(X)
    X = Dropout(dropout_rate, seed=0, name='dropout_final')(X)
    
    # Add extra dense layers
    #if extra_layers is not None:
        #assert len(extra_layers) == len(dropouts), \
               # "Arguments do Not match in length: extra_layers, dropouts."
        #for i, layer, dpout in (zip(range(len(extra_layers)), extra_layers, dropouts)):
           # X = Dense(layer, name='fc_'+str(i)+'_'+str(layer), activation='relu',
               #       kernel_initializer=glorot_uniform(seed=0))(X)
            #X = Dropout(dpout, seed=0, name='dropout_'+str(i)+'_'+str(dpout))(X)

    # Output  # 最终输出层：预测尺寸数值
    X = Dense(1, activation='linear')(X)

    
    # Create model
    model = Model(inputs = X_input, outputs = X)
    
    # Compile model
    learning_rate = 1e-5
    # TF 2.x 中 lr 已改为 learning_rate
    optim = Adam(learning_rate=learning_rate, epsilon=1e-8)
    # 必须传入 optim 对象，而不是字符串 'adam'
    model.compile(loss='mse', optimizer=optim, metrics=['mae'])

    
    return model

#模块 7：全局训练超参数设置
max_epoch = 50  #整个训练数据集要完整遍历 50 次，和论文一致  
validation_ratio=0.1  #：拿出 10% 的训练数据当验证集，训练过程中看模型在没见过的数据上的表现，防止过拟合，和论文里 9:1 的划分一致
batch_size = 32  #每次给模型喂 32 条数据，更新一次参数，和论文一致 RTX 5060 Laptop GPU 显存是 8GB先改为16
learning_rate = 5e-5  #初始学习率，和前面的衰减函数对应
dropout_rate = 0.5 # 0.2   
source_data_path = './data/EV/'   #训练数据存放的路径，里面要有X_train.npy（输入的位移序列）和Y_train.npy（对应的真实粒径）
save_train_model_path = './TweezerNet/EV/'  #训练好的模型、标准化参数、训练日志保存的路径
#模块 8：训练数据加载与预处理
#1 read in train datasets and get sizes
print("Step 1: read in train and test data")
X = np.load(source_data_path+'X_train.npy')
Y = np.load(source_data_path+'Y_train.npy')
X = np.real(X).astype('float32')# 这一行必须加上！
split_marker = np.int64(np.round((1-validation_ratio)*Y.shape[0]))

# 2 normalization: zero-mean along column, with_std=False  数据标准化：零均值、单位方差，只用训练集拟合标准化参数
X_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X[:split_marker, :])
X[:split_marker, :] = X_scaler.transform(X[:split_marker, :])
X[split_marker:, :] = X_scaler.transform(X[split_marker:, :])

#3 保存标准化的均值和方差，预测的时候必须用同样的参数处理数据
np.save(save_train_model_path+'X_scale_mean.npy', X_scaler.mean_)
np.save(save_train_model_path+'X_scale_std.npy', X_scaler.scale_)

# 4. 划分训练集和验证集
x_train = X[:split_marker, ]
y_train = Y[:split_marker, ]
x_test  = X[split_marker:,]
y_test  = Y[split_marker:, ]

    
# 5 reshape train data  数据维度重塑，适配Conv1D的输入要求
X = np.reshape(X, [X.shape[0], X.shape[1], 1])
    
# 6 define train size  打印输入形状，方便检查
input_shape = X.shape[1:]
output_shape = 1

print(input_shape)
print(X.shape)
print(X[:split_marker, :].shape)
print(X[split_marker:, :].shape)




# 模块 9：模型训练与断点续训 (保留原始输出风格修正版)
# 1. 统一准备回调函数 (确保逻辑正确)
lrate = keras.callbacks.LearningRateScheduler(step_decay)
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_train_model_path + 'regression_model.h5', 
    verbose=1, 
    save_best_only=True
)
tbCallBack = keras.callbacks.TensorBoard(histogram_freq=0, write_graph=True, write_images=True)

# 在定义 tbCallBack 附近添加：
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # 监控验证集损失
    patience=10,            # 如果连续 10 个 Epoch 验证集都没进步，就停止
    restore_best_weights=True # 停止后，自动把模型的权重恢复到表现最好的那一轮
)


# 2. 根据是否存在旧模型进行逻辑分支
if os.path.exists(save_train_model_path + 'regression_model.h5'):
    # --- 以下保留你的原始 print 输出 ---
    model = keras.models.load_model(save_train_model_path + 'regression_model.h5')
    model.summary()
    print(model.summary) # 保留你原来的写法
    print('Load saved model and train again!!!')
    print("###################################################################")
    print("Step 3: train saved Resnet model")
    # ----------------------------------
else:
    # 如果是新模型，正常构建
    print("未检测到预训练模型，开始从头训练...")
    model = ResNet50(input_shape, output_shape=1, dropout_rate=0.8, learning_rate=learning_rate)


# 3. 统一执行训练 (修复了 history 记录和学习率调度)(已加入 early_stop 监控验证集)
# 无论加载还是新建，都统一使用 callbacks=[lrate, checkpointer, tbCallBack]
history = model.fit(
    X[:split_marker, ], Y[:split_marker, ], 
    batch_size=batch_size, 
    epochs=max_epoch, 
    validation_data=(X[split_marker:, ], Y[split_marker:, ]), 
    # 注意这里：要把早停变量 early_stop 名字加进这个列表
    callbacks=[lrate, checkpointer, tbCallBack, early_stop] 
)

# 4. 统一保存日志 (修复了 history 报错问题)
print("正在生成训练日志 CSV 文件...")
train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
train_lr = np.array(history.history['lr'])
epoch = np.arange(len(train_loss))

df_train_loss = pd.DataFrame({'Step': epoch, 'Value': train_loss})
df_train_loss.to_csv("run-train-tag-epoch_loss.csv", index=False, sep=',')

df_val_loss = pd.DataFrame({'Step': epoch, 'Value': val_loss})
df_val_loss.to_csv("run-validation-tag-epoch_loss.csv", index=False, sep=',')

df_train_lr = pd.DataFrame({'Step': epoch, 'Value': train_lr})
df_train_lr.to_csv("run-train-tag-epoch_lr.csv", index=False, sep=',')

print('Finish training!!!')