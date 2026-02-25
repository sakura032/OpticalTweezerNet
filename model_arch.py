# model_arch.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Input, Add, \
                                    Activation, BatchNormalization, \
                                    AveragePooling1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def idn_block(input_tensor):
    """恒等块：输入和输出尺寸一致时使用"""
    X_shortcut = input_tensor
    
    # 三层卷积加工
    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    # 核心：将原始输入加回来
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def conv_block(input_tensor):
    """卷积块：用于每个 Stage 的开头，调整捷径维度"""
    X_shortcut = input_tensor

    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    # 捷径上也做一次卷积，保证维度匹配
    X_shortcut = Conv1D(filters=64, kernel_size=5, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def build_resnet50(input_shape, learning_rate=5e-5):
    """构建完整的 ResNet50 架构"""
    X_input = Input(input_shape)
    
    # Stage 1: 预处理
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D((1, 3), strides=(1, 2))(X)

    # Stage 2
    X = conv_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 3
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 4 (最深的一层)
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

    X = AveragePooling1D((1,2), name='avg_pool')(X)
    X = Flatten()(X)
    
    # 最终输出层：预测尺寸数值
    X = Dense(1, activation='relu')(X)

    model = Model(inputs = X_input, outputs = X)
    
    # 编译模型
    optim = Adam(learning_rate=learning_rate, epsilon=1e-8)
    model.compile(loss='mse', optimizer=optim, metrics=['mae'])
    
    return model