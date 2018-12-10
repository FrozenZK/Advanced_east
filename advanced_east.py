import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import cfg
from network import East
from losses import quad_loss
from data_generator import gen

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置GPU占用哪一个

config = tf.ConfigProto()
config.gpu_options.allow_growth=True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

east = East()
east_network = east.east_network()
east_network.summary()

# 设置 loss 函数
east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay))

# fine-tuning						
if cfg.load_weights and os.path.exists(cfg.last_saved_model_weights_file_path):
    print("ggggggggggggggggggggggggggggggggggggggggggggg")
    print("load : " + cfg.last_saved_model_weights_file_path)
    east_network.load_weights(cfg.last_saved_model_weights_file_path)   # 上一次加载权重参数

east_network.fit_generator(generator=gen(),    # 产生训练集
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),   # 产生验证集
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,   # 设置初始epoch数
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
