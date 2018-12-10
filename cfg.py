import os

train_task_id = '3T512'  # 后面3位设置输入图片的大小
initial_epoch = 0
epoch_num = 50   # epoch 数
lr = 1e-4    # 学习率
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip

patience = 12   # 最大验证集容忍次数
load_weights = True   # fine-tuning
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000  
validation_split_ratio = 0.1   # 分割系数

#------------------------------自定义参数----------------------------
train_val_split_control = True   # 如果为真，说明训练和测试在同一个目录，要用上面的分割系数
								 # 如果为假，说明不在同一个目录，要用到下面的自定义目录

test_origin_image_dir_name = 'image_1000/'
test_origin_txt_dir_name = 'txt_1000/'

#--------------------------------------------------------------------



max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400

assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 4
else:
    batch_size = 4
	
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

origin_data_dir = '/home/zhouxd/Data/IDCard/'  # 原始数据地址 
data_dir = 'IDCard/'                           # 存放训练数据的地址

origin_image_dir_name = 'img/'
origin_txt_dir_name = 'img/'


train_image_dir_name = 'images_%s/' % train_task_id   # 训练图片总目录
train_label_dir_name = 'labels_%s/' % train_task_id	  # 训练图片标签总目录

show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
gen_origin_img = True

draw_gt_quad = False
draw_act_quad = False
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.3
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)   # 5->2
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)

# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id

# 该次存储的权重参数文件地址
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                % train_task_id
								
# 上次存储的权重参数文件地址
last_saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                % '3T512'				

pixel_threshold = 0.8  # predit 的 NMS 阈值

side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True
