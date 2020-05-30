#raw/ initial data path
path_idl_files = ["data\\train-210\\train-210.idl",
                  'data\\train-400\\train-400.idl',
                  'data\\tud-pedestrians\\tud-pedestrians.idl']
path_image_folders = ["data\\train-210\\", 'data\\train-400\\', 'data\\tud-pedestrians\\']
#path for total positive instances
total_pos_im_path = 'data\\total_pos\\'
#path for total negative instances
total_neg_im_path = 'data\\total_neg\\'
#path for training positive
train_pos_im_path = 'data/images/train_pos'
#path for training negative
train_neg_im_path = 'data/images/train_neg'
test_pos_im_path = 'data/images/test_pos'
test_neg_im_path = 'data/images/test_neg'
min_wdw_sz = [100, 200]
step_size = [10, 10]
image_height=200
image_width=100
orientations = 9
pixels_per_cell = [8,8]
cells_per_block = [2, 2]
train_pos_feat_ph = 'data/features/pos'
train_neg_feat_ph = 'data/features/neg'
test_pos_feat_ph = 'data/features/neg'
test_neg_feat_ph = 'data/features/neg'
model_path = 'data/models'
threshold = 0.3