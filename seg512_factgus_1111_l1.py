
target_type = "noise-similarity"
model_name = "penob_alldata".format(target_type)
model_weight = None
workers = 36 
epochs = 400
start_epoch = 0
batch_size = 16 
crop_size = 128                
num_channel = 1
num_sim = 1 
num_select = 1  
print_freq = 10
test_freq = 5

resume = None
world_size = 1
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "gloo"  #nccl
seed = None
gpu = 0
multiprocessing_distributed = False


data_train = dict(
    type="lmdb",
    
    lmdb_file='......',
    meta_info_file='......',
   
    crop_size=crop_size,
    target_type=target_type,
    random_flip=False,
    prune_dataset=None,
    num_sim=num_sim,
    num_select=num_select,
    load_data_all=False,
    incorporate_noise=False,
    dtype="float32",
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type="bsd_npy",
    data_file='/home/dell/disk/dengwen_exp_data/H8/NQLP_PSTM_GAIN_2022_0328_t2000_3022.npy',
    target_file='/home/dell/disk/dengwen_exp_data/H8/NQLP_PSTM_GAIN_2022_0328_t2000_3022.npy',


    norm=None,    
    shuffle=True,
    ims_per_batch=37,
    train=False,
)

model = dict(
    type="common_denoiser",
    base_net=dict(
        type="unet2",
        n_channels=1,
        n_classes=1,
        activation_type="relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),

    denoiser_head=dict(
        head_type="supervise",
        loss_type="l2",
        loss_weight={"l2": 1},
    ),

    weight= None,
)

solver = dict(
    type="adam",
    # base_lr=0.0001,
    base_lr=0.0001,
    bias_lr_factor=1,
    betas=(0.1, 0.99),
    weight_decay=0,
    weight_decay_bias=0,
    lr_type="ramp",
    max_iter=epochs,
    ramp_up_fraction=0.1,
    ramp_down_fraction=0.3,
)

results = dict(
    output_dir="./resultss/{}".format(model_name),
)