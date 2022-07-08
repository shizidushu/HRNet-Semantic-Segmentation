# %%
# python -m torch.distributed.launch --nproc_per_node=1 get_output_shape.py
import sys
sys.path.append('./lib/')

import torch
import torch.backends.cudnn as cudnn

# %%
from config import config

# %%
config.merge_from_file('/home/ubuntu/data/yong/projects/HRNet-Semantic-Segmentation/experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')


cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.deterministic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED
gpus = list(config.GPUS)
distributed = True
if distributed:
    device = torch.device('cuda:{}'.format(0))    
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )  


# %%
from models.seg_hrnet_ocr import get_seg_model

# %%
model = get_seg_model(config)

# %%
model = model.cuda()

# %%


# %%
x = torch.rand((1, 3, 473, 473)).cuda()

# %%
outputs = model(x)

# %%
for output in outputs:
    print(output.shape)

# %%



