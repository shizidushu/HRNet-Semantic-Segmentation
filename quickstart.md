


```bash
conda create --name hrnet python=3.8
conda activate hrnet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```



python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/people_seg/seg_hrnet_w48_512x512_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml