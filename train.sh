export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_left.py
NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file

