import os

if __name__ == '__main__':
    os.system("CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node 2 ddp_main.py  --batch_size=64 --epoch=20 --load_from='./ckpt/19.ckpt' ")