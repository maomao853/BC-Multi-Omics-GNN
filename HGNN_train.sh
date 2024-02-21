export CUDA_LAUNCH_BLOCKING=1

python train.py --model_filename checkpoint/deepglint_sampler.pth --knn_k 5 --levels 5 --hidden 128 --epochs 200 --lr 0.01 --batch_size 64 --num_conv 3 --balance --use_cluster_feat