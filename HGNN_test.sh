export CUDA_LAUNCH_BLOCKING=1

python test.py --model_filename checkpoint/deepglint_sampler.pth --knn_k 5 --tau 0.8 --level 5 --threshold prob --faiss_gpu --hidden 128 --num_conv 3 --batch_size 64 --early_stop --use_cluster_feat