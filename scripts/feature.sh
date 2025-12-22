# data_paths=(
#   "/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json"
#   "/home/fwj/workspace/code/UniTime/UniTime_data/charades/test.json"
#   "/home/fwj/workspace/code/UniTime/UniTime_data/charades/test.json"
# )
# data_paths=(
#   "/home/fwj/workspace/code/UniTime/UniTime_data/anet/train.json"
#   "/home/fwj/workspace/code/UniTime/UniTime_data/anet/test.json"
# )
# data_paths=(
#   "/home/fwj/workspace/code/UniTime/UniTime_data/anet/train_x.json"
#   "/home/fwj/workspace/code/UniTime/UniTime_data/anet/test_x.json"
# )
data_paths=(
  "/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/train.json"
  "/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/val.json"
  "/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/test.json"
)


export DECORD_EOF_RETRY_MAX=20480
# gpu_list=(4 5 6 7)
# part_list=(0 1 2 3)
gpu_list=(2 3)
# part_list=(0 1)
part_list=(2 3)
# model_local_path=/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct
model_local_path=/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-7B-Instruct
# charads
# feat_root=/home/fwj/workspace/VisualSearch/unitime_feat
# video_root=/home/fwj/workspace/VisualSearch/charades/Charades_v1
# # anet
# feat_root=/raid0/fwj/VisualSearch/unitime_feat
# video_root=/raid5/hl/VisualSearch/activitynet/VideoData
# qvhl
feat_root=/raid0/fwj/VisualSearch/unitime_feat
video_root=/raid5/fwj/VisualSearch/qvhighlight/videos

# You can adjust the parallelism by modifying the num_parts parameter, and update gpu_list and part_list to assign GPUs to each process
for data_path in "${data_paths[@]}"; do
  for i in ${!gpu_list[@]}; do
    python feature_offline.py --data_path $data_path --part ${part_list[$i]} --gpu ${gpu_list[$i]} --num_parts 4 --model_local_path $model_local_path --feat_root $feat_root --video_root $video_root &
  done
  wait
done