source .venv/bin/activate
project_root=./test

sample_step=40
n_motion=2
n_frame=16
extrapolation_time=16

test_data=datasets/inference/inference_data.csv
max_protein_num=100
train_step=400000 

model_path=step_400000.pth

noise_scale=1.0
save_root=${project_root}/inference/

CUDA_VISIBLE_DEVICES=0 python -m DyneTrion.inference_DyneTrion eval.weights_path=${model_path} \
experiment.use_ddp=False \
eval.extrapolation_time=${extrapolation_time} \
eval.name="test" \
experiment.batch_size=1 \
experiment.noise_scale=${noise_scale} \
experiment.base_root=${save_root} \
experiment.num_loader_workers=1 \
data.frame_time=${n_frame} data.motion_number=${n_motion} data.frame_sample_step=${sample_step} \
data.test_csv_path=${test_data} \
data.max_protein_num=${max_protein_num}

