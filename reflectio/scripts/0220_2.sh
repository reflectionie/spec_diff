cd ..

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule squaredcos_cap_v2 \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.000192 \
  --beta_end 0.0384 \
  --num_train_epochs 1

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule squaredcos_cap_v2 \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 1

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule squaredcos_cap_v2 \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.1 \
  --num_train_epochs 1