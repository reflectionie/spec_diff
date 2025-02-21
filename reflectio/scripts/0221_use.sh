python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 2e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 5e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

# ############################################### beta  end


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.05 \
  --num_train_epochs 3


  
python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.02 \
  --num_train_epochs 3



# ############################################### model / large lr


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v3_concat \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v4_concat \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3



python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat_std_gt \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v3_concat_std_gt \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v4_concat_std_gt \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat_3hidden \
  --learning_rate 1e-4 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3




# ====================== batch size
python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \[]
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 5e-6 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 8192 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 8192 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 5e-6 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 8192 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 2.5e-6 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3



# ===================== new model

python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v5_concat_diff \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


python train_hidden_state_diffuser_v3_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 32768 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v6_concat_emb \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3


# +====================# new loss

python train_hidden_state_diffuser_v4_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3 \
  --lm_loss_weight 1.0


python train_hidden_state_diffuser_v4_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3 \
  --lm_loss_weight 3


python train_hidden_state_diffuser_v4_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3 \
  --lm_loss_weight 1.5


python train_hidden_state_diffuser_v4_scheduler.py \
  --data_dir /home/5/uu02155/data/llama/eagle_new/eagle/reflectio/draft_train_data \
  --beta_schedule linear \
  --basepath /home/5/uu02155/data/llama/eagle_new/base_model/Meta-Llama-3-8B-Instruct \
  --checkpointing_steps 100 --train_batch_size 16384 --warmup_steps 1000 \
  --model_module my_hidden_state_diffusion_v1_concat \
  --learning_rate 1e-5 \
  --time_embed_dim 64 \
  --beta_start 0.0001 \
  --beta_end 0.01 \
  --num_train_epochs 3 \
  --lm_loss_weight 0.5
