run_name="URM-SwiGLU"
checkpoint_path="/volume/pt-train/users/ztgao/loop_arcagi/checkpoints/${run_name}" 
mkdir -p $checkpoint_path

source activate 
conda activate /volume/pt-train/users/ztgao/miniconda3/envs/arc  
cd /volume/pt-train/users/ztgao/loop_arcagi
swanlab login --host https://swanlab-longmen.siflow.cn -k QEFqFRU1GYVcqrTVEOYve
export WANDB_API_KEY=local-16b5d3b31f9577b6184a756ab611ed0342d17dc4 
export WANDB_BASE_URL=http://wandb-app.t-skyinfer-phwu.svc:8080   

torchrun --nproc-per-node 8 pretrain.py \
data_path=/volume/pt-train/users/ztgao/tmp/TinyRecursiveModels/data/arc1concept-aug-1000 \
arch=looped_transformer_v22 arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
epochs=100000 \
eval_interval=2000 \
puzzle_emb_lr=1e-4 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path > /volume/pt-train/users/ztgao/loop_arcagi/nonlinearity/swiglu.log 2>&1 &