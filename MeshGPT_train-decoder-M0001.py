#!/usr/bin/env python
# coding: utf-8

# In[1]:


project_name = "model_M0001"
dataset_path = f"datasets/objverse_shapenet_modelnet_max_250faces_186M_tokens.npz"
encoder_parameters_path = f'{project_name}/2024-07-23-M0001-encoder-loss_0.170.pt'
restart_training_checkpoint_path = None

import torch
import trimesh
import numpy as np
import os
import csv
import json
from collections import OrderedDict

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
) 


# In[2]:


from pathlib import Path 
import gc     
import os
from meshgpt_pytorch import MeshDataset 
 
working_dir = f'{project_name}'
working_dir = Path(working_dir)
working_dir.mkdir(exist_ok = True, parents = True)
    
print(f"Loading {dataset_path}")
dataset = MeshDataset.load(dataset_path) 
print(f"Loaded datasset with keys {dataset.data[0].keys()}")


# In[3]:


autoencoder = MeshAutoencoder(     
    decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,   
    dim_codebook = 192,  
    dim_area_embed = 16,
    dim_coor_embed = 16, 
    dim_normal_embed = 16,
    dim_angle_embed = 8,    
    attn_decoder_depth  = 4,
    attn_encoder_depth = 2
).to("cuda")
    
total_params = sum(p.numel() for p in autoencoder.parameters()) 
total_params = f"{total_params / 1000000:.1f}M"
print(f"Encoder Total parameters: {total_params}")


# In[4]:


pkg = torch.load(encoder_parameters_path) 
autoencoder.load_state_dict(pkg['model'])
print(f"Loaded encoder parameters from {encoder_parameters_path}")
del pkg

# for param in autoencoder.parameters():
#      param.requires_grad = True


# In[5]:


import gc  
torch.cuda.empty_cache()
gc.collect()   
max_seq = max(len(d["faces"]) for d in dataset if "faces" in d)  * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers) 
print("Max token sequence:" , max_seq)  

# # GPT2-Small model
transformer = MeshTransformer(
    autoencoder,
    dim =768,
    coarse_pre_gateloop_depth = 6,  
    fine_pre_gateloop_depth= 4, 
    attn_depth = 24,  
    attn_heads = 16,
    dropout  = 0.0,
    max_seq_len = max_seq,
    condition_on_text = True, 
    gateloop_use_heinsen = False,
    text_condition_model_types = "bge", 
    text_condition_cond_drop_prob = 0.0, 
).to("cuda") 

total_params = sum(p.numel() for p in transformer.decoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Decoder total parameters: {total_params}")


# In[6]:


labels = list(set(item["texts"] for item in dataset.data))
dataset.embed_texts(transformer, batch_size = 25)
dataset.generate_codes(autoencoder, batch_size = 50)
print(dataset.data[0].keys())


# **Train to about 0.0001 loss (or less) if you are using a small dataset**

# In[7]:


batch_size = 3 # Max 64
grad_accum_every = 20

# Set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  4 * 16 = 64
learning_rate = 1e-2 # Start training with the learning rate at 1e-2 then lower it to 1e-3 at stagnation or at 0.5 loss.

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,num_train_steps=100, dataset = dataset,
                                 grad_accum_every=grad_accum_every,
                                 learning_rate = learning_rate,
                                 batch_size=batch_size,
                                 checkpoint_every_epoch = 1,
                                 # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                                 # However it might cause nan after a while.
                                 # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7} 
                                 )

print("Training starting")
#trainer.load('checkpoints/mesh-transformer.ckpt.epoch_8_avg_loss_0.927.pt')

loss = trainer.train(300, stop_at_loss = 0.5)  
trainer.save(f'{working_dir}/mesh-transformer_lr-2.pt')


