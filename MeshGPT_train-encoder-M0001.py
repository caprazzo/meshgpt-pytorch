#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from pathlib import Path 
import gc     
import os
from meshgpt_pytorch import MeshDataset 


# In[ ]:


project_name = "model_M0001"
dataset_path = f"datasets/objverse_shapenet_modelnet_max_250faces_186M_tokens.npz"

working_dir = f'{project_name}'
working_dir = Path(working_dir)
working_dir.mkdir(exist_ok = True, parents = True)

print(f"Loading dataset {dataset_path}")
dataset = MeshDataset.load(dataset_path) 
print(f"Loaded with keys {dataset.data[0].keys()}")


# In[ ]:


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
print(f"Total Encoder parameters: {total_params}")


# In[ ]:


#pkg = torch.load(str(f'proj_large/20240718-mesh-encoder-loss_0.157.pt')) 
#autoencoder.load_state_dict(pkg['model'])
#for param in autoencoder.parameters():
#      param.requires_grad = True


# **Train to about 0.3 loss if you are using a small dataset**

# In[ ]:


batch_size=16 # The batch size should be max 64.
grad_accum_every = 4
# # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
learning_rate = 1e-3 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

autoencoder.commit_loss_weight = 0.4 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                             batch_size=batch_size,
                                             grad_accum_every = grad_accum_every,
                                             learning_rate = learning_rate,
                                             checkpoint_every_epoch=1) 

autoencoder_trainer.load(f'{working_dir}/2024-07-23-MOOO1-encoder-loss_0.273.pt')
print("Start Encoder Training")

loss = autoencoder_trainer.train(480, stop_at_loss = 0.35, diplay_graph=False)    
autoencoder_trainer.save(f'{working_dir}/2024-07-23-M0001-encoder-2.pt')   


# ### Inspect how the autoencoder can encode and then provide the decoder with the codes to reconstruct the mesh

# In[ ]:


import torch
import random
from tqdm import tqdm 
from meshgpt_pytorch import mesh_render 

min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
total_mse, sample_size = 0.0, 200

random.shuffle(dataset.data)

for item in tqdm(dataset.data[:sample_size]):
    codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges']) 
    
    codes = codes.flatten().unsqueeze(0)
    codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
 
    coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    orgs = item['vertices'][item['faces']].unsqueeze(0)

    mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
    total_mse += mse 

    if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
 
    if len(random_samples) <= 30:
        random_samples.append(coords)
        random_samples_pred.append(orgs)
    else:
        all_random_samples.extend([random_samples_pred, random_samples])
        random_samples, random_samples_pred = [], []

print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')    
mesh_render.combind_mesh_with_rows(f'{working_dir}/mse_rows.obj', all_random_samples)

