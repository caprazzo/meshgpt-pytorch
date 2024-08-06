#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install  -q git+https://github.com/MarcusLoppe/meshgpt-pytorch.git


# In[2]:


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

def get_mesh(file_path): 
    mesh = trimesh.load(file_path, force='mesh') 
    vertices = mesh.vertices.tolist()
    if ".off" in file_path:  # ModelNet dataset
       mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]] 
       rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
       mesh.apply_transform(rotation_matrix) 
        # Extract vertices and faces from the rotated mesh
       vertices = mesh.vertices.tolist()
            
    faces = mesh.faces.tolist()
    
    centered_vertices = vertices - np.mean(vertices, axis=0)  
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)     # Limit vertices to [-0.95, 0.95]
      
    min_y = np.min(vertices[:, 1]) 
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    
    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]   
 
    seen = OrderedDict()
    for point in vertices: 
      key = tuple(point)
      if key not in seen:
        seen[key] = point
        
    unique_vertices =  list(seen.values()) 
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
      
    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if vertex_tuple == sorted_vertex_tuple} 
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces] 
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]   
    return np.array(sorted_vertices), np.array(sorted_faces)
 
 

def augment_mesh(vertices, scale_factor):     
    jitter_factor=0.01 
    possible_values = np.arange(-jitter_factor, jitter_factor , 0.0005) 
    offsets = np.random.choice(possible_values, size=vertices.shape) 
    vertices = vertices + offsets   
    
    vertices = vertices * scale_factor 
    # To ensure that the mesh models are on the "ground"
    min_y = np.min(vertices[:, 1])  
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    return vertices


#load_shapenet("./shapenet", "./shapenet_csv_files", 10, 10)   
#Find the csv files with the labels in the ShapeNetCore.v1.zip, download at  https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive  
def load_shapenet(directory, per_category, variations ):
    obj_datas = []   
    chosen_models_count = {}    
    print(f"per_category: {per_category} variations {variations}")
    
    with open('shapenet_labels.json' , 'r') as f:
        id_info = json.load(f) 
    
    possible_values = np.arange(0.75, 1.0 , 0.005) 
    scale_factors = np.random.choice(possible_values, size=variations) 
    
    for category in os.listdir(directory): 
        category_path = os.path.join(directory, category)   
        if os.path.isdir(category_path) == False:
            continue 
        
        num_files_in_category = len(os.listdir(category_path))
        print(f"{category_path} got {num_files_in_category} files") 
        chosen_models_count[category] = 0  
        
        for filename in os.listdir(category_path):
            if filename.endswith((".obj", ".glb", ".off")):
                file_path = os.path.join(category_path, filename)
                
                if chosen_models_count[category] >= per_category:
                    break 
                if os.path.getsize(file_path) >  20 * 1024: # 20 kb limit = less then 400-600 faces
                    continue 
                if filename[:-4] not in id_info:
                    print("Unable to find id info for ", filename)
                    continue 
                vertices, faces = get_mesh(file_path) 
                if len(faces) > 800: 
                    continue
                
                chosen_models_count[category] += 1  
                textName = id_info[filename[:-4]]   
                
                face_edges =  derive_face_edges_from_faces(faces)  
                for scale_factor in scale_factors: 
                    aug_vertices = augment_mesh(vertices.copy(), scale_factor)   
                    obj_data = {"vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"), "faces":  torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"), "face_edges" : face_edges, "texts": textName }  
                    obj_datas.append(obj_data)
                    
    print("="*25)
    print("Chosen models count for each category:")
    for category, count in chosen_models_count.items():
        print(f"{category}: {count}") 
    total_chosen_models = sum(chosen_models_count.values())
    print(f"Total number of chosen models: {total_chosen_models}")
    return obj_datas

  
   
def load_filename(directory, variations):
    obj_datas = []    
    possible_values = np.arange(0.75, 1.0 , 0.005) 
    scale_factors = np.random.choice(possible_values, size=variations) 
    
    for filename in os.listdir(directory):
        if filename.endswith((".obj", ".glb", ".off")): 
            file_path = os.path.join(directory, filename) 
            vertices, faces = get_mesh(file_path)  
            
            faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
            face_edges =  derive_face_edges_from_faces(faces)  
            texts, ext = os.path.splitext(filename)     
            
            for scale_factor in scale_factors: 
                aug_vertices = augment_mesh(vertices.copy(), scale_factor)  
                obj_data = {"vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"), "faces":  faces, "face_edges" : face_edges, "texts": texts } 
                obj_datas.append(obj_data)
                     
    print(f"[create_mesh_dataset] Returning {len(obj_data)} meshes")
    return obj_datas


# In[3]:


# import gzip,json
# from tqdm import tqdm
# import pandas as pd

# # Instruction to download objverse meshes: https://github.com/MarcusLoppe/Objaverse-downloader/tree/main
# def load_objverse(directory, variations ):
#     obj_datas = []     
#     id_info = {}   
#     pali_captions = pd.read_csv('.\pali_captions.csv', sep=';') # https://github.com/google-deepmind/objaverse_annotations/blob/main/pali_captions.csv
#     pali_captions_dict = pali_captions.set_index("object_uid").to_dict()["top_aggregate_caption"]  
        
#     possible_values = np.arange(0.75, 1.0) 
#     scale_factors = np.random.choice(possible_values, size=variations) 
    
#     for folder in os.listdir(directory):  
#         full_folder_path = os.path.join(directory, folder)   
#         if os.path.isdir(full_folder_path) == False:
#             continue    
         
#         for filename in tqdm(os.listdir(full_folder_path)):  
#             if filename.endswith((".obj", ".glb", ".off")):
#                 file_path = os.path.join(full_folder_path, filename)
#                 kb = os.path.getsize(file_path)  / 1024 
#                 if kb < 1 or kb > 30:
#                     continue
                  
#                 if filename[:-4] not in pali_captions_dict: 
#                     continue   
#                 textName =  pali_captions_dict[filename[:-4]]
#                 try:    
#                     vertices, faces = get_mesh(file_path)   
#                 except Exception as e:
#                     continue
                
#                 if len(faces) > 250 or len(faces) < 50: 
#                     continue
                
#                 faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
#                 face_edges = derive_face_edges_from_faces(faces)   
#                 for scale_factor in scale_factors: 
#                     aug_vertices = augment_mesh(vertices.copy(), scale_factor)   
#                     obj_data = {"filename": filename, "vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"), "faces":  faces, "face_edges" : face_edges, "texts": textName }   
#                     obj_datas.append(obj_data)  
#     return obj_datas


# In[4]:


from pathlib import Path 
import gc     
import os
from meshgpt_pytorch import MeshDataset 
 
#project_name = "demo_mesh" 
project_name = "proj_large"

working_dir = f'{project_name}'

working_dir = Path(working_dir)
working_dir.mkdir(exist_ok = True, parents = True)
dataset_path = f"{working_dir}/objverse_shapenet_modelnet_max_250faces_186M_tokens.npz"
#dataset_path = f"{working_dir}/demo_mesh.npz"
 
#if not os.path.isfile(dataset_path):
#    data = load_filename("./demo_mesh",50)  
#    dataset = MeshDataset(data) 
#    dataset.generate_face_edges()  
#    dataset.save(dataset_path)
    
print(dataset_path)
dataset = MeshDataset.load(dataset_path) 
print(dataset.data[0].keys())


# #### Inspect imported meshes (optional)

# In[5]:


# from pathlib import Path
 
# folder = working_dir / f'renders' 
# obj_file_path = Path(folder)
# obj_file_path.mkdir(exist_ok = True, parents = True)
   
# all_vertices = []
# all_faces = []
# vertex_offset = 0
# translation_distance = 0.5  

# for r, item in enumerate(data): 
#     vertices_copy =  np.copy(item['vertices'])
#     vertices_copy += translation_distance * (r / 0.2 - 1) 
    
#     for vert in vertices_copy:
#         vertex = vert.to('cpu')
#         all_vertices.append(f"v {float(vertex[0])}  {float(vertex[1])}  {float(vertex[2])}\n") 
#     for face in item['faces']:
#         all_faces.append(f"f {face[0]+1+ vertex_offset} {face[1]+ 1+vertex_offset} {face[2]+ 1+vertex_offset}\n")  
#     vertex_offset = len(all_vertices)
 
# obj_file_content = "".join(all_vertices) + "".join(all_faces)
 
# obj_file_path = f'{folder}/3d_models_inspect.obj' 
# with open(obj_file_path, "w") as file:
#     file.write(obj_file_content)    
    


# ### Train!

# In[6]:


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
print(f"Total parameters: {total_params}")


# **Have at least 400-2000 items in the dataset, use this to multiply the dataset**  

# In[7]:


# dataset.data = [dict(d) for d in dataset.data]
# print(len(dataset.data))


# *Load previous saved model if you had to restart session*

# In[8]:


pkg = torch.load(str(f'proj_large/20240718-mesh-encoder-loss_0.157.pt')) 
autoencoder.load_state_dict(pkg['model'])
for param in autoencoder.parameters():
     param.requires_grad = True


# **Train to about 0.3 loss if you are using a small dataset**

# In[9]:


# batch_size=16 # The batch size should be max 64.
# grad_accum_every = 4
# # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
# learning_rate = 1e-4 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

# autoencoder.commit_loss_weight = 0.4 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
# autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
#                                              batch_size=batch_size,
#                                              grad_accum_every = grad_accum_every,
#                                              learning_rate = learning_rate,
#                                              checkpoint_every_epoch=1) 
# loss = autoencoder_trainer.train(480, stop_at_loss = 0.2, diplay_graph= True)    
# autoencoder_trainer.save(f'{working_dir}/mesh-encoder_{project_name}.pt')   


# In[ ]:





# ### Inspect how the autoencoder can encode and then provide the decoder with the codes to reconstruct the mesh

# In[10]:


# import torch
# import random
# from tqdm import tqdm 
# from meshgpt_pytorch import mesh_render 

# min_mse, max_mse = float('inf'), float('-inf')
# min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
# random_samples, random_samples_pred, all_random_samples = [], [], []
# total_mse, sample_size = 0.0, 200

# random.shuffle(dataset.data)

# for item in tqdm(dataset.data[:sample_size]):
#     codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges']) 
    
#     codes = codes.flatten().unsqueeze(0)
#     codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
 
#     coords, mask = autoencoder.decode_from_codes_to_faces(codes)
#     orgs = item['vertices'][item['faces']].unsqueeze(0)

#     mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
#     total_mse += mse 

#     if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
#     if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
 
#     if len(random_samples) <= 30:
#         random_samples.append(coords)
#         random_samples_pred.append(orgs)
#     else:
#         all_random_samples.extend([random_samples_pred, random_samples])
#         random_samples, random_samples_pred = [], []

# print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')    
# mesh_render.combind_mesh_with_rows(f'{working_dir}/mse_rows.obj', all_random_samples)


# ### Training & fine-tuning
# 
# **Pre-train:** Train the transformer on the full dataset with all the augmentations, the longer / more epochs will create a more robust model.<br/>
# 
# **Fine-tune:** Since it will take a long time to train on all the possible augmentations of the meshes, I recommend that you remove all the augmentations so you are left with x1 model per mesh.<br/>
# Below is the function **filter_dataset** that will return a single copy of each mesh.<br/>
# The function can also check for duplicate labels, this may speed up the fine-tuning process (not recommanded) however this most likely will remove it's ability for novel mesh generation.

# In[11]:


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


# In[12]:


# def filter_dataset(dataset, unique_labels = False):
#     unique_dicts = []
#     unique_tensors = set()
#     texts = set()
#     for d in dataset.data:
#         tensor = d["faces"]
#         tensor_tuple = tuple(tensor.cpu().numpy().flatten())
#         if unique_labels and d['texts'] in texts:
#             continue
#         if tensor_tuple not in unique_tensors:
#             unique_tensors.add(tensor_tuple)
#             unique_dicts.append(d)
#             texts.add(d['texts'])
#     return unique_dicts 
# dataset.data = filter_dataset(dataset, unique_labels = False)


# ## **Required!**, embed the text and run generate_codes to save 4-96 GB VRAM (dependant on dataset) ##
# 
# **If you don't;** <br>
# During each during each training step the autoencoder will generate the codes and the text encoder will embed the text.
# <br>
# After these fields are generate: **they will be deleted and next time it generates the code again:**<br>
# 
# This is due to the dataloaders nature, it writes this information to a temporary COPY of the dataset
# 

# In[13]:


labels = list(set(item["texts"] for item in dataset.data))
dataset.embed_texts(transformer, batch_size = 25)
dataset.generate_codes(autoencoder, batch_size = 50)
print(dataset.data[0].keys())


# *Load previous saved model if you had to restart session*

# In[14]:


# pkg = torch.load(str(f'checkpoints/mesh-transformer.ckpt.epoch_8_avg_loss_0.927.pt')) 
# transformer.load_state_dict(pkg['model'])
# del pkg
# import gc  
# torch.cuda.empty_cache()
# gc.collect()


# **Train to about 0.0001 loss (or less) if you are using a small dataset**

# In[ ]:


batch_size = 3 # Max 64
grad_accum_every = 8

# Set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  4 * 16 = 64
learning_rate = 1e-3 # Start training with the learning rate at 1e-2 then lower it to 1e-3 at stagnation or at 0.5 loss.

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,num_train_steps=100, dataset = dataset,
                                 grad_accum_every=grad_accum_every,
                                 learning_rate = learning_rate,
                                 batch_size=batch_size,
                                 checkpoint_every_epoch = 1,
                                 # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                                 # However it might cause nan after a while.
                                 # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7} 
                                 )


trainer.load('checkpoints/mesh-transformer.ckpt.epoch_8_avg_loss_0.927.pt')

loss = trainer.train(300, stop_at_loss = 0.5)  
#trainer.save(f'{working_dir}/mesh-transformer_{project_name}.pt')   



# In[ ]:


trainer.save(f'{working_dir}/2024-07-21_M0000-transformer.pt')   


# ## Generate and view mesh

# **Using only text**

# In[ ]:


from meshgpt_pytorch import mesh_render 
from pathlib import Path
 
folder = working_dir / 'renders'
obj_file_path = Path(folder)
obj_file_path.mkdir(exist_ok = True, parents = True)  
 
text_coords = [] 
for text in ('box', 'cube', 'ball', 'sphere'):
    print(f"Generating {text}") 
    text_coords.append(transformer.generate(texts = [text],  temperature = 0.0))   
    
mesh_render.save_rendering(f'{folder}/3d_models_all.obj', text_coords)


# In[ ]:





# **Text + prompt of tokens**

# **Prompt with 10% of codes/tokens**

# In[ ]:


from pathlib import Path 
from meshgpt_pytorch import mesh_render 
folder = working_dir / f'renders/text+codes'
obj_file_path = Path(folder)
obj_file_path.mkdir(exist_ok = True, parents = True)  

token_length_procent = 0.10 
codes = []
texts = []
for label in labels:
    for item in dataset.data: 
        if item['texts'] == label:
            tokens = autoencoder.tokenize(
                vertices = item['vertices'],
                faces = item['faces'],
                face_edges = item['face_edges']
            ) 
            num_tokens = int(tokens.shape[0] * token_length_procent)  
            texts.append(item['texts']) 
            codes.append(tokens.flatten()[:num_tokens].unsqueeze(0))  
            break
        
coords = []  
for text, prompt in zip(texts, codes): 
    print(f"Generating {text} with {prompt.shape[1]} tokens") 
    coords.append(transformer.generate(texts = [text],  prompt = prompt, temperature = 0) )    
      
mesh_render.save_rendering(f'{folder}/text+prompt_{token_length_procent*100}.obj', coords)


# **Prompt with 0% to 80% of tokens**

# In[ ]:


from pathlib import Path
from meshgpt_pytorch import mesh_render 
 
folder = working_dir / f'renders/text+codes_rows'
obj_file_path = Path(folder)
obj_file_path.mkdir(exist_ok = True, parents = True)   

mesh_rows = []
for token_length_procent in np.arange(0, 0.8, 0.1):
    codes = []
    texts = []
    for label in labels:
        for item in dataset.data: 
            if item['texts'] == label:
                tokens = autoencoder.tokenize(
                    vertices = item['vertices'],
                    faces = item['faces'],
                    face_edges = item['face_edges']
                ) 
                num_tokens = int(tokens.shape[0] * token_length_procent) 
                
                texts.append(item['texts']) 
                codes.append(tokens.flatten()[:num_tokens].unsqueeze(0))  
                break
            
    coords = []   
    for text, prompt in zip(texts, codes):  
        print(f"Generating {text} with {prompt.shape[1]} tokens") 
        coords.append(transformer.generate(texts = [text],  prompt = prompt, temperature = 0)) 
         
    mesh_rows.append(coords)  
    
mesh_render.save_rendering(f'{folder}/all.obj', mesh_rows)
 


# **Just some testing for text embedding similarity**

# In[ ]:


import numpy as np 
texts = list(labels)
vectors = [transformer.conditioner.text_models[0].embed_text([text], return_text_encodings = False).cpu().flatten() for text in texts]
 
max_label_length = max(len(text) for text in texts)
 
# Print the table header
print(f"{'Text':<{max_label_length}} |", end=" ")
for text in texts:
    print(f"{text:<{max_label_length}} |", end=" ")
print()

# Print the similarity matrix as a table with fixed-length columns
for i in range(len(texts)):
    print(f"{texts[i]:<{max_label_length}} |", end=" ")
    for j in range(len(texts)):
        # Encode the texts and calculate cosine similarity manually
        vector_i = vectors[i]
        vector_j = vectors[j]
        
        dot_product = torch.sum(vector_i * vector_j)
        norm_vector1 = torch.norm(vector_i)
        norm_vector2 = torch.norm(vector_j)
        similarity_score = dot_product / (norm_vector1 * norm_vector2)
        
        # Print with fixed-length columns
        print(f"{similarity_score.item():<{max_label_length}.4f} |", end=" ")
    print()

