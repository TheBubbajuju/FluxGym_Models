import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import subprocess

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download

MAX_IMAGES = 150

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error("Please upload at least 2 images to train your model (ideal: 4-30)")
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"Max {MAX_IMAGES} images allowed for training")
    
    updates.append(gr.update(visible=True))  # captioning_area
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)
        updates.append(gr.update(visible=visible))  # row
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))  # image
        
        corresponding_caption = False
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()
