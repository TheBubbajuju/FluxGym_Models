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
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible else None
        updates.append(gr.update(value=text_value, visible=visible))  # caption
    
    updates.append(gr.update(visible=True))  # start button
    updates.append(gr.update(visible=True))  # sample area
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size / width) * height)
        else:
            new_height = size
            new_width = int((size / height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        new_image_path = shutil.copy(image, destination_folder)
        resize_image(new_image_path, new_image_path, size)
        original_caption = inputs[index + 1]
        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path(os.path.join(destination_folder, caption_file_name), quote=False)
        print(f"image_path={new_image_path}, caption_path={caption_path}, caption={original_caption}")
        with open(caption_path, 'w') as file:
            file.write(original_caption)
    return destination_folder

def run_captioning(images, concept_sentence, *captions):
    print(f"run_captioning with concept: {concept_sentence}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text
        yield captions
    model.to("cpu")
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resolve_path(p, quote=True):
    norm_path = os.path.normpath(os.path.join(os.path.dirname(__file__), p))
    return f'"{norm_path}"' if quote else norm_path

def gen_sh(
    output_name, resolution, seed, workers, learning_rate, network_dim,
    max_train_epochs, save_every_n_epochs, timestep_sampling, guidance_scale,
    vram, sample_prompts, sample_every_n_steps
):
    line_break = "\\" if sys.platform != "win32" else "^"
    file_type = "sh" if sys.platform != "win32" else "bat"

    sample = ""
    if sample_prompts and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={resolve_path('sample_prompts.txt')} --sample_every_n
