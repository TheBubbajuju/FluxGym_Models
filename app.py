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

    # Build the sample part separately
    sample = ""
    if sample_prompts and sample_every_n_steps > 0:
        sample = (
            f"--sample_prompts={resolve_path('sample_prompts.txt')} "
            f"--sample_every_n_steps=\"{sample_every_n_steps}\" {line_break}"
        )

    # Paths
    pretrained_model_path = resolve_path("models/unet/flux1-dev-fp8.safetensors")
    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp8.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    output_dir = resolve_path("outputs")

    # Optimizer settings
    optimizer = (
        f"--optimizer_type adafactor {line_break}"
        f"--optimizer_args \"relative_step=False\" \"scale_parameter=False\" \"warmup_init=False\" {line_break}"
        f"--lr_scheduler constant_with_warmup {line_break}"
        f"--max_grad_norm 0.0 {line_break}"
    ) if vram in ["16G (T4x2)", "12G"] else f"--optimizer_type adamw8bit {line_break}"
    if vram == "12G":
        optimizer += f"--split_mode {line_break} --network_args \"train_blocks=single\" {line_break}"

    # Construct the full command
    sh = (
        f"accelerate launch --num_processes 2 --multi_gpu {line_break}"
        f"--mixed_precision bf16 {line_break}"
        f"--num_cpu_threads_per_process 1 {line_break}"
        f"sd-scripts/flux_train_network.py {line_break}"
        f"--pretrained_model_name_or_path {pretrained_model_path} {line_break}"
        f"--clip_l {clip_path} {line_break}"
        f"--t5xxl {t5_path} {line_break}"
        f"--ae {ae_path} {line_break}"
        f"--cache_latents_to_disk {line_break}"
        f"--save_model_as safetensors {line_break}"
        f"--sdpa --persistent_data_loader_workers {line_break}"
        f"--max_data_loader_n_workers {workers} {line_break}"
        f"--seed {seed} {line_break}"
        f"--gradient_checkpointing {line_break}"
        f"--gradient_accumulation_steps 4 {line_break}"
        f"--batch_size 1 {line_break}"
        f"--mixed_precision bf16 {line_break}"
        f"--save_precision bf16 {line_break}"
        f"--network_module networks.lora_flux {line_break}"
        f"--network_dim {network_dim} {line_break}"
        f"{optimizer}{sample}"
        f"--learning_rate {learning_rate} {line_break}"
        f"--cache_text_encoder_outputs {line_break}"
        f"--cache_text_encoder_outputs_to_disk {line_break}"
        f"--fp8_base {line_break}"
        f"--highvram {line_break}"
        f"--max_train_epochs {max_train_epochs} {line_break}"
        f"--save_every_n_epochs {save_every_n_epochs} {line_break}"
        f"--dataset_config {resolve_path('dataset.toml')} {line_break}"
        f"--output_dir {output_dir} {line_break}"
        f"--output_name {output_name} {line_break}"
        f"--timestep_sampling {timestep_sampling} {line_break}"
        f"--discrete_flow_shift 3.1582 {line_break}"
        f"--model_prediction_type raw {line_break}"
        f"--guidance_scale {guidance_scale} {line_break}"
        f"--loss_type l2 {line_break}"
    )
    return sh

def gen_toml(dataset_folder, resolution, class_tokens, num_repeats):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path(dataset_folder, quote=False)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs}, num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value=total_steps)
    except:
        return gr.update(value=0)

def get_samples():
    try:
        samples_path = resolve_path('outputs/sample', quote=False)
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(train_script, train_config, sample_prompts):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    file_type = "sh" if sys.platform != "win32" else "bat"
    sh_filename = f"train.{file_type}"
    with open(sh_filename, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")

    with open('dataset.toml', 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    with open('sample_prompts.txt', 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    command = f"bash {resolve_path('train.sh')}" if sys.platform != "win32" else resolve_path('train.bat', quote=False)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    gr.Info("Checking GPU memory usage before training")
    yield from runner.run_command(["nvidia-smi"], cwd=cwd)
    gr.Info("Started training")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")
    gr.Info("Training Complete. Check the outputs folder for the LoRA files.", duration=None)

def update(
    lora_name, resolution, seed, workers, class_tokens, learning_rate,
    network_dim, max_train_epochs, save_every_n_epochs, timestep_sampling,
    guidance_scale, vram, num_repeats, sample_prompts, sample_every_n_steps
):
    output_name = slugify(lora_name)
    dataset_folder = f"datasets/{output_name}"
    sh = gen_sh(
        output_name, resolution, seed, workers, learning_rate, network_dim,
        max_train_epochs, save_every_n_epochs, timestep_sampling, guidance_scale,
        vram, sample_prompts, sample_every_n_steps
    )
    toml = gen_toml(dataset_folder, resolution, class_tokens, num_repeats)
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

def loaded():
    print("launched")

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
h1 { font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px; }
h3 { margin-top: 0; }
.tabitem { border: 0px; }
.group_padding {}
nav { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
#container { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll");
    if (window.iidxx) { window.clearInterval(window.iidxx); }
    window.iidxx = window.setInterval(function() {
        let text = document.querySelector(".codemirror-wrapper .cm-line")?.innerText.trim();
        let img = document.querySelector("#logo");
        if (text?.length > 0) {
            autoscroll.classList.remove("hidden");
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON";
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate");
            } else {
                autoscroll.textContent = "Autoscroll OFF";
                img.classList.remove("rotate");
            }
        }
    }, 500);
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on");
    });
}
"""

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    output_components = []
    with gr.Row():
        gr.HTML("""<nav>
    <img id='logo' src='/file=icon.png' width='80' height='80'>
    <div class='flexible'></div>
    <button id='autoscroll' class='on hidden'></button>
</nav>""")
    with gr.Row(elem_id='container'):
        with gr.Column():
            gr.Markdown(
                """# Step 1. LoRA Info
<p style="margin-top:0">Configure your LoRA train settings. Running on Kaggle T4x2 (2x 16GB VRAM).</p>
""", elem_classes="group_padding")
            lora_name = gr.Textbox(
                label="The name of your LoRA",
                info="This has to be a unique name",
                placeholder="e.g., Persian Miniature Painting style, Cat Toy",
            )
            concept_sentence = gr.Textbox(
                label="Trigger word/sentence",
                info="Trigger word or sentence to be used",
                placeholder="e.g., p3rs0n or 'in the style of CNSTLL'",
                interactive=True,
            )
            vram = gr.Radio(["16G (T4x2)", "12G"], value="16G (T4x2)", label="VRAM", interactive=True)
            num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
            max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
            total_steps = gr.Number(0, interactive=False, label="Expected training steps")
            sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
            sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Number(label="Seed", value=42, interactive=True)
                workers = gr.Number(label="Workers", value=2, interactive=True)
                learning_rate = gr.Textbox(label="Learning Rate", value="8e-4", interactive=True)
                save_every_n_epochs = gr.Number(label="Save every N epochs", value=4, interactive=True)
                guidance_scale = gr.Number(label="Guidance Scale", value=1.0, interactive=True)
                timestep_sampling = gr.Textbox(label="Timestep Sampling", value="shift", interactive=True)
                network_dim = gr.Number(label="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                resolution = gr.Dropdown([256, 512, 768], value=512, label="Resize dataset images", info="512 recommended; use 256 if memory issues occur")
        with gr.Column():
            gr.Markdown(
                """# Step 2. Dataset
<p style="margin-top:0">Make sure captions include the trigger word.</p>
""", elem_classes="group_padding")
            with gr.Group():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="Upload your images",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
            with gr.Group(visible=False) as captioning_area:
                do_captioning = gr.Button("Add AI captions with Florence-2")
                output_components.append(captioning_area)
                caption_list = []
                for i in range(1, MAX_IMAGES + 1):
                    locals()[f"captioning_row_{i}"] = gr
