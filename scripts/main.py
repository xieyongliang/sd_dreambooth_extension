import importlib
import time
from typing import List

import gradio as gr

from dreambooth.dataclasses.db_config import (
    from_file,
    save_keys
)
from dreambooth.diff_to_sd import compile_checkpoint
from dreambooth.secret import (
    get_secret,
    create_secret,
    clear_secret,
)
from dreambooth.shared import (
    status
)
from dreambooth.ui_functions import (
    load_model_params
)
from dreambooth.utils.image_utils import (
    get_scheduler_names,
)
from dreambooth.utils.model_utils import (
    get_db_models,
    get_sorted_lora_models,
    get_model_snapshots,
)
from dreambooth.utils.utils import (
    list_attention,
    list_precisions,
    printm,
    list_optimizer,
    list_schedulers,
)
from helpers.log_parser import LogParser
from modules import script_callbacks, shared
from modules.ui import gr_show, create_refresh_button
import json
import requests
import uuid

refresh_symbol = "\U0001f504"  # 🔄
delete_symbol = "\U0001F5D1"  # 🗑️
update_symbol = "\U0001F51D"  # 🠝
log_parser = LogParser()

def calc_time_left(progress, threshold, label, force_display):
    if progress == 0:
        return ""
    else:
        if status.time_start is None:
            time_since_start = 0
        else:
            time_since_start = time.time() - status.time_start
        eta = time_since_start / progress
        eta_relative = eta - time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 86400:
                days = eta_relative // 86400
                remainder = days * 86400
                eta_relative -= remainder
                return f"{label}{days}:{time.strftime('%H:%M:%S', time.gmtime(eta_relative))}"
            if eta_relative > 3600:
                return label + time.strftime("%H:%M:%S", time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime("%M:%S", time.gmtime(eta_relative))
            else:
                return label + time.strftime("%Ss", time.gmtime(eta_relative))
        else:
            return ""


def has_face_swap():
    script_class = None
    try:
        from modules.scripts import list_scripts

        scripts = list_scripts("scripts", ".py")
        for script_file in scripts:
            if script_file.filename == "batch_face_swap.py":
                path = script_file.path
                module_name = "batch_face_swap"
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                script_class = module.Script
                break
    except Exception as f:
        print(f"Can't check face swap: {f}")
    return script_class is not None


def check_progress_call():
    """
    Check the progress from share dreamstate and return appropriate UI elements.
    @return:
    active: Checkbox to physically hold an active state
    pspan: Progress bar span contents
    preview: Preview Image/Visibility
    gallery: Gallery Image/Visibility
    textinfo_result: Primary status
    sample_prompts: List = A list of prompts corresponding with gallery contents
    check_progress_initial: Hides the manual 'check progress' button
    """
    active_box = gr.update(value=status.active)
    if not status.active:
        return (
            active_box,
            "",
            gr.update(visible=False, value=None),
            gr.update(visible=True),
            gr_show(True),
            gr_show(True),
            gr_show(False),
        )

    progress = 0

    if status.job_count > 0:
        progress += status.job_no / status.job_count

    time_left = calc_time_left(progress, 1, " ETA: ", status.time_left_force_display)
    if time_left:
        status.time_left_force_display = True

    progress = min(progress, 1)
    progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress * 100)) + "%" + time_left if progress > 0.01 else ""}</div></div>"""
    status.set_current_image()
    image = status.current_image
    preview = None
    gallery = None

    if image is None:
        preview = gr.update(visible=False, value=None)
        gallery = gr.update(visible=True)
    else:
        if isinstance(image, List):
            if len(image) > 1:
                status.current_image = None
                preview = gr.update(visible=False, value=None)
                gallery = gr.update(visible=True, value=image)
            elif len(image) == 1:
                preview = gr.update(visible=True, value=image[0])
                gallery = gr.update(visible=True, value=None)
        else:
            preview = gr.update(visible=True, value=image)
            gallery = gr.update(visible=True, value=None)

    if status.textinfo is not None:
        textinfo_result = status.textinfo
    else:
        textinfo_result = ""

    if status.textinfo2 is not None:
        textinfo_result = f"{textinfo_result}<br>{status.textinfo2}"

    prompts = ""
    if len(status.sample_prompts) > 0:
        if len(status.sample_prompts) > 1:
            prompts = "<br>".join(status.sample_prompts)
        else:
            prompts = status.sample_prompts[0]

    pspan = f"<span id='db_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>"
    return (
        active_box,
        pspan,
        preview,
        gallery,
        textinfo_result,
        gr.update(value=prompts),
        gr_show(False),
    )


def check_progress_call_initial():
    status.begin()
    (
        active_box,
        pspan,
        preview,
        gallery,
        textinfo_result,
        prompts_result,
        pbutton_result,
    ) = check_progress_call()
    return (
        active_box,
        pspan,
        gr_show(False),
        gr.update(value=[]),
        textinfo_result,
        gr.update(value=[]),
        gr_show(False),
    )


def ui_gen_ckpt(model_name: str):
    if isinstance(model_name, List):
        model_name = model_name[0]
    if model_name == "" or model_name is None:
        return "Please select a model."
    config = from_file(model_name)
    printm("Config loaded")
    lora_path = config.lora_model_name
    print(f"Lora path: {lora_path}")
    res = compile_checkpoint(model_name, lora_path, True, True, config.snapshot)
    return res

training_instance_types = [
    'ml.p2.xlarge',
    'ml.p2.8xlarge',
    'ml.p2.16xlarge',
    'ml.p3.2xlarge',
    'ml.p3.8xlarge',
    'ml.p3.16xlarge',
    'ml.g4dn.xlarge',
    'ml.g4dn.2xlarge',
    'ml.g4dn.4xlarge',
    'ml.g4dn.8xlarge',
    'ml.g4dn.12xlarge',
    'ml.g4dn.16xlarge',
    'ml.g5.xlarge',
    'ml.g5.2xlarge',
    'ml.g5.4xlarge',
    'ml.g5.8xlarge',
    'ml.g5.12xlarge',
    'ml.g5.16xlarge',
    'ml.g5.24xlarge',
    'ml.g5.48xlarge',
    'ml.p4d.24xlarge'
]

def on_ui_tabs():
    with gr.Blocks() as dreambooth_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel", elem_id="ModelPanel"):
                with gr.Tab("Model"):
                    db_create_new_db_model = gr.Checkbox(label="Create new model", value=True)

                    with gr.Box(visible=False) as select_existing_model_box:
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Select existing model</b></p>")
                        with gr.Row():
                                db_model_name = gr.Dropdown(
                                    label="Model", choices=sorted(get_db_models())
                                )
                                create_refresh_button(
                                    db_model_name,
                                    get_db_models,
                                    lambda: {"choices": sorted(get_db_models())},
                                    "refresh_db_models",
                                )
                        with gr.Row():
                            db_snapshot = gr.Dropdown(
                                label="Snapshot to Resume",
                                choices=sorted(get_model_snapshots()),
                            )
                            create_refresh_button(
                                db_snapshot,
                                get_model_snapshots,
                                lambda: {"choices": sorted(get_model_snapshots())},
                                "refresh_db_snapshots",
                            )
                        with gr.Row(visible=False) as lora_model_row:
                            db_lora_model_name = gr.Dropdown(
                                label="Lora Model", choices=get_sorted_lora_models()
                            )
                            create_refresh_button(
                                db_lora_model_name,
                                get_sorted_lora_models,
                                lambda: {"choices": get_sorted_lora_models()},
                                "refresh_lora_models",
                            )
                        with gr.Row():
                            gr.HTML(value="Loaded Model:")
                            db_model_path = gr.HTML()
                        with gr.Row():
                            gr.HTML(value="Model Revision:")
                            db_revision = gr.HTML(elem_id="db_revision")
                        with gr.Row():
                            gr.HTML(value="Model Epoch:")
                            db_epochs = gr.HTML(elem_id="db_epochs")
                        with gr.Row():
                            gr.HTML(value="V2 Model:")
                            db_v2 = gr.HTML(elem_id="db_v2")
                        with gr.Row():
                            gr.HTML(value="Has EMA:")
                            db_has_ema = gr.HTML(elem_id="db_has_ema")
                        with gr.Row():
                            gr.HTML(value="Source Checkpoint:")
                            db_src = gr.HTML()

                    with gr.Box() as create_new_model_box:
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Create new model</b></p>")

                        db_new_model_name = gr.Textbox(label="Name")
                        with gr.Row():
                            db_create_from_hub = gr.Checkbox(
                                label="Create From Hub", value=False
                            )
                            db_512_model = gr.Checkbox(label="512x Model", value=True, interactive=True)
                        with gr.Row(visible=False) as hub_row:
                            db_new_model_url = gr.Textbox(
                                label="Model Path",
                                placeholder="runwayml/stable-diffusion-v1-5",
                            )
                            db_new_model_token = gr.Textbox(
                                label="HuggingFace Token", value=""
                            )
                        with gr.Row() as local_row:
                            db_new_model_src = gr.Dropdown(
                                label="Source Checkpoint",
                                choices=sorted(shared.list_sd_models()),
                                interactive=True
                            )
                            create_refresh_button(
                                db_new_model_src,
                                shared.refresh_sd_models,
                                lambda: {"choices": sorted(shared.list_sd_models())},
                                "refresh_sd_models",
                            )
                        db_new_model_extract_ema = gr.Checkbox(
                            label="Extract EMA Weights", value=False
                        )
                        db_train_unfrozen = gr.Checkbox(label="Unfreeze Model", value=False)

                    def swap_model_box_visibility(db_create_new_model):
                        return gr.update(visible=not db_create_new_model), gr.update(visible=db_create_new_model)

                    db_create_new_db_model.change(
                        fn=swap_model_box_visibility,
                        inputs=[db_create_new_db_model],
                        outputs=[select_existing_model_box, create_new_model_box],
                    )

                    with gr.Row():
                        db_train_wizard_person = gr.Checkbox(label="Optimization for training Person", value=False)
                        db_train_wizard_object = gr.Checkbox(label="Optimization for training Object/Style", value=False)
                        db_performance_wizard = gr.Checkbox(label="Optimzation for training performance (WIP)", value=False)

                with gr.Tab("Settings", elem_id="TabSettings"):
                    with gr.Column():
                        gr.HTML(value="SageMaker")
                        db_training_instance_type = gr.Dropdown(label='Instance type', value="ml.g4dn.xlarge", choices=training_instance_types, interactive=True)
                        db_training_instance_count = gr.Number(label='Instance count', value=1, precision=0, interactive=True)
                        db_concepts_s3uri = gr.Textbox(label='Concepts S3 URI')
                        db_models_s3uri = gr.Textbox(label='Models S3 URI')

                    with gr.Accordion(open=True, label="Basic"):
                        with gr.Column():
                            gr.HTML(value="General")
                            db_use_lora = gr.Checkbox(label="Use LORA", value=False)
                            db_use_lora_extended = gr.Checkbox(
                                label="Use Lora Extended",
                                value=False,
                                visible=False,
                            )
                            db_train_imagic_only = gr.Checkbox(label="Train Imagic Only", value=False)
                        with gr.Column():
                            gr.HTML(value="Intervals")
                            db_num_train_epochs = gr.Slider(
                                label="Training Steps Per Image (Epochs)",
                                value=100,
                                maximum=1000,
                                step=1,
                            )
                            db_epoch_pause_frequency = gr.Slider(
                                label="Pause After N Epochs",
                                value=0,
                                maximum=100,
                                step=1,
                            )
                            db_epoch_pause_time = gr.Slider(
                                label="Amount of time to pause between Epochs (s)",
                                value=0,
                                maximum=3600,
                                step=1,
                            )
                            db_save_embedding_every = gr.Slider(
                                label="Save Model Frequency (Epochs)",
                                value=25,
                                maximum=1000,
                                step=1,
                            )
                            db_save_preview_every = gr.Slider(
                                label="Save Preview(s) Frequency (Epochs)",
                                value=5,
                                maximum=1000,
                                step=1,
                            )

                        with gr.Column():
                            gr.HTML(value="Batching")
                            db_train_batch_size = gr.Slider(
                                label="Batch Size",
                                value=1,
                                minimum=1,
                                maximum=100,
                                step=1,
                            )
                            db_gradient_accumulation_steps = gr.Slider(
                                label="Gradient Accumulation Steps",
                                value=1,
                                minimum=1,
                                maximum=100,
                                step=1,
                            )
                            db_sample_batch_size = gr.Slider(
                                label="Class Batch Size",
                                minimum=1,
                                maximum=100,
                                value=1,
                                step=1,
                            )
                            db_gradient_set_to_none = gr.Checkbox(
                                label="Set Gradients to None When Zeroing", value=True
                            )
                            db_gradient_checkpointing = gr.Checkbox(
                                label="Gradient Checkpointing", value=True
                            )

                        with gr.Column():
                            gr.HTML(value="Learning Rate")
                            with gr.Row(visible=False) as lora_lr_row:
                                db_lora_learning_rate = gr.Number(
                                    label="Lora UNET Learning Rate", value=1e-4
                                )
                                db_lora_txt_learning_rate = gr.Number(
                                    label="Lora Text Encoder Learning Rate", value=5e-5
                                )
                            with gr.Row() as standard_lr_row:
                                db_learning_rate = gr.Number(
                                    label="Learning Rate", value=2e-6
                                )

                            db_lr_scheduler = gr.Dropdown(
                                label="Learning Rate Scheduler",
                                value="constant_with_warmup",
                                choices=list_schedulers(),
                            )
                            db_learning_rate_min = gr.Number(
                                label="Min Learning Rate", value=1e-6, visible=False
                            )
                            db_lr_cycles = gr.Number(
                                label="Number of Hard Resets",
                                value=1,
                                precision=0,
                                visible=False,
                            )
                            db_lr_factor = gr.Number(
                                label="Constant/Linear Starting Factor",
                                value=0.5,
                                precision=2,
                                visible=False,
                            )
                            db_lr_power = gr.Number(
                                label="Polynomial Power",
                                value=1.0,
                                precision=1,
                                visible=False,
                            )
                            db_lr_scale_pos = gr.Slider(
                                label="Scale Position",
                                value=0.5,
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                visible=False,
                            )
                            db_lr_warmup_steps = gr.Slider(
                                label="Learning Rate Warmup Steps",
                                value=0,
                                step=5,
                                maximum=10000,
                            )

                        with gr.Column():
                            gr.HTML(value="Image Processing")
                            db_resolution = gr.Slider(
                                label="Max Resolution",
                                step=64,
                                minimum=128,
                                value=512,
                                maximum=2048,
                                elem_id="max_res",
                            )
                            db_hflip = gr.Checkbox(
                                label="Apply Horizontal Flip", value=False
                            )

                        with gr.Column():
                            gr.HTML(value="Tuning")
                            db_use_ema = gr.Checkbox(
                                label="Use EMA", value=False, interactive=True
                            )
                            db_optimizer = gr.Dropdown(
                                label="Optimizer",
                                value="8bit AdamW",
                                choices=list_optimizer(),
                            )
                            db_mixed_precision = gr.Dropdown(
                                label="Mixed Precision",
                                value="no",
                                choices=list_precisions(),
                            )
                            db_attention = gr.Dropdown(
                                label="Memory Attention",
                                value="default",
                                choices=list_attention(),
                            )
                            db_cache_latents = gr.Checkbox(
                                label="Cache Latents", value=True
                            )
                            db_train_unet = gr.Checkbox(
                                label="Train UNET", value=True
                            )
                            db_stop_text_encoder = gr.Slider(
                                label="Step Ratio of Text Encoder Training",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0,
                                visible=True,
                            )
                            db_offset_noise = gr.Slider(
                                label="Offset Noise",
                                minimum=-1,
                                maximum=1,
                                step=0.01,
                                value=0,
                            )
                            db_freeze_clip_normalization = gr.Checkbox(
                                label="Freeze CLIP Normalization Layers",
                                visible=True,
                                value=False,
                            )
                            db_clip_skip = gr.Slider(
                                label="Clip Skip",
                                value=1,
                                minimum=1,
                                maximum=12,
                                step=1,
                            )
                            db_adamw_weight_decay = gr.Slider(
                                label="Weight Decay",
                                minimum=0,
                                maximum=1,
                                step=1e-7,
                                value=1e-2,
                                visible=True,
                            )
                            db_pad_tokens = gr.Checkbox(
                                label="Pad Tokens", value=True
                            )
                            db_strict_tokens = gr.Checkbox(
                                label="Strict Tokens", value=False
                            )
                            db_shuffle_tags = gr.Checkbox(
                                label="Shuffle Tags", value=True
                            )
                            db_max_token_length = gr.Slider(
                                label="Max Token Length",
                                minimum=75,
                                maximum=300,
                                step=75,
                            )
                        with gr.Column():
                            gr.HTML(value="Prior Loss")
                            db_prior_loss_scale = gr.Checkbox(
                                label="Scale Prior Loss", value=False
                            )
                            db_prior_loss_weight = gr.Slider(
                                label="Prior Loss Weight",
                                minimum=0.01,
                                maximum=1,
                                step=0.01,
                                value=0.75,
                            )
                            db_prior_loss_target = gr.Number(
                                label="Prior Loss Target",
                                value=100,
                                visible=False,
                            )
                            db_prior_loss_weight_min = gr.Slider(
                                label="Minimum Prior Loss Weight",
                                minimum=0.01,
                                maximum=1,
                                step=0.01,
                                value=0.1,
                                visible=False,
                            )

                    with gr.Accordion(open=False, label="Advanced"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Column():
                                    gr.HTML(value="Sanity Samples")
                                    db_sanity_prompt = gr.Textbox(
                                        label="Sanity Sample Prompt",
                                        placeholder="A generic prompt used to generate a sample image "
                                                    "to verify model fidelity.",
                                    )
                                    db_sanity_negative_prompt = gr.Textbox(
                                        label="Sanity Sample Negative Prompt",
                                        placeholder="A negative prompt for the generic sample image.",
                                    )
                                    db_sanity_seed = gr.Number(
                                        label="Sanity Sample Seed", value=420420
                                    )

                                with gr.Column():
                                    gr.HTML(value="Miscellaneous")
                                    db_pretrained_vae_name_or_path = gr.Textbox(
                                        label="Pretrained VAE Name or Path",
                                        placeholder="Leave blank to use base model VAE.",
                                        value="",
                                    )
                                    db_use_concepts = gr.Checkbox(
                                        label="Use Concepts List", value=False
                                    )
                                    db_concepts_path = gr.Textbox(
                                        label="Concepts List",
                                        placeholder="Path to JSON file with concepts to train.",
                                    )
                                    with gr.Row():
                                        db_secret = gr.Textbox(
                                            label="API Key", value=get_secret, interactive=False
                                        )
                                        db_refresh_button = gr.Button(
                                            value=refresh_symbol, elem_id="refresh_secret"
                                        )
                                        db_clear_secret = gr.Button(
                                            value=delete_symbol, elem_id="clear_secret"
                                        )

                                    db_class_gen_method = gr.Dropdown(
                                        label="Image Generation Library",
                                        value="Native Diffusers",
                                        choices=[
                                            "A1111 txt2img (Euler a)",
                                            "Native Diffusers",
                                        ]
                                    )
                                    db_scheduler = gr.Dropdown(
                                        label="Image Generation Scheduler",
                                        value="DEISMultistep",
                                        choices=get_scheduler_names(),
                                    )

                                    db_deterministic = gr.Checkbox(label="Deterministic")
                                    db_ema_predict = gr.Checkbox(label="Use EMA for prediction")
                                    db_split_loss = gr.Checkbox(
                                        label="Calculate Split Loss", value=True
                                    )
                                    db_tf32_enable = gr.Checkbox(
                                        label="Use TensorFloat 32", value=False
                                    )

                                    db_noise_scheduler = gr.Dropdown(
                                        label="Noise scheduler",
                                        value="DDPM",
                                        choices=[
                                            "DDPM",
                                            "DEIS",
                                            "UniPC"
                                        ]
                                    )


                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML(value="")
                with gr.Tab("Concepts", elem_id="TabConcepts") as concept_tab:
                    with gr.Column(variant="panel"):
                        with gr.Tab("Concept 1"):
                            (
                                c1_instance_data_dir,
                                c1_class_data_dir,
                                c1_instance_prompt,
                                c1_class_prompt,
                                c1_save_sample_prompt,
                                c1_save_sample_template,
                                c1_instance_token,
                                c1_class_token,
                                c1_num_class_images_per,
                                c1_class_negative_prompt,
                                c1_class_guidance_scale,
                                c1_class_infer_steps,
                                c1_save_sample_negative_prompt,
                                c1_n_save_sample,
                                c1_sample_seed,
                                c1_save_guidance_scale,
                                c1_save_infer_steps,
                            ) = build_concept_panel(1)

                        with gr.Tab("Concept 2"):
                            (
                                c2_instance_data_dir,
                                c2_class_data_dir,
                                c2_instance_prompt,
                                c2_class_prompt,
                                c2_save_sample_prompt,
                                c2_save_sample_template,
                                c2_instance_token,
                                c2_class_token,
                                c2_num_class_images_per,
                                c2_class_negative_prompt,
                                c2_class_guidance_scale,
                                c2_class_infer_steps,
                                c2_save_sample_negative_prompt,
                                c2_n_save_sample,
                                c2_sample_seed,
                                c2_save_guidance_scale,
                                c2_save_infer_steps,
                            ) = build_concept_panel(2)

                        with gr.Tab("Concept 3"):
                            (
                                c3_instance_data_dir,
                                c3_class_data_dir,
                                c3_instance_prompt,
                                c3_class_prompt,
                                c3_save_sample_prompt,
                                c3_save_sample_template,
                                c3_instance_token,
                                c3_class_token,
                                c3_num_class_images_per,
                                c3_class_negative_prompt,
                                c3_class_guidance_scale,
                                c3_class_infer_steps,
                                c3_save_sample_negative_prompt,
                                c3_n_save_sample,
                                c3_sample_seed,
                                c3_save_guidance_scale,
                                c3_save_infer_steps,
                            ) = build_concept_panel(3)

                        with gr.Tab("Concept 4"):
                            (
                                c4_instance_data_dir,
                                c4_class_data_dir,
                                c4_instance_prompt,
                                c4_class_prompt,
                                c4_save_sample_prompt,
                                c4_save_sample_template,
                                c4_instance_token,
                                c4_class_token,
                                c4_num_class_images_per,
                                c4_class_negative_prompt,
                                c4_class_guidance_scale,
                                c4_class_infer_steps,
                                c4_save_sample_negative_prompt,
                                c4_n_save_sample,
                                c4_sample_seed,
                                c4_save_guidance_scale,
                                c4_save_infer_steps,
                            ) = build_concept_panel(4)

                with gr.Tab("Saving"):
                    with gr.Column():
                        gr.HTML("General")
                        db_custom_model_name = gr.Textbox(
                            label="Custom Model Name",
                            value="",
                            placeholder="Enter a model name for saving checkpoints and lora models.",
                        )
                        db_save_safetensors = gr.Checkbox(
                            label="Save in .safetensors format",
                            value=True,
                            visible=False,
                        )
                        db_save_ema = gr.Checkbox(
                            label="Save EMA Weights to Generated Models", value=True
                        )
                        db_infer_ema = gr.Checkbox(
                            label="Use EMA Weights for Inference", value=False
                        )
                    with gr.Column():
                        gr.HTML("Checkpoints")
                        db_half_model = gr.Checkbox(label="Half Model", value=False)
                        db_use_subdir = gr.Checkbox(
                            label="Save Checkpoint to Subdirectory", value=True
                        )
                        db_save_ckpt_during = gr.Checkbox(
                            label="Generate a .ckpt file when saving during training."
                        )
                        db_save_ckpt_after = gr.Checkbox(
                            label="Generate a .ckpt file when training completes.",
                            value=True,
                        )
                        db_save_ckpt_cancel = gr.Checkbox(
                            label="Generate a .ckpt file when training is canceled."
                        )
                    with gr.Column(visible=False) as lora_save_col:
                        gr.HTML("Lora")
                        db_lora_unet_rank = gr.Slider(
                            label="Lora UNET Rank",
                            value=4,
                            minimum=2,
                            maximum=128,
                            step=2,
                        )
                        db_lora_txt_rank = gr.Slider(
                            label="Lora Text Encoder Rank",
                            value=4,
                            minimum=2,
                            maximum=768,
                            step=2,
                        )
                        db_lora_weight = gr.Slider(
                            label="Lora Weight",
                            value=1,
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                        )
                        db_lora_txt_weight = gr.Slider(
                            label="Lora Text Weight",
                            value=1,
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                        )
                        db_save_lora_during = gr.Checkbox(
                            label="Generate lora weights when saving during training."
                        )
                        db_save_lora_after = gr.Checkbox(
                            label="Generate lora weights when training completes.",
                            value=True,
                        )
                        db_save_lora_cancel = gr.Checkbox(
                            label="Generate lora weights when training is canceled."
                        )
                        db_save_lora_for_extra_net = gr.Checkbox(
                            label="Generate lora weights for extra networks."
                        )
                    with gr.Column():
                        gr.HTML("Diffusion Weights (training snapshots)")
                        db_save_state_during = gr.Checkbox(
                            label="Save separate diffusers snapshots when saving during training."
                        )
                        db_save_state_after = gr.Checkbox(
                            label="Save separate diffusers snapshots when training completes."
                        )
                        db_save_state_cancel = gr.Checkbox(
                            label="Save separate diffusers snapshots when training is canceled."
                        )

        with gr.Row():
            with gr.Column(scale=3):
                db_status = gr.Label(label='Output')
                training_job = gr.Markdown('Job detail')

            with gr.Column():
                shared.create_train_dreambooth_component = db_train_model = gr.Button(value="Train Dreambooth", variant='primary')

                db_refresh_button.click(
                    fn=create_secret, inputs=[], outputs=[db_secret]
                )

                def update_stop_tenc(train_unet):
                    # If train unet enabled, read "hidden" value from stop_tenc and restore
                    if train_unet:
                        return gr.update(interactive=True)
                    else:
                        return gr.update(interactive=False)

                db_train_unet.change(
                    fn=update_stop_tenc,
                    inputs=[db_train_unet],
                    outputs=[db_stop_text_encoder],
                )

                db_clear_secret.click(fn=clear_secret, inputs=[], outputs=[db_secret])

        def toggle_new_rows(create_from):
            return gr.update(visible=create_from), gr.update(visible=not create_from)

        def toggle_loss_items(scale):
            return gr.update(visible=scale), gr.update(visible=scale)

        db_create_from_hub.change(
            fn=toggle_new_rows,
            inputs=[db_create_from_hub],
            outputs=[hub_row, local_row],
        )

        db_prior_loss_scale.change(
            fn=toggle_loss_items,
            inputs=[db_prior_loss_scale],
            outputs=[db_prior_loss_weight_min, db_prior_loss_target],
        )

        def disable_lora(x):
            use_ema = gr.update(interactive=not x)
            use_lora_extended = gr.update(visible=x)
            lora_save = gr.update(visible=x)
            lora_lr = gr.update(visible=x)
            standard_lr = gr.update(visible=not x)
            lora_model = gr.update(visible=x)
            return (
                use_ema,
                use_lora_extended,
                lora_save,
                lora_lr,
                standard_lr,
                lora_model,
            )

        def lr_scheduler_changed(sched):
            show_scale_pos = gr.update(visible=False)
            show_min_lr = gr.update(visible=False)
            show_lr_factor = gr.update(visible=False)
            show_lr_warmup = gr.update(visible=False)
            show_lr_power = gr.update(visible=sched == "polynomial")
            show_lr_cycles = gr.update(visible=sched == "cosine_with_restarts")
            scale_scheds = [
                "constant",
                "linear",
                "cosine_annealing",
                "cosine_annealing_with_restarts",
            ]
            if sched in scale_scheds:
                show_scale_pos = gr.update(visible=True)
            else:
                show_lr_warmup = gr.update(visible=True)
            if sched in ["cosine_annealing", "cosine_annealing_with_restarts"]:
                show_min_lr = gr.update(visible=True)
            if sched in ["linear", "constant"]:
                show_lr_factor = gr.update(visible=True)
            return (
                show_lr_power,
                show_lr_cycles,
                show_scale_pos,
                show_lr_factor,
                show_min_lr,
                show_lr_warmup,
            )

        def class_gen_method_changed(method):
            show_scheduler = method == "Native Diffusers"
            scheduler = gr.update(visible=show_scheduler)
            return scheduler

        db_use_lora.change(
            fn=disable_lora,
            inputs=[db_use_lora],
            outputs=[
                db_use_ema,
                db_use_lora_extended,
                lora_save_col,
                lora_lr_row,
                standard_lr_row,
                lora_model_row,
            ],
        )

        db_lr_scheduler.change(
            fn=lr_scheduler_changed,
            inputs=[db_lr_scheduler],
            outputs=[
                db_lr_power,
                db_lr_cycles,
                db_lr_scale_pos,
                db_lr_factor,
                db_learning_rate_min,
                db_lr_warmup_steps,
            ],
        )

        db_class_gen_method.change(
            fn=class_gen_method_changed,
            inputs=[db_class_gen_method],
            outputs=[db_scheduler],
        )

        db_model_name.change(
            _js="clear_loaded",
            fn=load_model_params,
            inputs=[db_model_name],
            outputs=[
                db_model_path,
                db_revision,
                db_epochs,
                db_v2,
                db_has_ema,
                db_src,
                db_snapshot,
                db_lora_model_name,
                db_status,
            ],
        )

        db_use_concepts.change(
            fn=lambda x: {concept_tab: gr_show(x is True)},
            inputs=[db_use_concepts],
            outputs=[concept_tab],
        )

        def sagemaker_train_dreambooth(
            db_create_new_db_model,
            db_new_model_name,
            db_new_model_src,
            db_create_from_hub,
            db_new_model_url,
            db_new_model_token,
            db_new_model_extract_ema,
            db_class_gen_method,
            db_512_model,
            db_model_name,
            db_attention,
            db_cache_latents,
            db_clip_skip,
            db_concepts_path,
            db_custom_model_name,
            db_noise_scheduler,
            db_deterministic,
            db_ema_predict,
            db_epochs,
            db_epoch_pause_frequency,
            db_epoch_pause_time,
            db_freeze_clip_normalization,
            db_gradient_accumulation_steps,
            db_gradient_checkpointing,
            db_gradient_set_to_none,
            db_half_model,
            db_hflip,
            db_infer_ema,
            db_learning_rate,
            db_learning_rate_min,
            db_lora_learning_rate,
            db_lora_model_name,
            db_lora_unet_rank,
            db_lora_txt_rank,
            db_lora_txt_learning_rate,
            db_lora_txt_weight,
            db_lora_weight,
            db_lr_cycles,
            db_lr_factor,
            db_lr_power,
            db_lr_scale_pos,
            db_lr_scheduler,
            db_lr_warmup_steps,
            db_max_token_length,
            db_mixed_precision,
            db_adamw_weight_decay,
            db_model_path,
            db_num_train_epochs,
            db_offset_noise,
            db_optimizer,
            db_pad_tokens,
            db_pretrained_vae_name_or_path,
            db_prior_loss_scale,
            db_prior_loss_target,
            db_prior_loss_weight,
            db_prior_loss_weight_min,
            db_resolution,
            db_revision,
            db_sample_batch_size,
            db_sanity_prompt,
            db_sanity_seed,
            db_save_ckpt_after,
            db_save_ckpt_cancel,
            db_save_ckpt_during,
            db_save_embedding_every,
            db_save_ema,
            db_save_lora_after,
            db_save_lora_cancel,
            db_save_lora_during,
            db_save_lora_for_extra_net,
            db_save_preview_every,
            db_save_safetensors,
            db_save_state_after,
            db_save_state_cancel,
            db_save_state_during,
            db_scheduler,
            db_split_loss,
            db_strict_tokens,
            db_shuffle_tags,
            db_snapshot,
            db_src,
            db_tf32_enable,
            db_train_batch_size,
            db_train_imagic_only,
            db_train_unet,
            db_stop_text_encoder,
            db_use_concepts,
            db_train_unfrozen,
            db_use_ema,
            db_use_lora,
            db_use_lora_extended,
            db_use_subdir,
            c1_class_data_dir,
            c1_class_guidance_scale,
            c1_class_infer_steps,
            c1_class_negative_prompt,
            c1_class_prompt,
            c1_class_token,
            c1_instance_data_dir,
            c1_instance_prompt,
            c1_instance_token,
            c1_n_save_sample,
            c1_num_class_images_per,
            c1_sample_seed,
            c1_save_guidance_scale,
            c1_save_infer_steps,
            c1_save_sample_negative_prompt,
            c1_save_sample_prompt,
            c1_save_sample_template,
            c2_class_data_dir,
            c2_class_guidance_scale,
            c2_class_infer_steps,
            c2_class_negative_prompt,
            c2_class_prompt,
            c2_class_token,
            c2_instance_data_dir,
            c2_instance_prompt,
            c2_instance_token,
            c2_n_save_sample,
            c2_num_class_images_per,
            c2_sample_seed,
            c2_save_guidance_scale,
            c2_save_infer_steps,
            c2_save_sample_negative_prompt,
            c2_save_sample_prompt,
            c2_save_sample_template,
            c3_class_data_dir,
            c3_class_guidance_scale,
            c3_class_infer_steps,
            c3_class_negative_prompt,
            c3_class_prompt,
            c3_class_token,
            c3_instance_data_dir,
            c3_instance_prompt,
            c3_instance_token,
            c3_n_save_sample,
            c3_num_class_images_per,
            c3_sample_seed,
            c3_save_guidance_scale,
            c3_save_infer_steps,
            c3_save_sample_negative_prompt,
            c3_save_sample_prompt,
            c3_save_sample_template,
            c4_class_data_dir,
            c4_class_guidance_scale,
            c4_class_infer_steps,
            c4_class_negative_prompt,
            c4_class_prompt,
            c4_class_token,
            c4_instance_data_dir,
            c4_instance_prompt,
            c4_instance_token,
            c4_n_save_sample,
            c4_num_class_images_per,
            c4_sample_seed,
            c4_save_guidance_scale,
            c4_save_infer_steps,
            c4_save_sample_negative_prompt,
            c4_save_sample_prompt,
            c4_save_sample_template,
            db_train_wizard_person,
            db_train_wizard_object,
            db_performance_wizard,
            db_training_instance_type,
            db_training_instance_count,
            db_concepts_s3uri,
            db_models_s3uri,
            request : gr.Request
        ):
            username = shared.get_webui_username(request)

            params_to_save = [
                db_model_name,
                db_attention,
                db_cache_latents,
                db_clip_skip,
                db_concepts_path,
                db_custom_model_name,
                db_noise_scheduler,
                db_deterministic,
                db_ema_predict,
                db_epochs,
                db_epoch_pause_frequency,
                db_epoch_pause_time,
                db_freeze_clip_normalization,
                db_gradient_accumulation_steps,
                db_gradient_checkpointing,
                db_gradient_set_to_none,
                db_half_model,
                db_hflip,
                db_infer_ema,
                db_learning_rate,
                db_learning_rate_min,
                db_lora_learning_rate,
                db_lora_model_name,
                db_lora_unet_rank,
                db_lora_txt_rank,
                db_lora_txt_learning_rate,
                db_lora_txt_weight,
                db_lora_weight,
                db_lr_cycles,
                db_lr_factor,
                db_lr_power,
                db_lr_scale_pos,
                db_lr_scheduler,
                db_lr_warmup_steps,
                db_max_token_length,
                db_mixed_precision,
                db_adamw_weight_decay,
                db_model_path,
                db_num_train_epochs,
                db_offset_noise,
                db_optimizer,
                db_pad_tokens,
                db_pretrained_vae_name_or_path,
                db_prior_loss_scale,
                db_prior_loss_target,
                db_prior_loss_weight,
                db_prior_loss_weight_min,
                db_resolution,
                db_revision,
                db_sample_batch_size,
                db_sanity_prompt,
                db_sanity_seed,
                db_save_ckpt_after,
                db_save_ckpt_cancel,
                db_save_ckpt_during,
                db_save_embedding_every,
                db_save_ema,
                db_save_lora_after,
                db_save_lora_cancel,
                db_save_lora_during,
                db_save_lora_for_extra_net,
                db_save_preview_every,
                db_save_safetensors,
                db_save_state_after,
                db_save_state_cancel,
                db_save_state_during,
                db_scheduler,
                db_split_loss,
                db_strict_tokens,
                db_shuffle_tags,
                db_snapshot,
                db_src,
                db_tf32_enable,
                db_train_batch_size,
                db_train_imagic_only,
                db_train_unet,
                db_stop_text_encoder,
                db_use_concepts,
                db_train_unfrozen,
                db_use_ema,
                db_use_lora,
                db_use_lora_extended,
                db_use_subdir,
                c1_class_data_dir,
                c1_class_guidance_scale,
                c1_class_infer_steps,
                c1_class_negative_prompt,
                c1_class_prompt,
                c1_class_token,
                c1_instance_data_dir,
                c1_instance_prompt,
                c1_instance_token,
                c1_n_save_sample,
                c1_num_class_images_per,
                c1_sample_seed,
                c1_save_guidance_scale,
                c1_save_infer_steps,
                c1_save_sample_negative_prompt,
                c1_save_sample_prompt,
                c1_save_sample_template,
                c2_class_data_dir,
                c2_class_guidance_scale,
                c2_class_infer_steps,
                c2_class_negative_prompt,
                c2_class_prompt,
                c2_class_token,
                c2_instance_data_dir,
                c2_instance_prompt,
                c2_instance_token,
                c2_n_save_sample,
                c2_num_class_images_per,
                c2_sample_seed,
                c2_save_guidance_scale,
                c2_save_infer_steps,
                c2_save_sample_negative_prompt,
                c2_save_sample_prompt,
                c2_save_sample_template,
                c3_class_data_dir,
                c3_class_guidance_scale,
                c3_class_infer_steps,
                c3_class_negative_prompt,
                c3_class_prompt,
                c3_class_token,
                c3_instance_data_dir,
                c3_instance_prompt,
                c3_instance_token,
                c3_n_save_sample,
                c3_num_class_images_per,
                c3_sample_seed,
                c3_save_guidance_scale,
                c3_save_infer_steps,
                c3_save_sample_negative_prompt,
                c3_save_sample_prompt,
                c3_save_sample_template,
                c4_class_data_dir,
                c4_class_guidance_scale,
                c4_class_infer_steps,
                c4_class_negative_prompt,
                c4_class_prompt,
                c4_class_token,
                c4_instance_data_dir,
                c4_instance_prompt,
                c4_instance_token,
                c4_n_save_sample,
                c4_num_class_images_per,
                c4_sample_seed,
                c4_save_guidance_scale,
                c4_save_infer_steps,
                c4_save_sample_negative_prompt,
                c4_save_sample_prompt,
                c4_save_sample_template
            ]

            dreambooth_config_id = str(uuid.uuid4())
            params = {'dreambooth_config_id': dreambooth_config_id}
            params_dict = dict(zip(save_keys, params_to_save))
            response = requests.post(url=f'{shared.api_endpoint}/sd/models', json=params_dict, params=params)

            if response.status_code != 200:
                return {
                    db_status: gr.update(value=response.text)
                }

            train_args = {
                'train_dreambooth_settings': {
                    'db_create_new_db_model': db_create_new_db_model,
                    'db_new_model_name': db_new_model_name,
                    'db_new_model_src': db_new_model_src,
                    'db_create_from_hub': db_create_from_hub,
                    'db_new_model_url': db_new_model_url,
                    'db_new_model_token': db_new_model_token,
                    'db_new_model_extract_ema': db_new_model_extract_ema,
                    'db_train_unfrozen': db_train_unfrozen,
                    'db_512_model': db_512_model,
                    'db_model_name': db_model_name,
                    'db_train_wizard_person': db_train_wizard_person,
                    'db_train_wizard_object': db_train_wizard_object,
                    'db_performance_wizard': db_performance_wizard,
                    'db_lora_model_name': db_lora_model_name,
                    'db_class_gen_method': db_class_gen_method
                }
            }

            hyperparameters = {
                'train-args': json.dumps(json.dumps(train_args)),
                'train-task': 'dreambooth',
                'username': username,
                'api-endpoint': shared.api_endpoint,
                'dreambooth-config-id': dreambooth_config_id
            }

            inputs = {
                'concepts': db_concepts_s3uri,
                'models': db_models_s3uri
            }

            data = {
                'training_job_name': '',
                'model_algorithm': 'stable-diffusion-webui',
                'model_hyperparameters': hyperparameters,
                'industrial_model': shared.industrial_model,
                'instance_type': db_training_instance_type,
                'instance_count': db_training_instance_count,
                'inputs': inputs
            }

            response = requests.post(url=f'{shared.api_endpoint}/train', json=data)
            if response.status_code == 200:
                training_job_url = response.text.replace('\"','')
                return {
                    db_status: gr.update(value=f'Submit training job sucessful'),
                    training_job:gr.update(value=f'Job detail:[{training_job_url}]({training_job_url})')
                }
            else:
                return {
                    db_status: gr.update(value=response.text)
                }

        db_train_model.click(
            fn=sagemaker_train_dreambooth,
            inputs=[
                db_create_new_db_model,
                db_new_model_name,
                db_new_model_src,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema,
                db_class_gen_method,
                db_512_model,
                db_model_name,
                db_attention,
                db_cache_latents,
                db_clip_skip,
                db_concepts_path,
                db_custom_model_name,
                db_noise_scheduler,
                db_deterministic,
                db_ema_predict,
                db_epochs,
                db_epoch_pause_frequency,
                db_epoch_pause_time,
                db_freeze_clip_normalization,
                db_gradient_accumulation_steps,
                db_gradient_checkpointing,
                db_gradient_set_to_none,
                db_half_model,
                db_hflip,
                db_infer_ema,
                db_learning_rate,
                db_learning_rate_min,
                db_lora_learning_rate,
                db_lora_model_name,
                db_lora_unet_rank,
                db_lora_txt_rank,
                db_lora_txt_learning_rate,
                db_lora_txt_weight,
                db_lora_weight,
                db_lr_cycles,
                db_lr_factor,
                db_lr_power,
                db_lr_scale_pos,
                db_lr_scheduler,
                db_lr_warmup_steps,
                db_max_token_length,
                db_mixed_precision,
                db_adamw_weight_decay,
                db_model_path,
                db_num_train_epochs,
                db_offset_noise,
                db_optimizer,
                db_pad_tokens,
                db_pretrained_vae_name_or_path,
                db_prior_loss_scale,
                db_prior_loss_target,
                db_prior_loss_weight,
                db_prior_loss_weight_min,
                db_resolution,
                db_revision,
                db_sample_batch_size,
                db_sanity_prompt,
                db_sanity_seed,
                db_save_ckpt_after,
                db_save_ckpt_cancel,
                db_save_ckpt_during,
                db_save_embedding_every,
                db_save_ema,
                db_save_lora_after,
                db_save_lora_cancel,
                db_save_lora_during,
                db_save_lora_for_extra_net,
                db_save_preview_every,
                db_save_safetensors,
                db_save_state_after,
                db_save_state_cancel,
                db_save_state_during,
                db_scheduler,
                db_split_loss,
                db_strict_tokens,
                db_shuffle_tags,
                db_snapshot,
                db_src,
                db_tf32_enable,
                db_train_batch_size,
                db_train_imagic_only,
                db_train_unet,
                db_stop_text_encoder,
                db_use_concepts,
                db_train_unfrozen,
                db_use_ema,
                db_use_lora,
                db_use_lora_extended,
                db_use_subdir,
                c1_class_data_dir,
                c1_class_guidance_scale,
                c1_class_infer_steps,
                c1_class_negative_prompt,
                c1_class_prompt,
                c1_class_token,
                c1_instance_data_dir,
                c1_instance_prompt,
                c1_instance_token,
                c1_n_save_sample,
                c1_num_class_images_per,
                c1_sample_seed,
                c1_save_guidance_scale,
                c1_save_infer_steps,
                c1_save_sample_negative_prompt,
                c1_save_sample_prompt,
                c1_save_sample_template,
                c2_class_data_dir,
                c2_class_guidance_scale,
                c2_class_infer_steps,
                c2_class_negative_prompt,
                c2_class_prompt,
                c2_class_token,
                c2_instance_data_dir,
                c2_instance_prompt,
                c2_instance_token,
                c2_n_save_sample,
                c2_num_class_images_per,
                c2_sample_seed,
                c2_save_guidance_scale,
                c2_save_infer_steps,
                c2_save_sample_negative_prompt,
                c2_save_sample_prompt,
                c2_save_sample_template,
                c3_class_data_dir,
                c3_class_guidance_scale,
                c3_class_infer_steps,
                c3_class_negative_prompt,
                c3_class_prompt,
                c3_class_token,
                c3_instance_data_dir,
                c3_instance_prompt,
                c3_instance_token,
                c3_n_save_sample,
                c3_num_class_images_per,
                c3_sample_seed,
                c3_save_guidance_scale,
                c3_save_infer_steps,
                c3_save_sample_negative_prompt,
                c3_save_sample_prompt,
                c3_save_sample_template,
                c4_class_data_dir,
                c4_class_guidance_scale,
                c4_class_infer_steps,
                c4_class_negative_prompt,
                c4_class_prompt,
                c4_class_token,
                c4_instance_data_dir,
                c4_instance_prompt,
                c4_instance_token,
                c4_n_save_sample,
                c4_num_class_images_per,
                c4_sample_seed,
                c4_save_guidance_scale,
                c4_save_infer_steps,
                c4_save_sample_negative_prompt,
                c4_save_sample_prompt,
                c4_save_sample_template,
                db_train_wizard_person,
                db_train_wizard_object,
                db_performance_wizard,
                db_training_instance_type,
                db_training_instance_count,
                db_concepts_s3uri,
                db_models_s3uri
            ],
            outputs=[
                db_status,
                training_job
            ]
        )

    return ((dreambooth_interface, "Dreambooth", "dreambooth_interface"),)


def build_concept_panel(concept: int):
    with gr.Column():
        gr.HTML(value="Directories")
        instance_data_dir = gr.Textbox(
            label="Dataset Directory",
            placeholder="Path to directory with input images",
            elem_id=f"idd{concept}",
        )
        class_data_dir = gr.Textbox(
            label="Classification Dataset Directory",
            placeholder="(Optional) Path to directory with "
            "classification/regularization images",
            elem_id=f"cdd{concept}",
        )
    with gr.Column():
        gr.HTML(value="Filewords")
        instance_token = gr.Textbox(
            label="Instance Token",
            placeholder="When using [filewords], this is the subject to use when building prompts.",
        )
        class_token = gr.Textbox(
            label="Class Token",
            placeholder="When using [filewords], this is the class to use when building prompts.",
        )

    with gr.Column():
        gr.HTML(value="Training Prompts")
        instance_prompt = gr.Textbox(
            label="Instance Prompt",
            placeholder="Optionally use [filewords] to read image "
            "captions from files.",
        )
        class_prompt = gr.Textbox(
            label="Class Prompt",
            placeholder="Optionally use [filewords] to read image "
            "captions from files.",
        )
        class_negative_prompt = gr.Textbox(
            label="Classification Image Negative Prompt"
        )
    with gr.Column():
        gr.HTML(value="Sample Prompts")
        save_sample_prompt = gr.Textbox(
            label="Sample Image Prompt",
            placeholder="Leave blank to use instance prompt. "
                        "Optionally use [filewords] to base "
                        "sample captions on instance images.",
        )
        save_sample_negative_prompt = gr.Textbox(
            label="Sample Negative Prompt"
        )
        sample_template = gr.Textbox(
            label="Sample Prompt Template File",
            placeholder="Enter the path to a txt file containing sample prompts.",
        )

    with gr.Column():
        gr.HTML("Class Image Generation")
        num_class_images_per = gr.Slider(
            label="Class Images Per Instance Image", value=0, precision=0
        )
        class_guidance_scale = gr.Slider(
            label="Classification CFG Scale", value=7.5, maximum=12, minimum=1, step=0.1
        )
        class_infer_steps = gr.Slider(
            label="Classification Steps", value=40, minimum=10, maximum=200, step=1
        )

    with gr.Column():
        gr.HTML("Sample Image Generation")
        n_save_sample = gr.Slider(
            label="Number of Samples to Generate", value=1, maximum=100, step=1
        )
        sample_seed = gr.Number(label="Sample Seed", value=-1, precision=0)
        save_guidance_scale = gr.Slider(
            label="Sample CFG Scale", value=7.5, maximum=12, minimum=1, step=0.1
        )
        save_infer_steps = gr.Slider(
            label="Sample Steps", value=20, minimum=10, maximum=200, step=1
        )
    return [
        instance_data_dir,
        class_data_dir,
        instance_prompt,
        class_prompt,
        save_sample_prompt,
        sample_template,
        instance_token,
        class_token,
        num_class_images_per,
        class_negative_prompt,
        class_guidance_scale,
        class_infer_steps,
        save_sample_negative_prompt,
        n_save_sample,
        sample_seed,
        save_guidance_scale,
        save_infer_steps,
    ]


script_callbacks.on_ui_tabs(on_ui_tabs)
