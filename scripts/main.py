import gradio as gr

from dreambooth import conversion, dreambooth
from dreambooth.dreambooth import get_db_models
from modules import script_callbacks, sd_models, shared
from modules.ui import paste_symbol, setup_progressbar
from webui import wrap_gradio_gpu_call


def on_ui_tabs():
    with gr.Blocks() as dreambooth_interface:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        db_pretrained_model_name_or_path = gr.Dropdown(label='Model', choices=sorted(get_db_models()))
            with gr.Column(scale=1, elem_id="roll_col"):
                db_load_params = gr.Button(label='Load Training Params', value=paste_symbol)

            with gr.Column(scale=1):
                with gr.Row():
                    db_interrupt_training = gr.Button(value="Cancel")
                    db_train_embedding = gr.Button(value="Train", variant='primary')
            # cancel/load buttons
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                with gr.Tab("Create Model"):
                    db_new_model_name = gr.Textbox(label="Name")
                    src_checkpoint = gr.Dropdown(label='Source Checkpoint', choices=sorted(
                        sd_models.checkpoints_list.keys()))
                    # I just randomly chose ddim here because we use it everywhere else. Not sure which of these
                    # are ideal, or if it matters at all.
                    diff_type = gr.Dropdown(label='Scheduler', choices=["pndm", "ddim", "lms"], value="pndm")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            db_create_embedding = gr.Button(value="Create", variant='primary')
                with gr.Tab("Train Model"):
                    with gr.Accordion(open=True, label="Settings"):
                        db_concepts_list = gr.Textbox(label="Concepts List (Overrides instance/class settings below)",
                                                      placeholder="Path to JSON file with concepts to train.")
                        db_instance_prompt = gr.Textbox(label="Instance prompt", value="*")
                        db_class_prompt = gr.Textbox(label="Class prompt", value="*")
                        db_instance_data_dir = gr.Textbox(label='Dataset directory',
                                                          placeholder="Path to directory with input images")
                        db_class_data_dir = gr.Textbox(label='Classification dataset directory (optional).',
                                                       placeholder="Path to directory with classification images")
                        db_num_class_images = gr.Number(
                            label='Total number of classification images to use. Set to 0 to disable.', value=0,
                            precision=0)
                        db_max_train_steps = gr.Number(label='Training steps', value=1000, precision=0)
                        db_train_batch_size = gr.Number(label="Batch Size", precision=1, value=1)
                        db_sample_batch_size = gr.Number(label="Class Batch Size", precision=1, value=1)
                        db_learning_rate = gr.Number(label='Learning rate', value=5e-6)
                        db_resolution = gr.Number(label="Resolution", precision=1, value=512)
                        db_save_embedding_every = gr.Number(
                            label='Save a checkpoint every N steps, 0 to disable', value=500,
                            precision=0)
                        db_save_preview_every = gr.Number(
                            label='Generate a preview image every N steps, 0 to disable', value=500,
                            precision=0)
                        db_save_sample_prompt = gr.Textbox(label="Preview image prompt",
                                                           placeholder="Leave blank to use instance prompt.")
                        db_save_sample_negative_prompt = gr.Textbox(label="Preview image negative prompt",
                                                                    placeholder="Leave blank to use instance prompt.")
                        db_n_save_sample = gr.Number(label="Number of samples to generate", value=1)
                        db_save_guidance_scale = gr.Number(label="Sample guidance scale", value=7.5, max=12, min=1,
                                                           precision=2)
                        db_save_infer_steps = gr.Number(label="Sample steps", value=40, min=10, max=200)

                    with gr.Accordion(open=False, label="Advanced"):
                        with gr.Row():
                            with gr.Column():
                                db_use_cpu = gr.Checkbox(label="Use CPU Only (SLOW)", value=False)
                                db_not_cache_latents = gr.Checkbox(label="Don't cache latents", value=True)
                                db_train_text_encoder = gr.Checkbox(label="Train Text Encoder", value=True)
                                db_use_8bit_adam = gr.Checkbox(label="Use 8bit Adam", value=False)
                                db_center_crop = gr.Checkbox(label="Center Crop", value=False)
                                db_gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True)
                                db_scale_lr = gr.Checkbox(label="Scale Learning Rate", value=False)
                                db_mixed_precision = gr.Dropdown(label="Mixed Precision", value="no",
                                                                 choices=["no", "fp16", "bf16"])
                                db_lr_scheduler = gr.Dropdown(label="Scheduler", value="constant",
                                                              choices=["linear", "cosine", "cosine_with_restarts",
                                                                       "polynomial", "constant",
                                                                       "constant_with_warmup"])
                                db_prior_loss_weight = gr.Number(label="Prior Loss Weight", precision=1, value=1)
                                db_num_train_epochs = gr.Number(label="# Training Epochs", precision=1, value=1)
                                db_adam_beta1 = gr.Number(label="Adam Beta 1", precision=1, value=0.9)
                                db_adam_beta2 = gr.Number(label="Adam Beta 2", precision=3, value=0.999)
                                db_adam_weight_decay = gr.Number(label="Adam Weight Decay", precision=3, value=0.01)
                                db_adam_epsilon = gr.Number(label="Adam Epsilon", precision=8, value=0.00000001)
                                db_max_grad_norm = gr.Number(label="Max Grad Norms", value=1.0, precision=1)
                                db_gradient_accumulation_steps = gr.Number(label="Grad Accumulation Steps", precision=1,
                                                                           value=1)
                                db_lr_warmup_steps = gr.Number(label="Warmup Steps", precision=1, value=0)
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML(value="")

            with gr.Column(variant="panel"):
                db_output = gr.Text(elem_id="db_output", value="", show_label=False)
                db_preview = gr.Image(elem_id='db_preview', visible=False)
                db_progress = gr.HTML(elem_id="db_progress", value="")
                db_progressbar = gr.HTML(elem_id="db_progressbar")
                db_outcome = gr.HTML(elem_id="db_error", value="")
                setup_progressbar(db_progressbar, db_preview, 'db', textinfo=db_progress)

        db_create_embedding.click(
            fn=conversion.extract_checkpoint,
            inputs=[
                db_new_model_name,
                src_checkpoint,
                diff_type
            ],
            outputs=[
                db_pretrained_model_name_or_path,
                db_output,
                db_outcome,
            ]
        )

        db_train_embedding.click(
            fn=wrap_gradio_gpu_call(dreambooth.start_training, extra_outputs=[gr.update()]),
            _js="start_training_dreambooth",
            inputs=[
                db_pretrained_model_name_or_path,
                db_instance_data_dir,
                db_class_data_dir,
                db_instance_prompt,
                db_class_prompt,
                db_save_sample_prompt,
                db_save_sample_negative_prompt,
                db_n_save_sample,
                db_save_guidance_scale,
                db_save_infer_steps,
                db_num_class_images,
                db_resolution,
                db_center_crop,
                db_train_text_encoder,
                db_train_batch_size,
                db_sample_batch_size,
                db_num_train_epochs,
                db_max_train_steps,
                db_gradient_accumulation_steps,
                db_gradient_checkpointing,
                db_learning_rate,
                db_scale_lr,
                db_lr_scheduler,
                db_lr_warmup_steps,
                db_use_8bit_adam,
                db_adam_beta1,
                db_adam_beta2,
                db_adam_weight_decay,
                db_adam_epsilon,
                db_max_grad_norm,
                db_save_preview_every,
                db_save_embedding_every,
                db_mixed_precision,
                db_not_cache_latents,
                db_concepts_list,
                db_use_cpu
            ],
            outputs=[
                db_output,
                db_outcome,
            ]
        )

        db_load_params.click(
            fn=dreambooth.load_params,
            inputs=[
                db_pretrained_model_name_or_path,
                db_class_prompt,
                db_learning_rate,
                db_instance_data_dir,
                db_class_data_dir,
                db_max_train_steps,
                db_save_preview_every,
                db_save_embedding_every,
                db_num_class_images,
                db_use_cpu,
                db_train_text_encoder,
                db_use_8bit_adam,
                db_center_crop,
                db_gradient_checkpointing,
                db_scale_lr,
                db_mixed_precision,
                db_lr_scheduler,
                db_resolution,
                db_prior_loss_weight,
                db_num_train_epochs,
                db_adam_beta1,
                db_adam_beta2,
                db_adam_weight_decay,
                db_adam_epsilon,
                db_max_grad_norm,
                db_train_batch_size,
                db_sample_batch_size,
                db_gradient_accumulation_steps,
                db_lr_warmup_steps
            ],
            outputs=[
                db_pretrained_model_name_or_path,
                db_instance_data_dir,
                db_class_data_dir,
                db_instance_prompt,
                db_class_prompt,
                db_save_sample_prompt,
                db_save_sample_negative_prompt,
                db_n_save_sample,
                db_save_guidance_scale,
                db_save_infer_steps,
                db_num_class_images,
                db_resolution,
                db_center_crop,
                db_train_text_encoder,
                db_train_batch_size,
                db_sample_batch_size,
                db_num_train_epochs,
                db_max_train_steps,
                db_gradient_accumulation_steps,
                db_gradient_checkpointing,
                db_learning_rate,
                db_scale_lr,
                db_lr_scheduler,
                db_lr_warmup_steps,
                db_use_8bit_adam,
                db_adam_beta1,
                db_adam_beta2,
                db_adam_weight_decay,
                db_adam_epsilon,
                db_max_grad_norm,
                db_save_preview_every,
                db_save_embedding_every,
                db_mixed_precision,
                db_not_cache_latents,
                db_concepts_list,
                db_use_cpu
            ]
        )

        db_interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    return (dreambooth_interface, "Dreambooth", "dreambooth_interface"),


script_callbacks.on_ui_tabs(on_ui_tabs)