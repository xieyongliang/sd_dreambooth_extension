import json
import os
import traceback
from typing import List, Dict

from pydantic import BaseModel

from dreambooth import shared  # noqa
from dreambooth.dataclasses.db_concept import Concept  # noqa
from dreambooth.utils.image_utils import get_scheduler_names  # noqa
from dreambooth.utils.utils import list_attention

# Keys to save, replacing our dumb __init__ method
save_keys = ['db_model_name', 'db_attention', 'db_cache_latents', 'db_clip_skip', 'db_concepts_path', 'db_custom_model_name', 'db_noise_scheduler', 'db_deterministic', 'db_ema_predict', 'db_epochs', 'db_epoch_pause_frequency', 'db_epoch_pause_time', 'db_freeze_clip_normalization', 'db_gradient_accumulation_steps', 'db_gradient_checkpointing', 'db_gradient_set_to_none', 'db_half_model', 'db_hflip', 'db_infer_ema', 'db_learning_rate', 'db_learning_rate_min', 'db_lora_learning_rate', 'db_lora_model_name', 'db_lora_unet_rank', 'db_lora_txt_rank', 'db_lora_txt_learning_rate', 'db_lora_txt_weight', 'db_lora_weight', 'db_lr_cycles', 'db_lr_factor', 'db_lr_power', 'db_lr_scale_pos', 'db_lr_scheduler', 'db_lr_warmup_steps', 'db_max_token_length', 'db_mixed_precision', 'db_adamw_weight_decay', 'db_model_path', 'db_num_train_epochs', 'db_offset_noise', 'db_optimizer', 'db_pad_tokens', 'db_pretrained_vae_name_or_path', 'db_prior_loss_scale', 'db_prior_loss_target', 'db_prior_loss_weight', 'db_prior_loss_weight_min', 'db_resolution', 'db_revision', 'db_sample_batch_size', 'db_sanity_prompt', 'db_sanity_seed', 'db_save_ckpt_after', 'db_save_ckpt_cancel', 'db_save_ckpt_during', 'db_save_embedding_every', 'db_save_ema', 'db_save_lora_after', 'db_save_lora_cancel', 'db_save_lora_during', 'db_save_lora_for_extra_net', 'db_save_preview_every', 'db_save_safetensors', 'db_save_state_after', 'db_save_state_cancel', 'db_save_state_during', 'db_scheduler', 'db_split_loss', 'db_strict_tokens', 'db_shuffle_tags', 'db_snapshot', 'db_src', 'db_tf32_enable', 'db_train_batch_size', 'db_train_imagic_only', 'db_train_unet', 'db_stop_text_encoder', 'db_use_concepts', 'db_train_unfrozen', 'db_use_ema', 'db_use_lora', 'db_use_lora_extended', 'db_use_subdir', 'c1_class_data_dir', 'c1_class_guidance_scale', 'c1_class_infer_steps', 'c1_class_negative_prompt', 'c1_class_prompt', 'c1_class_token', 'c1_instance_data_dir', 'c1_instance_prompt', 'c1_instance_token', 'c1_n_save_sample', 'c1_num_class_images_per', 'c1_sample_seed', 'c1_save_guidance_scale', 'c1_save_infer_steps', 'c1_save_sample_negative_prompt', 'c1_save_sample_prompt', 'c1_save_sample_template', 'c2_class_data_dir', 'c2_class_guidance_scale', 'c2_class_infer_steps', 'c2_class_negative_prompt', 'c2_class_prompt', 'c2_class_token', 'c2_instance_data_dir', 'c2_instance_prompt', 'c2_instance_token', 'c2_n_save_sample', 'c2_num_class_images_per', 'c2_sample_seed', 'c2_save_guidance_scale', 'c2_save_infer_steps', 'c2_save_sample_negative_prompt', 'c2_save_sample_prompt', 'c2_save_sample_template', 'c3_class_data_dir', 'c3_class_guidance_scale', 'c3_class_infer_steps', 'c3_class_negative_prompt', 'c3_class_prompt', 'c3_class_token', 'c3_instance_data_dir', 'c3_instance_prompt', 'c3_instance_token', 'c3_n_save_sample', 'c3_num_class_images_per', 'c3_sample_seed', 'c3_save_guidance_scale', 'c3_save_infer_steps', 'c3_save_sample_negative_prompt', 'c3_save_sample_prompt', 'c3_save_sample_template', 'c4_class_data_dir', 'c4_class_guidance_scale', 'c4_class_infer_steps', 'c4_class_negative_prompt', 'c4_class_prompt', 'c4_class_token', 'c4_instance_data_dir', 'c4_instance_prompt', 'c4_instance_token', 'c4_n_save_sample', 'c4_num_class_images_per', 'c4_sample_seed', 'c4_save_guidance_scale', 'c4_save_infer_steps', 'c4_save_sample_negative_prompt', 'c4_save_sample_prompt', 'c4_save_sample_template']

# Keys to return to the ui when Load Settings is clicked.
ui_keys = []


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in "._- "))


class DreamboothConfig(BaseModel):
    # These properties MUST be sorted alphabetically
    adamw_weight_decay: float = 0.01
    adaptation_beta1: int = 0
    adaptation_beta2: int = 0
    adaptation_d0: float = 1e-8
    adaptation_eps: float = 1e-8
    attention: str = "xformers"
    cache_latents: bool = True
    clip_skip: int = 1
    concepts_list: List[Dict] = []
    concepts_path: str = ""
    custom_model_name: str = ""
    noise_scheduler: str = "DDPM"
    deterministic: bool = False
    ema_predict: bool = False
    epoch: int = 0
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 0
    freeze_clip_normalization: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_set_to_none: bool = True
    graph_smoothing: int = 50
    half_model: bool = False
    train_unfrozen: bool = True
    has_ema: bool = False
    hflip: bool = False
    infer_ema: bool = False
    initial_revision: int = 0
    learning_rate: float = 5e-6
    learning_rate_min: float = 1e-6
    lifetime_revision: int = 0
    lora_learning_rate: float = 1e-4
    lora_model_name: str = ""
    lora_unet_rank: int = 4
    lora_txt_rank: int = 4
    lora_txt_learning_rate: float = 5e-5
    lora_txt_weight: float = 1.0
    lora_weight: float = 1.0
    lr_cycles: int = 1
    lr_factor: float = 0.5
    lr_power: float = 1.0
    lr_scale_pos: float = 0.5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_token_length: int = 75
    mixed_precision: str = "fp16"
    model_name: str = ""
    model_dir: str = ""
    model_path: str = ""
    num_train_epochs: int = 100
    offset_noise: float = 0
    optimizer: str = "8bit AdamW"
    pad_tokens: bool = True
    pretrained_model_name_or_path: str = ""
    pretrained_vae_name_or_path: str = ""
    prior_loss_scale: bool = False
    prior_loss_target: int = 100
    prior_loss_weight: float = 0.75
    prior_loss_weight_min: float = 0.1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    sanity_prompt: str = ""
    sanity_seed: int = 420420
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = False
    save_ckpt_during: bool = True
    save_ema: bool = True
    save_embedding_every: int = 25
    save_lora_after: bool = True
    save_lora_cancel: bool = False
    save_lora_during: bool = True
    save_lora_for_extra_net: bool = True
    save_preview_every: int = 5
    save_safetensors: bool = True
    save_state_after: bool = False
    save_state_cancel: bool = False
    save_state_during: bool = False
    scheduler: str = "ddim"
    shuffle_tags: bool = True
    snapshot: str = ""
    split_loss: bool = True
    src: str = ""
    stop_text_encoder: float = 1.0
    strict_tokens: bool = False
    tf32_enable: bool = False
    train_batch_size: int = 1
    train_imagic: bool = False
    train_unet: bool = True
    use_concepts: bool = False
    use_ema: bool = True
    use_lora: bool = False
    use_lora_extended: bool = False
    use_subdir: bool = False
    v2: bool = False

    def __init__(
            self,
            model_name: str = "",
            v2: bool = False,
            src: str = "",
            resolution: int = 512,
            **kwargs
    ):

        super().__init__(**kwargs)
        model_name = sanitize_name(model_name)
        models_path = shared.dreambooth_models_path

        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")

        # If we're using the new UI, this should be populated, so load models from here.
        if len(shared.paths):
            models_path = os.path.join(shared.paths["models"], "dreambooth")

        if not self.use_lora:
            self.lora_model_name = ""

        model_dir = os.path.join(models_path, model_name)

        working_dir = os.path.join(model_dir, "working")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.model_name = model_name
        self.model_dir = model_dir
        self.pretrained_model_name_or_path = working_dir
        self.resolution = resolution
        self.src = src
        self.scheduler = "ddim"
        self.v2 = v2

    # Actually save as a file
    def save(self, backup=False):
        """
        Save the config file
        """
        models_path = self.model_dir
        config_file = os.path.join(models_path, "db_config.json")
        if backup:
            backup_dir = os.path.join(models_path, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            config_file = os.path.join(models_path, "backups", f"db_config_{self.revision}.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def load_params(self, params_dict):
        sched_swap = False
        for key, value in params_dict.items():
            if "db_" in key:
                key = key.replace("db_", "")
            if key == "attention" and value == "flash_attention":
                value = list_attention()[-1]
                print(f"Replacing flash attention in config to {value}")

            if key == "scheduler":
                schedulers = get_scheduler_names()
                if value not in schedulers:
                    sched_swap = True
                    for scheduler in schedulers:
                        if value.lower() in scheduler.lower():
                            print(f"Updating scheduler name to: {scheduler}")
                            value = scheduler
                            break

            if hasattr(self, key):
                key, value = self.validate_param(key, value)
                setattr(self, key, value)
        if sched_swap:
            self.save()

    @staticmethod
    def validate_param(key, value):
        replaced_params = {
            # "old_key" : {
            #   "new_key": "...",
            #   "values": [{
            #       "old": ["...", "..."]
            #       "new": "..."
            #   }]
            # }
            "deis_train_scheduler": {
                "new_key": "noise_scheduler",
                "values": [{
                    "old": [True],
                    "new": "DDPM"
                }],
            },
            "optimizer": {
                "values": [{
                    "old": ["8Bit Adam"],
                    "new": "8bit AdamW"
                }],
            },
            "save_safetensors": {
                "values": [{
                    "old": [False],
                    "new": True
                }],
            }
        }

        if key in replaced_params.keys():
            replacement = replaced_params[key]
            if hasattr(replacement, "new_key"):
                key = replacement["new_key"]
            if hasattr(replacement, "values"):
                for _value in replacement["values"]:
                    if value in _value["old"]:
                        value = _value["new"]
        return key, value

    # Pass a dict and return a list of Concept objects
    def concepts(self, required: int = -1):
        concepts = []
        c_idx = 0
        # If using a file for concepts and not requesting from UI, load from file
        if self.use_concepts and self.concepts_path and required == -1:
            concepts_list = concepts_from_file(self.concepts_path)

        # Otherwise, use 'stored' list
        else:
            concepts_list = self.concepts_list
        if required == -1:
            required = len(concepts_list)

        for concept_dict in concepts_list:
            concept = Concept(input_dict=concept_dict)
            if concept.is_valid:
                if concept.class_data_dir == "" or concept.class_data_dir is None:
                    concept.class_data_dir = os.path.join(self.model_dir, f"classifiers_{c_idx}")
                concepts.append(concept)
                c_idx += 1

        missing = len(concepts) - required
        if missing > 0:
            concepts.extend([Concept(None)] * missing)
        return concepts

    # Set default values
    def check_defaults(self):
        if self.model_name:
            if self.revision == "" or self.revision is None:
                self.revision = 0
            if self.epoch == "" or self.epoch is None:
                self.epoch = 0
            self.model_name = "".join(x for x in self.model_name if (x.isalnum() or x in "._- "))
            models_path = shared.dreambooth_models_path
            try:
                from core.handlers.models import ModelHandler
                mh = ModelHandler()
                models_path = mh.models_path
            except:
                pass
            if models_path == "" or models_path is None:
                models_path = os.path.join(shared.models_path, "dreambooth")
            model_dir = os.path.join(models_path, self.model_name)
            working_dir = os.path.join(model_dir, "working")
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            self.model_dir = model_dir
            self.pretrained_model_name_or_path = working_dir

    def refresh(self):
        """
        Reload self from file

        """
        models_path = shared.dreambooth_models_path
        if models_path == "" or models_path is None:
            models_path = os.path.join(shared.models_path, "dreambooth")
        config_file = os.path.join(models_path, self.model_name, "db_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config_dict = json.load(openfile)

            self.load_params(config_dict)
            shared.db_model_config = self
        except Exception as e:
            print(f"Exception loading config: {e}")
            traceback.print_exc()
            return None


def concepts_from_file(concepts_path: str):
    concepts = []
    if os.path.exists(concepts_path) and os.path.isfile(concepts_path):
        try:
            with open(concepts_path, "r") as concepts_file:
                concepts_str = concepts_file.read()
        except Exception as e:
            print(f"Exception opening concepts file: {e}")
    else:
        concepts_str = concepts_path

    try:
        concepts_data = json.loads(concepts_str)
        for concept_data in concepts_data:
            concept = Concept(input_dict=concept_data)
            if concept.is_valid:
                concepts.append(concept.__dict__)
    except Exception as e:
        print(f"Exception parsing concepts: {e}")
    return concepts


def save_config(*args):
    params = list(args)
    concept_keys = ["c1_", "c2_", "c3_", "c4_"]
    model_name = params[0]
    if model_name is None or model_name == "":
        print("Invalid model name.")
        return
    params_dict = dict(zip(save_keys, params))
    concepts_list = []
    # If using a concepts file/string, keep concepts_list empty.
    if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
        concepts_list = []
        params_dict["concepts_list"] = concepts_list
    else:
        for concept_key in concept_keys:
            concept_dict = {}
            for key, param in params_dict.items():
                if concept_key in key and param is not None:
                    concept_dict[key.replace(concept_key, "")] = param
            concept_test = Concept(concept_dict)
            if concept_test.is_valid:
                concepts_list.append(concept_test.__dict__)
        existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
        if len(concepts_list) and not len(existing_concepts):
            params_dict["concepts_list"] = concepts_list

    config = from_file(model_name)
    if config is None:
        config = DreamboothConfig(model_name)
    config.load_params(params_dict)
    shared.db_model_config = config
    config.save()


def from_file(model_name):
    """
    Load config data from UI
    Args:
        model_name: The config to load

    Returns: Dict | None

    """
    if isinstance(model_name, list) and len(model_name) > 0:
        model_name = model_name[0]
        
    if model_name == "" or model_name is None:
        return None

    model_name = sanitize_name(model_name)
    models_path = shared.dreambooth_models_path
    if models_path == "" or models_path is None:
        models_path = os.path.join(shared.models_path, "dreambooth")
    config_file = os.path.join(models_path, model_name, "db_config.json")
    try:
        with open(config_file, 'r') as openfile:
            config_dict = json.load(openfile)

        config = DreamboothConfig(model_name)
        config.load_params(config_dict)
        shared.db_model_config = config
        return config
    except Exception as e:
        print(f"Exception loading config: {e}")
        traceback.print_exc()
        return None
