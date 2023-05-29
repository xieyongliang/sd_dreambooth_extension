import modules.shared as shared
import json
import requests
import os
import shutil
import traceback
from modules import paths

from dreambooth.db_config import DreamboothConfig
from scripts.dreambooth import performance_wizard, training_wizard, start_training_from_config, create_model
from dreambooth.db_concept import Concept

def train_dreambooth(api_endpoint, train_args, sd_models_s3uri, db_models_s3uri, lora_models_s3uri, username):
    db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']
    db_use_txt2img = train_args['train_dreambooth_settings']['db_use_txt2img']
    db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
    db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
    db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

    if db_create_new_db_model:
        db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
        db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
        db_new_model_scheduler = train_args['train_dreambooth_settings']['db_new_model_scheduler']
        db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
        db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
        db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
        db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
        db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
        db_512_model = train_args['train_dreambooth_settings']['db_512_model']

        db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = create_model(
            db_new_model_name,
            db_new_model_src,
            db_new_model_scheduler,
            db_create_from_hub,
            db_new_model_url,
            db_new_model_token,
            db_new_model_extract_ema,
            db_train_unfrozen,
            db_512_model
        )
        dreambooth_config_id = shared.cmd_opts.dreambooth_config_id
        try:
            with open(f'/opt/ml/input/data/config/{dreambooth_config_id}.json', 'r') as f:
                content = f.read()
        except Exception:
            params = {'module': 'dreambooth_config', 'dreambooth_config_id': dreambooth_config_id}
            response = requests.get(url=f'{api_endpoint}/sd/models', params=params)
            if response.status_code == 200:
                content = response.text
            else:
                content = None

        if content:
            params_dict = json.loads(content)

            params_dict['db_model_name'] = db_model_name
            params_dict['db_model_path'] = db_model_path
            params_dict['db_revision'] = db_revision
            params_dict['db_epochs'] = db_epochs
            params_dict['db_scheduler'] = db_scheduler
            params_dict['db_src'] = db_src
            params_dict['db_has_ema'] = db_has_ema
            params_dict['db_v2'] = db_v2
            params_dict['db_resolution'] = db_resolution

            if db_train_wizard_person or db_train_wizard_object:
                db_num_train_epochs, \
                c1_num_class_images_per, \
                c2_num_class_images_per, \
                c3_num_class_images_per, \
                c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

                params_dict['db_num_train_epochs'] = db_num_train_epochs
                params_dict['c1_num_class_images_per'] = c1_num_class_images_per
                params_dict['c1_num_class_images_per'] = c2_num_class_images_per
                params_dict['c1_num_class_images_per'] = c3_num_class_images_per
                params_dict['c1_num_class_images_per'] = c4_num_class_images_per
            if db_performance_wizard:
                attention, \
                gradient_checkpointing, \
                gradient_accumulation_steps, \
                mixed_precision, \
                cache_latents, \
                sample_batch_size, \
                train_batch_size, \
                stop_text_encoder, \
                use_8bit_adam, \
                use_lora, \
                use_ema, \
                save_samples_every, \
                save_weights_every = performance_wizard()

                params_dict['attention'] = attention
                params_dict['gradient_checkpointing'] = gradient_checkpointing
                params_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
                params_dict['mixed_precision'] = mixed_precision
                params_dict['cache_latents'] = cache_latents
                params_dict['sample_batch_size'] = sample_batch_size
                params_dict['train_batch_size'] = train_batch_size
                params_dict['stop_text_encoder'] = stop_text_encoder
                params_dict['use_8bit_adam'] = use_8bit_adam
                params_dict['use_lora'] = use_lora
                params_dict['use_ema'] = use_ema
                params_dict['save_samples_every'] = save_samples_every 
                params_dict['params_dict'] = save_weights_every

            db_config = DreamboothConfig(db_model_name)
            concept_keys = ["c1_", "c2_", "c3_", "c4_"]
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

            db_config.load_params(params_dict)
    else:
        db_model_name = train_args['train_dreambooth_settings']['db_model_name']
        db_config = DreamboothConfig(db_model_name)

    print(vars(db_config))
    start_training_from_config(
        db_config,
        db_use_txt2img,
    )

    cmd_sd_models_path = shared.cmd_opts.ckpt_dir
    sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
    if cmd_sd_models_path is not None:
        sd_models_dir = cmd_sd_models_path

    try:
        cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
    except:
        cmd_dreambooth_models_path = None

    try:
        cmd_lora_models_path = shared.cmd_opts.lora_models_path
    except:
        cmd_lora_models_path = None

    db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
    db_model_dir = os.path.join(db_model_dir, "dreambooth")

    lora_model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
    lora_model_dir = os.path.join(lora_model_dir, "lora")

    print('---models path---', sd_models_dir, lora_model_dir)
    print(os.system(f'ls -l {sd_models_dir}'))
    print(os.system('ls -l {0}'.format(os.path.join(sd_models_dir, db_model_name))))
    print(os.system(f'ls -l {lora_model_dir}'))

    try:
        print('Uploading SD Models...')
        if db_config.v2:
            shared.upload_s3files(
                f'{sd_models_s3uri}{username}/',
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
            )
        if db_config.save_safetensors:
            shared.upload_s3files(
                f'{sd_models_s3uri}{username}/',
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
            )
        else:
            shared.upload_s3files(
                f'{sd_models_s3uri}{username}/',
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
            )
        print('Uploading DB Models...')
        shared.upload_s3folder(
            f'{db_models_s3uri}{username}/{db_model_name}',
            os.path.join(db_model_dir, db_model_name)
        )
        if db_config.use_lora:
            print('Uploading Lora Models...')
            shared.upload_s3files(
                f'{lora_models_s3uri}{username}/',
                os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
            )
        #automatic tar latest checkpoint and upload to s3 by zheng on 2023.03.22
        os.makedirs(os.path.dirname("/opt/ml/model/"), exist_ok=True)
        os.makedirs(os.path.dirname("/opt/ml/model/Stable-diffusion/"), exist_ok=True)
        os.makedirs(os.path.dirname("/opt/ml/model/ControlNet/"), exist_ok=True)
        
        train_steps=int(db_config.revision)
        model_file_basename = f'{db_model_name}_{train_steps}_lora' if db_config.use_lora else f'{db_model_name}_{train_steps}'
        if db_config.v2:
            f1=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.yaml')
            if os.path.exists(f1):
                shutil.copy(f1,"/opt/ml/model/Stable-diffusion/")
        if db_config.save_safetensors:
            f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.safetensors')
            if os.path.exists(f2):
                shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
        else:
            f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.ckpt')
            if os.path.exists(f2):
                shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
    except Exception as e:
        traceback.print_exc()
        print(e)
