import modules.shared as shared
import json
import requests
import os
import shutil
import traceback
from modules import paths

from dreambooth.dataclasses.db_config import DreamboothConfig
from dreambooth.ui_functions import performance_wizard, training_wizard, start_training_from_config, create_model
from dreambooth.dataclasses.db_concept import Concept

def train_dreambooth(api_endpoint, train_args, sd_models_s3uri, db_models_s3uri, lora_models_s3uri, username):
    db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']
    db_class_gen_method = train_args['train_dreambooth_settings']['db_class_gen_method']
    db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
    db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
    db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

    if db_create_new_db_model:
        db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
        db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
        db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
        db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
        db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
        db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
        db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
        db_512_model = train_args['train_dreambooth_settings']['db_512_model']

        result = create_model(
                db_new_model_name,
                db_new_model_src,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema,
                db_train_unfrozen,
                db_512_model
            )
        print(result)
        if db_create_from_hub:
            db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = result
        else:
            db_model_name, db_model_path, db_revision, db_epochs, db_src, db_has_ema, db_v2, db_resolution = result
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
            params_dict['db_epoch'] = db_epochs
            params_dict['db_src'] = db_src
            params_dict['db_has_ema'] = db_has_ema
            params_dict['db_v2'] = db_v2
            params_dict['db_resolution'] = db_resolution
            if db_create_from_hub:
                params_dict['db_scheduler'] = db_scheduler

            if db_train_wizard_person or db_train_wizard_object:
                db_num_train_epochs, \
                db_c1_num_class_images_per, \
                db_c2_num_class_images_per, \
                db_c3_num_class_images_per, \
                db_c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

                params_dict['db_num_train_epochs'] = db_num_train_epochs
                params_dict['c1_num_class_images_per'] = db_c1_num_class_images_per
                params_dict['c1_num_class_images_per'] = db_c2_num_class_images_per
                params_dict['c1_num_class_images_per'] = db_c3_num_class_images_per
                params_dict['c1_num_class_images_per'] = db_c4_num_class_images_per
            if db_performance_wizard:
                db_attention, \
                db_gradient_checkpointing, \
                db_gradient_accumulation_steps, \
                db_mixed_precision, \
                db_cache_latents, \
                db_optimizer, \
                db_sample_batch_size, \
                db_train_batch_size, \
                db_stop_text_encoder, \
                db_use_lora, \
                db_use_ema, \
                db_save_preview_every, \
                db_save_embedding_every = performance_wizard()

                params_dict['attention'] = db_attention
                params_dict['gradient_checkpointing'] = db_gradient_checkpointing
                params_dict['gradient_accumulation_steps'] = db_gradient_accumulation_steps
                params_dict['mixed_precision'] = db_mixed_precision
                params_dict['cache_latents'] = db_cache_latents
                params_dict['optimizer'] = db_optimizer
                params_dict['sample_batch_size'] = db_sample_batch_size
                params_dict['train_batch_size'] = db_train_batch_size
                params_dict['stop_text_encoder'] = db_stop_text_encoder
                params_dict['use_lora'] = db_use_lora
                params_dict['use_ema'] = db_use_ema
                params_dict['save_preview_every'] = db_save_preview_every
                params_dict['save_embedding_every'] = db_save_embedding_every

            db_config = DreamboothConfig(db_model_name, src=db_new_model_src if not db_create_from_hub else db_new_model_url)
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
        db_class_gen_method
    )

    cmd_sd_models_path = shared.cmd_opts.ckpt_dir
    sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
    if cmd_sd_models_path is not None:
        sd_models_dir = cmd_sd_models_path

    try:
        cmd_dreambooth_models_path = shared.cmd_opts.dreambooth_models_path
    except:
        cmd_dreambooth_models_path = None

    db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
    db_model_dir = os.path.join(db_model_dir, "dreambooth")

    lora_model_dir = os.path.join(db_model_dir, "lora")

    print('---models path---', sd_models_dir, lora_model_dir)
    print(os.system(f'ls -l {sd_models_dir}'))
    print(os.system('ls -l {0}'.format(os.path.join(sd_models_dir, db_model_name))))
    print(os.system(f'ls -l {lora_model_dir}'))

    try:
        print('Uploading SD Models...')
        s3uri = f'{sd_models_s3uri}{username}/'
        if username == '':
            s3uri = s3uri[0 : s3uri.rfind('/')] + '/'
        if db_config.v2:
            shared.upload_s3files(
                s3uri,
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
            )
        if db_config.save_safetensors:
            shared.upload_s3files(
                s3uri,
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
            )
        else:
            shared.upload_s3files(
                s3uri,
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
            )

        print('Uploading DB Models...')
        if username == '':
            s3uri = f'{db_models_s3uri}{db_model_name}'
        else:
            s3uri = f'{db_models_s3uri}{username}/{db_model_name}'
        shared.upload_s3folder(
            s3uri,
            os.path.join(db_model_dir, db_model_name)
        )

        if db_config.use_lora:
            if username == '':
                s3uri = f'{lora_models_s3uri}'
            else:
                s3uri = f'{lora_models_s3uri}{username}/'
            print('Uploading Lora Models...')
            shared.upload_s3files(
                s3uri,
                os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
            )

        os.makedirs(os.path.dirname("/opt/ml/model/Stable-diffusion/"), exist_ok=True)

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
