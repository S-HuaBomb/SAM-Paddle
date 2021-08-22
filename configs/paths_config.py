import os

project_root = r"/public/home/jd_shb/code_source/SAM_P"  # 项目根目录
model_root = os.path.join(project_root, 'pretrained_models')  # 用于训练的预训练模型目录
image_root = os.path.join(project_root, 'images/datasets')  # ffhq、celeba_hq数据集目录

train_exp_dir = os.path.join(project_root, 'exp_dir')  # 训练输出目录
infer_exp_dir = os.path.join(project_root, 'infer_exp_dir')  # 预测输出目录
best_model_path = os.path.join(train_exp_dir, 'checkpoints/best_model.pdparams')  # 在训练输出目录中

dataset_paths = {
    'celeba_test': os.path.join(image_root, 'celeba_hq/val512'),  # 2000张图片512×512
    'ffhq': os.path.join(image_root, 'ffhq512unzip'),  # 42000张512×512
}


model_paths = {
    'alex': os.path.join(model_root, 'alex.pdparams'),
    'alex_owt': os.path.join(model_root, 'alex_owt.pdparams'),
    'psp_ffhq_encoder': os.path.join(model_root, 'psp_ffhq_encoder.pdparams'),
    'latent_avg': os.path.join(model_root, 'latent_avg.pdparams'),
    'stylegan_decoder': os.path.join(model_root, 'stylegan2.pdparams'),
    'psp_encoder': os.path.join(model_root, 'psp_encoder.pdparams'),
    'ir_se50_backbone': os.path.join(model_root, 'irse50_backbone.pdparams'),
    'vgg_age_predictor': os.path.join(model_root, 'dex_age_classifier.pdparams'),
    'sam_encoder': os.path.join(model_root, 'sam_encoder.pdparams'),
    'sam_decoder': os.path.join(model_root, 'sam_decoder.pdparams'),
    'sam_psp3_encoder': os.path.join(model_root, 'sam_psp3_encoder.pdparams'),
}
