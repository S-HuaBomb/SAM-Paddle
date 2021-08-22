# SAM-Paddle
PaddlePaddle Implementation for "[Only a Matter of Style: Age Transformation Using a Style-Based Regression Model](https://paperswithcode.com/paper/only-a-matter-of-style-age-transformation)" (SIGGRAPH 2021)

数据集： CelebA 下载地址：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

验收标准： CelebA 人眼评估生成的图像（可参考论文中展示的生成图片 Figure 4，6，8）

---

[SAM](https://github.com/yuval-alaluf/SAM) is the Official Implementation in PyTorch


[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-fli4QJhz-1629644936768)(https://github.com/yuval-alaluf/SAM/blob/master/docs/2195.jpg)]
9644936768)]

### 训练
请在 `configs/paths_config.py` 中定义训练和预测所需的数据路径和模型路径。

例如，在 `configs/paths_config.py` 中定义训练数据集路径：
Name | Description
| --- | --- |
FFHQ512×512 | SAM-Paddle trained on the FFHQ dataset for age transformation.
Celeb A HQ | SAM-Paddle trained on the FFHQ dataset for age transformation.
```
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}
```
然后，在 `configs/data_configs.py` 中，我们定义训练集和测试集：
```
DATASETS = {
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	}
}
```

训练脚本可以在 `scripts/train.py` 中找到。中间训练结果保存到 `opts.exp_dir`。这包括模型、训练输出和测试输出。详情请看 `options/train_options.py`，并把相应的参数在 `default=` 中修改：
```
python scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir=/path/to/experiment \
--workers=6 \
--batch_size=6 \
--test_batch_size=6 \
--test_workers=6 \
--val_interval=2500 \
--save_interval=10000 \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
```

最后，通过运行以下命令来训练 SAM：
```
python train.py --start_from_encoded_w_plus --use_weighted_id_loss --start_from_latent_avg 
```
> Note：按照上面的设置，训练需要 32 GB 的显存。

如果你希望从 SAM 的预训练模型（例如 best_model.pdparams）开始继续训练，可以加上 `--train_from_sam_ckpt`。
#### Best Model
Name | Description
| --- | --- |
best_model.pdparams | SAM-Paddle trained on the FFHQ dataset for age transformation.

此外，我们提供了从头开始训练你自己的 SAM 模型所需的各种辅助模型。包括用于生成输入图像编码的预训练 pSp 编码器模型和用于计算训练期间 aging-loss 的老化分类器。
Name | Description
| --- | --- |
best_model.pdparams | SAM-Paddle trained on the FFHQ dataset for age transformation.
pSp Encoder | pSp taken from pixel2style2pixel trained on the FFHQ dataset for StyleGAN inversion.
FFHQ StyleGAN | StyleGAN model pretrained on FFHQ taken from rosinality with 1024x1024 output resolution.
IR-SE50 Model |Pretrained IR-SE50 model taken from TreB1eN for use in our ID loss during training.
VGG Age Classifier | VGG age classifier from DEX and fine-tuned on the FFHQ-Aging dataset for use in our aging loss
### 测试
Name | Description
| --- | --- |
best_model.pdparams | SAM-Paddle trained on the FFHQ dataset for age transformation.

把 best_model.pdparams 放到你在 `config/paths_config.py` 中指定的路径，然后在 `options/test_options.py` 中指定测试图像路径和 batch_size等信息，最后使用 `scripts/inference_side_by_side.py` 对一组图像进行老化预测：
```
python inference_side_by_side.py
```

### 项目目录树
Path | Description 
| :--- | --- | 
SAM-Paddle | 项目根目录
├  configs | 包含定义模型、数据路径和数据转换的配置的文件夹
├  criteria |	包含各种训练损失函数的文件夹
├  datasets | 包含构建数据集对象和数据增广的文件夹
├  docs | 包含 README 文件中显示的图像的文件夹 
├  environment | 包含我们实验中使用的 Anaconda 环境的文件夹
├convert_models	| 由pytorch转paddle的网络和训练对象的文件夹
│  ├  encoders | 包含神经网络编码器架构的实现的文件夹
│  ├  stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
│  ├  psp.py | Implementation of pSp encoder
│  └  dex_vgg.py	 | Implementation of DEX VGG classifier used in computation of aging loss
├  notebook | Folder with jupyter notebook containing SAM inference playground
├  options | 包含训练和测试的各个选项的文件夹
├  scripts	| Folder with running scripts for training and inference
├  training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
├  utils	| Folder with various utility functions
