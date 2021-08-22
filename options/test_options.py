import os
from argparse import ArgumentParser
from configs.paths_config import project_root, infer_exp_dir, best_model_path


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str,
                                 default=infer_exp_dir,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path',  # 指定 best_model.pdparams
                                 default=best_model_path,
                                 type=str,
                                 help='Path to pSp model checkpoint')
        self.parser.add_argument('--data_path', type=str,  # 测试输入的人脸图像所在文件夹
                                 default=os.path.join(project_root, "images/sample"),
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--couple_outputs', action='store_true',
                                 help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at 1024x1024')

        self.parser.add_argument('--test_batch_size', default=5, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=1, type=int,
                                 help='Number of test/inference dataloader workers')

        # arguments for style-mixing script
        self.parser.add_argument('--n_images', type=int, default=1,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--n_outputs_to_generate', type=int, default=5,
                                 help='Number of outputs to generate per input image.')
        self.parser.add_argument('--mix_alpha', type=float, default=None,
                                 help='Alpha value for style-mixing')
        self.parser.add_argument('--latent_mask', type=str, default=None,
                                 help='Comma-separated list of latents to perform style-mixing with')

        # arguments for aging
        self.parser.add_argument('--target_age', type=str,
                                 default="10,20,30,40,50,60,70,80",
                                 help='Target age for inference. Can be comma-separated list for multiple ages.')

        # arguments for reference guided aging inference
        self.parser.add_argument('--ref_images_paths_file', type=str, default='./ref_images.txt',
                                 help='Path to file containing a list of reference images to use for '
                                      'reference guided inference.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
