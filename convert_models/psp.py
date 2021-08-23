"""
This file defines the core research contribution
"""
import copy
from argparse import Namespace

import paddle
from paddle import nn
import math

from configs.paths_config import model_paths
from convert_models.encoders import psp_encoders
from convert_models.stylegan2.model import Generator


class pSp(nn.Layer):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = nn.AdaptiveAvgPool2D((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        return psp_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, self.opts)

    def load_weights(self):
        if self.opts.checkpoint_path is not None:  # 预测加载
            print(f'Loading SAM from checkpoint: {self.opts.checkpoint_path}')
            ckpt = paddle.load(self.opts.checkpoint_path)
            self.encoder.set_state_dict(self.__get_keys(ckpt, 'encoder'))
            self.decoder.set_state_dict(self.__get_keys(ckpt, 'decoder'))
            if self.opts.start_from_encoded_w_plus:
                self.pretrained_encoder = self.__get_pretrained_psp_encoder()
                self.pretrained_encoder.set_state_dict(self.__get_keys(ckpt, 'pretrained_encoder'))
            if ckpt['latent_avg'] is not None:
                self.latent_avg = ckpt['latent_avg']  # .to(self.opts.device)
        else:  # 训练加载
            if self.opts.train_from_sam_ckpt:
                print(f'Loading encoder weights from: {self.opts.sam_encoder}')
                self.encoder.set_state_dict(paddle.load(self.opts.sam_encoder))
                print(f'Loading decoder weights from: {self.opts.sam_decoder}')
                self.decoder.set_state_dict(paddle.load(self.opts.sam_decoder))
                if self.opts.start_from_encoded_w_plus:
                    self.pretrained_encoder = self.__get_pretrained_psp_encoder()
                    self.pretrained_encoder.set_state_dict(paddle.load(self.opts.sam_psp3_encoder))
            else:
                print('Loading encoders weights from irse50!')
                encoder_ckpt = paddle.load(model_paths['psp_encoder'])
                # Transfer the RGB input of the irse50 network to the first 3 input channels of SAM's encoder
                # if self.opts.input_nc != 3:
                #     shape = encoder_ckpt['input_layer.0.weight'].shape
                #     altered_input_layer = paddle.randn([shape[0], self.opts.input_nc, shape[2], shape[3]], dtype='float32')
                #     altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                #     encoder_ckpt['input_layer.0.weight'] = altered_input_layer
                self.encoder.set_state_dict(encoder_ckpt)
                print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
                decoder_ckpt = paddle.load(self.opts.stylegan_decoder)
                self.decoder.set_state_dict(decoder_ckpt)

                if self.opts.start_from_encoded_w_plus:
                    self.pretrained_encoder = self.__load_pretrained_psp_encoder()

            # self.pretrained_encoder.eval()
            for name, param in self.pretrained_encoder.named_parameters():
                param.trainable = False
            latent_avg = paddle.load(self.opts.latent_avg)
            self.__load_latent_avg(latent_avg, repeat=self.n_styles)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                codes = codes + self.latent_avg
            # normalize with respect to the latent of the encoded image of pretrained pSp encoder
            elif self.opts.start_from_encoded_w_plus:
                with paddle.no_grad():
                    encoded_latents = self.pretrained_encoder(x[:, :-1, :, :])
                    encoded_latents = encoded_latents + self.latent_avg
                codes = codes + encoded_latents

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = (not input_code) or (input_is_full)
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, latent_avg, repeat=None):
        if latent_avg is not None:
            self.latent_avg = latent_avg  # .to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.tile([repeat, 1])
        else:
            self.latent_avg = None

    def __get_pretrained_psp_encoder(self):
        opts_encoder = vars(copy.deepcopy(self.opts))
        opts_encoder['input_nc'] = 3
        opts_encoder = Namespace(**opts_encoder)
        encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, opts_encoder)
        return encoder

    def __load_pretrained_psp_encoder(self):
        print(f'Loading pSp encoder from checkpoint: {self.opts.psp_ffhq_encoder}')
        encoder_ckpt = paddle.load(self.opts.psp_ffhq_encoder)
        # encoder_ckpt = self.__get_keys(ckpt, name='encoder')
        encoder = self.__get_pretrained_psp_encoder()
        encoder.set_state_dict(encoder_ckpt)
        return encoder

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
