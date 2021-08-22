import paddle
import paddle.nn as nn
import paddle.nn.functional as F
"""
VGG implementation from [InterDigitalInc](https://github.com/InterDigitalInc/HRFAE/blob/master/nets.py)
"""

class VGG(nn.Layer):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8_101 = nn.Linear(4096, 101)
        if pool == 'max':
            self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2D(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        batch_size = out['p5'].shape[0]
        out['p5'] = paddle.reshape(out['p5'], (batch_size, -1))
        out['fc6'] = F.relu(self.fc6(out['p5']))
        out['fc7'] = F.relu(self.fc7(out['fc6']))
        out['fc8'] = self.fc8_101(out['fc7'])
        return out


# def convert(model):
#     import torch
#     save_dst = r'D:\code_sources\from_github\paddlepaddle\SAM\paddle_models\vgg_aging'
#     torch_model = r'D:\code_sources\VSCode\SAM_P\criteria\reload_models\dex_vgg.pth'
#     torch_dict = torch.load(torch_model, map_location="cpu")
#     paddle_dict = {}
#     fc_keys = ['fc6.weight', 'fc7.weight', 'fc8_101.weight']
#     for key in torch_dict:
#         weight = torch_dict[key].numpy()
#         flag = [i in key for i in fc_keys]
#         if any(flag):
#             print(f'weight {key} need to be traned')
#             weight = weight.transpose()
#         paddle_dict[key] = weight
#     paddle.save(paddle_dict, "vgg_aging.pdparams")
#

# if __name__ == '__main__':
#     import numpy as np
#     np.random.seed(42)
#     x = np.random.randn(1, 3, 224, 224)
#
#     model_t = VGG_t()
#     model_t.load_state_dict(torch.load(torch_model))
#     model_t.eval()
#
#     model_p = VGG()
#     # convert(model)
#     model_p.set_state_dict(paddle.load('vgg_aging.pdparams'))
#     model_p.eval()
#
#     xp = paddle.to_tensor(x, dtype='float32')
#     xt = torch.tensor(x, dtype=torch.float32)
#     print(model_t(xt)['fc8'])
#     print(model_p(xp)['fc8'])
