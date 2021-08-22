import paddle
from paddle.nn import Linear, Conv2D, BatchNorm2D, PReLU, Dropout, Sequential, Layer
from convert_models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm

"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class BatchNorm1D(paddle.nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super(BatchNorm1D, self).__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class Backbone(Layer):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2D(3, 64, (3, 3), 1, 1, bias_attr=False),
                                      BatchNorm2D(64),
                                      PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1D(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1D(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def main():
    import numpy as np
    import torch
    import paddle
    from configs.paths_config import model_paths

    pnet_names = np.loadtxt("./p_dict.txt", dtype=str)
    tnet_names = np.loadtxt("D:/code_sources/VSCode/SAM_P/t_dict.txt", dtype=str)
    print("len p net names:", len(pnet_names), "len t net:", len(tnet_names))
    diff = list(set(tnet_names) - set(pnet_names))
    same = list(set(tnet_names) & set(pnet_names))
    diff_in_p = list(set(pnet_names) - set(same))
    print("diff_in_p num:", len(diff_in_p))
    print(sorted(diff_in_p))
    print("total diff num:", len(diff))
    print(sorted(diff))
    print("same num:", len(same))
    print(sorted(same))

    t_state_dict = torch.load(model_paths['ir_se50'], map_location='cpu')

    old = {'running_mean': '_mean', 'weight': '_weight', 'running_var': '_variance'}
    p_state_dict = {}
    for key, val in t_state_dict.items():
        val = val.detach().numpy()
        o = key.split('.')[-1]
        if key in diff and o in old.keys():
            key = key.replace(o, old[o])
            p_state_dict[key] = val
        elif 'output_layer.3.weight' == key:
            val = val.transpose()
            p_state_dict[key] = val
        else:
            p_state_dict[key] = val

    paddle.save(p_state_dict, './irse50_backbone.pdparams')
    model = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.set_state_dict(paddle.load('./irse50_backbone.pdparams'))
    model.eval()


if __name__ == '__main__':
    # main()
    pass