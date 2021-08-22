from collections import OrderedDict

import paddle
from configs.paths_config import model_paths


def normalize_activation(x, eps=1e-10):
    norm_factor = paddle.sqrt(paddle.sum(x ** 2, axis=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    # old_state_dict = torch.hub.load_state_dict_from_url(
    #     url, progress=True,
    #     map_location=None if paddle.is_compiled_with_cuda() else 'cpu'
    # )
    old_state_dict = paddle.load(model_paths['alex'])

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
