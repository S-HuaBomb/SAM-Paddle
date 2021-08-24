import paddle
from paddle import nn
import paddle.nn.functional as F

from configs.paths_config import model_paths
from convert_models.dex_vgg import VGG


class AgingLoss(nn.Layer):

    def __init__(self, opts):
        super(AgingLoss, self).__init__()
        self.age_net = VGG()
        self.age_net.set_state_dict(paddle.load(model_paths['vgg_age_predictor']))
        # self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100
        self.opts = opts

        self.age = paddle.arange(self.min_age,self.max_age+1,dtype=paddle.float32)


    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = predict_age_pb.matmul(self.age)
        return predict_age

    def extract_ages(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        predict_age_pb = self.age_net(x)
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age

    def forward(self, y_hat, y, target_ages, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        input_ages = self.extract_ages(y) / 100.
        output_ages = self.extract_ages(y_hat) / 100.

        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_ages[i]) * 100,
                                   f'output_age_{label}': float(output_ages[i]) * 100,
                                   f'target_age_{label}': float(target_ages[i]) * 100})
            # otherwise, create a new entry for the sample
            else:
                id_logs.append({f'input_age_{label}': float(input_ages[i]) * 100,
                                f'output_age_{label}': float(output_ages[i]) * 100,
                                f'target_age_{label}': float(target_ages[i]) * 100})

        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs
