from collections import namedtuple
import paddle
from paddle.nn import Conv2D, BatchNorm2D, PReLU, ReLU, Sigmoid, MaxPool2D, AdaptiveAvgPool2D, Sequential, Layer

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Layer):
	def forward(self, input):
		batch_size = input.shape[0]
		return paddle.reshape(input, (batch_size, -1))


def l2_norm(input, axis=1):
	norm = paddle.norm(input, 2, axis, True)
	output = paddle.divide(input, norm)
	return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks


class SEModule(Layer):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = AdaptiveAvgPool2D(1)
		self.fc1 = Conv2D(channels, channels // reduction, kernel_size=1, padding=0, bias_attr=False)
		self.relu = ReLU()
		self.fc2 = Conv2D(channels // reduction, channels, kernel_size=1, padding=0, bias_attr=False)
		self.sigmoid = Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class bottleneck_IR(Layer):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2D(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
				BatchNorm2D(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2D(in_channel),
			Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(depth),
			Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False), BatchNorm2D(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class bottleneck_IR_SE(Layer):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2D(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
				BatchNorm2D(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2D(in_channel),
			Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False),
			PReLU(depth),
			Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False),
			BatchNorm2D(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut
