# SAM-Paddle
PaddlePaddle Implementation for "Only a Matter of Style: Age Transformation Using a Style-Based Regression Model" (SIGGRAPH 2021)

年龄转变说明了一个人的外貌随时间的变化。在输入的面部图像上对这种复杂的变换进行准确建模极具挑战性，因为它需要对面部特征和头部形状做出令人信服的、改变较大的变换，同时仍保留原始的可辨识的身份特征。本文提出了一种 image-to-image 的翻译方法，该方法学习将真实的面部图像直接编码到预先训练的 GAN（例如，StyleGAN）的潜在空间中，并受到给定的老化转变。我们采用预先训练的年龄回归网络，用于明确指导编码器生成与所需年龄相对应的潜在编码。

Here [SAM](https://github.com/yuval-alaluf/SAM) is the Official Implementation in PyTorch

![](https://github.com/yuval-alaluf/SAM/blob/master/docs/2195.jpg)
