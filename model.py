import torch.nn as nn
import torchvision.models as models


class multiModel(nn.Module):
    def __init__(self):
        super(multiModel, self).__init__()

        # 定义三个ResNet-18模型
        #         resnet_modality1 = resnet18(pretrained=True)
        #         resnet_modality2 = resnet18(pretrained=True)
        #         resnet_modality3 = resnet18(pretrained=True)

        # 去除各模态的最后全连接层
        self.resnet1 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.resnet2 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.resnet3 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

        # 前面的层不给梯度
        # for param in self.resnet1.parameters():
        #     param.requires_grad = False
        # for param in self.resnet2.parameters():
        #     param.requires_grad = False
        # for param in self.resnet3.parameters():
        #     param.requires_grad = False

        # self.resnet1 = models.resnet18(pretrained=True)
        # self.resnet1.fc = nn.Linear(self.resnet1.fc.in_features, 128)
        # self.resnet2 = models.resnet18(pretrained=True)
        # self.resnet2.fc = nn.Linear(self.resnet1.fc.in_features, 128)
        # self.resnet3 = models.resnet18(pretrained=True)
        # self.resnet3.fc = nn.Linear(self.resnet1.fc.in_features, 128)

        self.fc = nn.Linear(3 * 512, 10)
        self.fc1 = nn.Linear(512, 10)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(512, 10)

        self.fc_1 = nn.Linear(3 * 224 * 224 + 512, 1)
        self.fc_2 = nn.Linear(3 * 224 * 224 + 512, 1)
        self.fc_3 = nn.Linear(3 * 224 * 224 + 512, 1)

    def forward(self, x1, x2, x3):
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        out3 = self.resnet3(x3)

        # MI

        # # 将三个特征拼接在一起
        # combined = torch.cat((out1.view(out1.size(0), -1),
        #                       out2.view(out2.size(0), -1),
        #                       out3.view(out3.size(0), -1)), dim=1)
        #
        # # 全连接层
        # output = self.fc(combined)

        # print(f"out1 size is: {out1.size()}")

        # 每个模态经过一个全连接层
        out1 = nn.Softplus()(self.fc1(out1.view(out1.size(0), -1)))
        out2 = nn.Softplus()(self.fc2(out2.view(out2.size(0), -1)))
        out3 = nn.Softplus()(self.fc3(out3.view(out3.size(0), -1)))

        # return out1, out2, out3, output
        return out1, out2, out3, 0



class DisModel(nn.Module):
    def __init__(self, latent_dim=120):
        super(DisModel, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.discriminator(z)
        return out


# class multiCNNModel(nn.Module):
#     def __init__(self):
#         super(multiCNNModel, self).__init__()
#
#         for i in range(3):
#             setattr(self, f'conv1_{i}', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
#             setattr(self, f'conv2_{i}', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
#             setattr(self, f'conv3_{i}', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
#
#             setattr(self, f'pool_{i}', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
#
#         self.fc1 = nn.Linear(768 * 28 * 28, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, x1, x2, x3):
#         concatenated_x = []
#         for i, x in enumerate((x1, x2, x3)):
#             # 对每个模态都进行相似的处理
#             x = F.relu(getattr(self, f'conv1_{i}')(x))
#             x = getattr(self, f'pool_{i}')(x)
#             x = F.relu(getattr(self, f'conv2_{i}')(x))
#             x = getattr(self, f'pool_{i}')(x)
#             x = F.relu(getattr(self, f'conv3_{i}')(x))
#             x = getattr(self, f'pool_{i}')(x)
#             concatenated_x.append(x)
#
#         x = torch.cat(concatenated_x, dim=1)
#
#         x = x.view(-1, 768 * 28 * 28)
#         print(x.size())
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         return x


if __name__ == '__main__':
    # 打印模型结构
    multiModel = multiModel()
    print(multiModel)
