import torchvision
import torch.nn as nn
import torch

# resnet = torchvision.models.resnet.resnet50(pretrained=True)

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    )
    return model


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    )
    return model

###########################################################################
# Code overlaps with previous assignments : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.layer = nn.Sequential(
                ##########################################
                ############## fill in here
                # Hint : use these functions (conv1x1, conv3x3)
                conv1x1(in_channels, middle_channels, 2, 0),
                conv3x3(middle_channels, middle_channels, 1, 1),
                conv1x1(middle_channels, out_channels, 1, 0)
                #########################################
            )
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            self.layer = nn.Sequential(
                ##########################################
                ############# fill in here
                conv1x1(in_channels, middle_channels, 1, 0),
                conv3x3(middle_channels, middle_channels, 1, 1),
                conv1x1(middle_channels, out_channels, 1, 0)
                #########################################
            )
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            out = self.layer(x)
            x = self.downsize(x)
            return self.activation(out + x)     # This part is slightly different from previous assignments of 'OSP-Lec14-CNN architecture-practice-v2.pdf'
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return self.activation(out + x)     # This part is slightly different from previous assignments of 'OSP-Lec14-CNN architecture-practice-v2.pdf'

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3: kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),  # When inplace = TRUE, ReLU modifies input activations, without allocating additional outputs. This often decrease the memory usage, but may sometimes cause some errors.
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class ResNet50_layer4(nn.Module):
    def __init__(self, num_classes=10): # Hint : How many classes in Cifar-10 dataset?
        super(ResNet50_layer4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3), #blank#, #blank#, #blank#, #blank#, #blank# ),
                # Hint : Through this conv-layer, the input image size is halved.
                #        Consider stride, kernel size, padding and input & output channel sizes.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)#blank#, #blank#, #blank#)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 256, False),#blank#, #blank#, #blank#, #blank#),
            ResidualBlock(256, 64, 256, False),#blank#, #blank#, #blank#, #blank#),
            ResidualBlock(256, 64, 256, True)#blank#, #blank#,#blank#, #blank#)
        )
        self.layer3 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ####### you can refer to the 'layer2' above

            ResidualBlock(256, 128, 512, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512, True)  # blank#, #blank#,#blank#, #blank#)

            #########################################
        )
        self.layer4 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ####### you can refer to the 'layer2' above

            ResidualBlock(512, 256, 1024, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(1024, 256, 1024, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(1024, 256, 1024, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(1024, 256, 1024, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(1024, 256, 1024, False),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(1024, 256, 1024, False)  # blank#, #blank#,#blank#, #blank#)

            #########################################
        )

        self.fc = nn.Linear(1024, num_classes) #blank#, #blank#) # Hint : Think about the reason why fc layer is needed
        self.avgpool = nn.AvgPool2d(2, 1) #blank#, #blank#)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                
                
class UNetWithResnet50Encoder(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3), # Code overlaps with previous assignments
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(3, 2, 1, return_indices=True)

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 256),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(256, 64, 256),  # blank#, #blank#, #blank#, #blank#),
            ResidualBlock(256, 64, 256, downsample=True)# Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(256, 128, 512),  # blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512),  # blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512),  # blank#, #blank#, #blank#),
            ResidualBlock(512, 128, 512, downsample=False)  # Code overlaps with previous assignments
        )
        self.bridge = conv(512, 512)
        self.UnetConv1 = conv(512, 256)
        self.UpConv1 = nn.Conv2d(512, 256, 3, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.upconv2_2 = nn.Conv2d(256, 64, 3, padding=1)

        self.unpool = nn.MaxUnpool2d(3, 2, 1)
        self.UnetConv2_1 = nn.ConvTranspose2d(64, 64, 3, 2, 1)
        self.UnetConv2_2 = nn.ConvTranspose2d(128, 128, 3, 2, 1)
        self.UnetConv2_3 = nn.Conv2d(128, 64, 3, padding=1)

        self.UnetConv3 = nn.Conv2d(64, self.n_classes, kernel_size=1, stride=1)

    ###########################################################################
    # Question 2 : Implement the forward function of Resnet_encoder_UNet.
    # Understand ResNet, UNet architecture and fill in the blanks below. (20 points)
    def forward(self, x, with_output_feature_map=False): #256
        out1 = self.layer1(x)
        out1, indices = self.pool(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.bridge(out3) # bridge
        x = self.UpConv1(x)
        x = torch.cat([out2, x], 1)#######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = self.UnetConv1(x)
        x = self.upconv2_1(x, output_size= torch.Size([x.size(0), 256, 64, 64]))
        x = self.upconv2_2(x)
        x = torch.cat([out1, x], 1)#######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = self.UnetConv2_2(x, output_size=torch.Size([x.size(0), 128, 128, 128]))
        x = self.UnetConv2_2(x, output_size=torch.Size([x.size(0), 128, 256, 256]))
        x = self.UnetConv2_3(x)
        x = self.UnetConv3(x)
        return x

