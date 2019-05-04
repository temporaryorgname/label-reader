import torch
import torchvision

class VGGNetwork(torch.nn.Module):
    def __init__(self, vgg=None):
        super(VGGNetwork, self).__init__()
        if vgg is None:
            vgg = torchvision.models.vgg16(pretrained=True)
        self.features = vgg.features
        self.confidence_conv = torch.nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1)
        self.box_centre_conv = torch.nn.Conv2d(in_channels=512,out_channels=2,kernel_size=1)
        self.box_size_conv = torch.nn.Conv2d(in_channels=512,out_channels=2,kernel_size=1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = self.features(x)
        confidence = self.sigmoid(self.confidence_conv(x))
        box_centre = self.sigmoid(self.box_centre_conv(x))*32
        box_size = torch.exp(self.box_size_conv(x))
        return confidence, box_centre, box_size

inputs = torch.rand([1,3,224,224])
net = VGGNetwork()
out = net(inputs)
