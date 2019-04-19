import torch
import torchvision

class FeatureExtractorStemVGG16(torch.nn.Module):
    def __init__(self, vgg=None):
        super(FeatureExtractorStemVGG16, self).__init__()
        if vgg is None:
            vgg = torchvision.models.vgg16(pretrained=False)
        self.features1 = torch.nn.Sequential(vgg.features[:10])
        self.features2 = torch.nn.Sequential(vgg.features[10:17])
        self.features3 = torch.nn.Sequential(vgg.features[17:24])
        self.features4 = torch.nn.Sequential(vgg.features[24:])

    def forward(self, inputs):
        out4 = self.features1(inputs)
        out3 = self.features2(out4)
        out2 = self.features3(out3)
        out1 = self.features4(out2)
        return out1, out2, out3, out4

class FeatureMergingBranch(torch.nn.Module):
    def __init__(self):
        super(FeatureMergingBranch, self).__init__()
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2,stride=2,padding=0)
        self.h2_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1),
                torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        )
        self.h3_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256+256,out_channels=128,kernel_size=1),
                torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        )
        self.h4_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128+128,out_channels=64,kernel_size=1),
                torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        )
        self.g4_conv = torch.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3)

    def forward(self, f1, f2, f3, f4):
        h1 = f1
        g1 = self.unpool(h1, generate_unpool_indices(h1.size(),2,2))

        h2 = self.h2_conv(torch.cat([g1,f2],1))
        g2 = self.unpool(h2, generate_unpool_indices(h2.size(),2,2))

        h3 = self.h3_conv(torch.cat([g2,f3],1))
        g3 = self.unpool(h3, generate_unpool_indices(h3.size(),2,2))

        h4 = self.h4_conv(torch.cat([g3,f4],1))
        g4 = self.g4_conv(h4)

        return g4

class EASTNetwork(torch.nn.Module):
    def __init__(self):
        super(EASTNetwork, self).__init__()
        self.feature_extractor_stem = FeatureExtractorStemVGG16()
        self.feature_merging_branch = FeatureMergingBranch()

        self.score_map_conv = torch.nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.text_boxes_conv = torch.nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
        self.text_rotation_conv = torch.nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.text_quad_conv = torch.nn.Conv2d(in_channels=32,out_channels=8,kernel_size=1)

    def forward(self, inputs):
        f1,f2,f3,f4 = self.feature_extractor_stem(inputs)
        out = self.feature_merging_branch(f1,f2,f3,f4)
        score_map = self.score_map_conv(out)
        text_boxes = self.text_boxes_conv(out)
        text_rotation = self.text_rotation_conv(out)
        text_quad = self.text_quad_conv(out)
        return score_map, text_boxes, text_rotation, text_quad

def generate_unpool_indices(input_dimensions,kernel_size,stride):
    # Create an indices tensor where the max comes from the top-left corner
    if len(input_dimensions) != 4:
        raise NotImplementedError('generate_unpool_indices only supports 2D unpooling.')
    width = input_dimensions[2]
    height = input_dimensions[3]
    i = torch.arange(0,width*height,dtype=torch.long)
    i = i%width*stride+i/width*stride*width
    i = i.repeat(input_dimensions[0],input_dimensions[1],1)
    return i.view(*input_dimensions)

inputs = torch.rand([1,3,224,224])
net = EASTNetwork()
out = net(inputs)
