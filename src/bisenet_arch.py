import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBlock(3, 64, stride=2)
        self.conv2 = ConvBlock(64, 128, stride=2)
        self.conv3 = ConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(512, 128, kernel_size=1, padding=0)
        )
        self.up16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat8 = x
        x = self.layer3(x)
        feat16 = x
        x = self.layer4(x)
        feat32 = x
        
        # global context
        cx = self.global_context(feat32)
        
        feat32_arm = self.arm32(feat32)
        feat32_arm = feat32_arm + cx
        feat32_up = self.up16(feat32_arm)
        
        feat16_arm = self.arm16(feat16)
        feat16_arm = feat16_arm + feat32_up
        
        return feat8, feat16_arm, feat32_arm

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu1(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        
        self.ffm = FeatureFusionModule(256 + 128, 256)
        
        self.conv_out = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        spatial_out = self.spatial_path(x)
        _, context_out16, _ = self.context_path(x)
        
        # Upsample context output to match spatial path output (stride 8)
        context_out16_up = F.interpolate(context_out16, size=spatial_out.size()[2:], mode='bilinear', align_corners=True)
        
        feat_fuse = self.ffm(spatial_out, context_out16_up)
        
        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        return out
