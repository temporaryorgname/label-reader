import torch
import torchvision
from PIL import Image, ImageDraw
import os

import models.vgg
import data.coco

device = torch.device('cuda')

# Initialize model and dataset
net = models.vgg.VGGNetwork().to(device)
transform = torchvision.transforms.Compose([
    data.coco.RandomScale(224),
    data.coco.RandomCrop(224,224),
    data.coco.ToTensor(32),
    data.coco.Normalize()
])
dataset = data.coco.COCODataset('/NOBACKUP/hhuang63/COCO/train2017',
        '/NOBACKUP/hhuang63/COCO/COCO_Text.json', transform=transform)
dataset = torch.utils.data.Subset(dataset,range(10))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#optimizer = torch.optim.Adam(list(net.confidence_conv.parameters())+list(net.box_centre_conv.parameters())+list(net.box_size_conv.parameters()), lr=1e-3)
for p in net.features.parameters():
    p.requires_grad = False

# Load checkpoint
#checkpoint_file = 'checkpoint.pt'
#if os.path.isfile(checkpoint_file):
#    print('Loading checkpoint')
#    checkpoint = torch.load(checkpoint_file)
#    net.load_state_dict(checkpoint['model'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

# Train
iter_count = 0
try:
    while True:
        iter_count += 1
        for samples in dataloader:
            x = samples['img'].to(device)
            c = samples['confidence'].to(device)
            centre = samples['centre'].to(device)
            size = samples['size'].to(device)
            # Predictions
            c_hat, centre_hat, size_hat = net(x)

            # Loss
            optimizer.zero_grad()
            obj = c
            noobj = torch.ones(obj.size()).to(device)-obj
            # IOU
            intersection = (
                    torch.min(centre+size/2,centre_hat+size_hat/2)-torch.max(centre-size/2,centre_hat-size_hat/2)
            ).clamp(0).prod(dim=1,keepdim=True)
            union = size.prod(dim=1,keepdim=True)+size_hat.prod(dim=1,keepdim=True)-intersection
            iou = intersection/union
            # Centres
            centre_loss = ((centre-centre_hat)**2).sum()
            # Confidence weighting
            lambda_obj = 1
            lambda_noobj = ((obj*(1-iou)).sum()/noobj.sum()).detach()
            # Total loss
            epsilon = 1e-10
            loss = ((1-iou)*(-obj*torch.log(c_hat+epsilon))-lambda_noobj*noobj*torch.log(1-c_hat+epsilon)).sum()+1e-5*centre_loss
            # Backprop
            print(loss.item(), c.sum().item(), c_hat.sum().item())
            loss.backward()
            optimizer.step()

            # Render to file
            if iter_count % 10 == 0 and not torch.isnan(loss):
                for i in range(10):
                    if c[i,:,:,:].sum() > 1:
                        data.coco.render_to_file(samples, index=i, file_name='bar.png')
                        data.coco.render_to_file({'img': x, 'confidence': c_hat,
                            'centre': centre_hat, 'size': size_hat}, index=i,
                            file_name='foo.png')
                        break
            if torch.isnan(loss):
                raise ValueError('Got a nan')
            for n,p in net.named_parameters():
                if torch.isnan(p).any():
                    raise ValueError('Got a nan in net params (%s)' % n)
            
except KeyboardInterrupt:
    print('Keyboard Interrupt')

# Save checkpoint
#print('Saving Checkpoint')
#torch.save({
#    'model': net.state_dict(),
#    'optimizer': optimizer.state_dict()
#}, checkpoint_file)
