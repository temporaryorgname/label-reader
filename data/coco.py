import torch
import torchvision
from torch.utils.data.dataloader import default_collate
import json
import os
from PIL import Image, ImageDraw

####################################################################################################
# Dataset
####################################################################################################

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, image_path, annotation_path, transform=None):
        self.image_path = image_path
        with open(annotation_path) as f:
            self.annotations = json.load(f)
        self.image_names = sorted(list(self.annotations['imgs'].keys()))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if type(index) is str:
            img_name = index
        else:
            img_name = self.image_names[index]
        ann_indices = self.annotations['imgToAnns'][img_name]
        anns = [self.annotations['anns'][str(i)] for i in ann_indices]

        # Load image
        file_name = os.path.join(self.image_path, '%012d.jpg' % int(img_name))
        img = Image.open(file_name).convert('RGB')

        sample = {'img': img, 'anns': anns}
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

####################################################################################################
# Data Augmentation functions
####################################################################################################

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    def __call__(self,sample):
        output = sample.copy()
        output['img'] = self.normalize(output['img'])
        return output

def scale(sample, scale):
    img = sample['img']
    anns = sample['anns']

    width, height = img.size

    resized_img = img.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
    resized_anns = []
    for ann in anns:
        l,t,w,h = ann['bbox']
        a = {
            'bbox': (l*scale, t*scale, w*scale, h*scale)
        }
        resized_anns.append(a)

    return {'img': resized_img, 'anns': resized_anns}

class RandomScale(object):
    def __init__(self, min_size=None, max_size=None):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):
        img = sample['img']
        anns = sample['anns']

        width, height = img.size

        min_scale = min(self.min_size/min(width,height),1)
        max_scale = 1
        if self.max_size is not None:
            max_scale = min(self.max_size/max(width,height),1)
        s = torch.rand(1).item()*(max_scale-min_scale)+min_scale
        return scale(sample, s)

class ScaleToFit(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['img']
        anns = sample['anns']

        width, height = img.size

        s = min(self.size/min(width,height),1)
        return scale(sample, s)

def crop(sample, crop_left, crop_top, crop_w, crop_h):
    img = sample['img']
    anns = sample['anns']

    width, height = img.size

    cropped_img = torchvision.transforms.functional.crop(img,crop_top,crop_left,crop_h,crop_w)
    cropped_anns = []
    for ann in anns:
        l,t,w,h = ann['bbox']
        if l+w <= crop_left:
            continue
        if t+h <= crop_top:
            continue
        if l >= crop_left+crop_w:
            continue
        if t >= crop_top+crop_h:
            continue
        if l < crop_left:
            w -= crop_left-l
            l = crop_left
        if t < crop_top:
            h -= crop_top-t
            t = crop_top
        if l+w > crop_left+crop_w:
            w -= (l+w)-(crop_left+crop_w)
        if t+h > crop_top+crop_h:
            h -= (t+h)-(crop_top+crop_h)
        if w < 5 or h < 5:
            continue
        l -= crop_left
        t -= crop_top
        cropped_anns.append({'bbox': (l,t,w,h)})

    return {'img': cropped_img, 'anns': cropped_anns}

class RandomCrop(object):
    def __init__(self, width, height):
        self.width = width
        self.height= height

    def __call__(self, sample):
        width, height = sample['img'].size
        crop_w, crop_h = self.width, self.height
        crop_left = int(torch.rand(1).item()*(width-crop_w))
        crop_top = int(torch.rand(1).item()*(height-crop_h))
        return crop(sample, crop_left, crop_top, crop_w, crop_h)

class CentreCrop(object):
    def __init__(self, width, height):
        self.width = width
        self.height= height

    def __call__(self, sample):
        width, height = sample['img'].size
        crop_w, crop_h = self.width, self.height
        crop_left = int(width/2-crop_w/2)
        crop_top = int(height/2-crop_h/2)
        return crop(sample, crop_left, crop_top, crop_w, crop_h)

class ToTensor(object):
    def __init__(self, window_size, clip=True):
        self.window_size = window_size
        self.to_tensor = torchvision.transforms.ToTensor()
        self.clip = clip

    def __call__(self, sample):
        img = sample['img']
        anns = sample['anns']

        width, height = img.size
        ws = self.window_size

        if width % ws != 0 or height % ws != 0:
            if self.clip:
                img = img.crop((0,0,int(width/ws)*ws,int(height/ws)*ws))
            else:
                raise Exception('(%d x %d) image cannot be divided into %d pixel windows.' % (width, height, ws))

        confidence = torch.zeros([1,int(width/ws),int(height/ws)])
        bbox_centre = torch.zeros([2,int(width/ws),int(height/ws)])
        bbox_size = torch.zeros([2,int(width/ws),int(height/ws)])

        for ann in anns:
            left, top, w, h = ann['bbox']
            centre = (left+w/2, top+h/2)
            indices = (int(centre[0]/ws),int(centre[1]/ws))
            dx = centre[0]-indices[0]*ws
            dy = centre[1]-indices[1]*ws
            try:
                confidence[0,indices[0],indices[1]] = 1
                bbox_centre[:,indices[0],indices[1]] = torch.Tensor([dx,dy])
                bbox_size[:,indices[0],indices[1]] = torch.Tensor([w,h])
            except IndexError as e:
                if self.clip:
                    continue
                else:
                    raise e

        return {
                'img': self.to_tensor(img),
                'confidence': confidence,
                'centre': bbox_centre,
                'size': bbox_size
        }

####################################################################################################
# Visualization
####################################################################################################

def render_to_file(samples, index, file_name, normalized=True, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    transforms = []
    if normalized:
        reverse_std = 1/torch.Tensor(std)
        reverse_mean = -torch.Tensor(mean)*reverse_std
        transforms.append(torchvision.transforms.Normalize(reverse_mean,reverse_std))
    transforms.append(torchvision.transforms.ToPILImage())
    transform = torchvision.transforms.Compose(transforms)
    img = transform(samples['img'][index].cpu())
    img_draw = ImageDraw.Draw(img)
    c = samples['confidence']
    centre = samples['centre']
    size = samples['size']
    ws = img.size[0]/c.size()[2]
    for i in range(c.size()[2]):
        for j in range(c.size()[3]):
            if c[index,0,i,j] > 0.5:
                dx,dy = centre[index,:,i,j]
                w,h = size[index,:,i,j]
                left = i*ws+dx-w/2
                top = j*ws+dy-h/2
                img_draw.rectangle(((left,top),(left+w,top+h)),outline='green')
    img.save(file_name)

if __name__=='__main__':
    dataset = COCODataset('/NOBACKUP/hhuang63/COCO/train2017','/NOBACKUP/hhuang63/COCO/COCO_Text.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for x,presence,centre,size in dataloader:
        print(presence.sum())
        if presence.sum() < 1:
            continue
        transform = torchvision.transforms.ToPILImage()
        img = transform(x[0])
        img_draw = ImageDraw.Draw(img)
        for i in range(7):
            for j in range(7):
                if presence[0,0,i,j] > 0.5:
                    dx,dy = centre[0,:,i,j]
                    w,h = size[0,:,i,j]
                    left = i*32+dx-w/2
                    top = j*32+dy-h/2
                    img_draw.rectangle(((left,top),(left+w,top+h)),outline='green')
        img.save('foo.png')
        break
