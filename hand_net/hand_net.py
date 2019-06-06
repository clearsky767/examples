import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import os
import json
import time
import shutil
import argparse
import numpy as np
from PIL import Image, ImageDraw
from resnet import resnet18
from roi_pooling import roi_pooling,adaptive_max_pool


parser = argparse.ArgumentParser(description='HandNet')
parser.add_argument('--trainjson', default='data/train.json', type=str, metavar='PATH',help='json file path')
parser.add_argument('--testjson', default='data/test.json', type=str, metavar='PATH',help='json file path')
parser.add_argument('--batchsize', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',help='path to checkpoint')
parser.add_argument('--m1', default='', type=str, metavar='PATH',help='path to m1')
parser.add_argument('--m2', default='', type=str, metavar='PATH',help='path to m2')
parser.add_argument('--m3', default='', type=str, metavar='PATH',help='path to m3')
parser.add_argument('--m4', default='', type=str, metavar='PATH',help='path to m4')
parser.add_argument('--m5', default='', type=str, metavar='PATH',help='path to m5')
parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')

args = parser.parse_args()


def read_ls_txt(filename):
    fo = open(filename, "r")
    fl = fo.readlines()
    fo.close()
    return [float(e.strip("\n")) for e in fl]

def write_ls_txt(filename,data):
    fo = open(filename, "w+")
    sep = "\n"
    data = [str(e) for e in data]
    fl = fo.write(sep.join(data))
    fo.close()

def read_txt(filename):
    fo = open(filename, "r")
    fl = fo.read()
    fo.close()
    return fl

def write_txt(filename,data):
    fo = open(filename, "w+")
    fl = fo.write(data)
    fo.close()

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return

def show(img,points):
    draw = ImageDraw.Draw(img)
    pts = [tuple(point )for point in points]
    draw.point(pts, fill = (255, 0, 0))
    draw.text((100,100), "hand", fill=(0,255,0))
    img.show()

def show2(img,target_tensor,w,h):
    target = target_tensor.numpy()
    target = target.tolist()
    points = []
    for i in range(0,len(target),2):
        p_w = target[i]*w
        p_h = target[i+1]*h
        points.append(tuple(p_w,p_h))
    show(img,points)

class AVGMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Hand(object):
    def __init__(self, v):
        self.num = 0
        self.points = []
        self.filename = None
        self.load(v)

    def load(self, v):
        self.filename = v["filename"]
        regions = v["regions"]
        for region in regions:
            shape_attr = region["shape_attributes"]
            cx = shape_attr["cx"]
            cy = shape_attr["cy"]
            self.points.append([cx,cy])
            self.num += 1

class HandDataset(data.Dataset):
    def __init__(self, filepath, transform=None, target_transform=None):
        self.path = os.path.dirname(filepath)
        self.hands = []
        self.transform = transform
        self.target_transform = target_transform
        self.loadjson(filepath)

    def __getitem__(self, index):
        hand = self.hands[index]
        img = Image.open(os.path.join(self.path,hand.filename)).convert('RGB')
        w = img.width
        h = img.height
        target = []
        for point in hand.points:
            point[0] = float(point[0])/w
            point[1] = float(point[1])/h
            target.append(point[0])
            target.append(point[1])
        target = torch.tensor(target)
        if self.transform is not None:
            img_tensor = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_tensor, target

    def __len__(self):
        return len(self.hands)

    def loadjson(self,path):
        c = read_txt(path)
        json_obj = json.loads(c)
        for k,v in json_obj.items():
            hand = Hand(v)
            if hand.num == 31 and os.path.exists(os.path.join(self.path,hand.filename)):
                self.hands.append(hand)

class HandNet(nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        )
        self.fc = nn.Linear(512, 2)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 512)
        output = self.fc(output)
        output = self.sig(output)
        return output

class HandNet2(nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        )
        self.fc = nn.Linear(512, 54)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 512)
        output = self.fc(output)
        output = self.sig(output)
        return output

class CascadedNet(object):
    def __init__(self,resnet18,m1,m2,m3,m4,m5,criterion):
        self.resnet18 = resnet18
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.criterion = criterion

    def get_map(self,idx,x,x_map,batch):
        w = (x[:,idx]*10).astype(np.int16)
        h = (x[:,idx+1]*10).astype(np.int16)
        batch_map = None
        for i in range(batch):
            dw = w[i]
            dh = h[i]
            o_map = x_map[i,:,dw,dh]
            o_map = np.expand_dims(o_map,axis=0)
            if batch_map is None:
                batch_map = o_map
            else:
                batch_map = np.vstack((o_map,batch_map))
        return batch_map

    def get_big_map(self,center_x,center_y,w,h,x_map,batch):
        batch_map = None
        for i in range(batch):
            cx = center_x[i]
            cy = center_y[i]
            dw = w[i]
            dh = h[i]
            l = abs(cx - dw)
            r = abs(cx + dw)
            up = abs(cy - dh)
            down = abs(cy + dh)
            if l > 1:
                l = 1
            if r > 1:
                r = 1
            if up > 1:
                up = 1
            if down > 1:
                down = 1
            o_map = x_map[i,:,min(l,r):max(l,r),min(up,down):max(up,down)]
            o_map = np.expand_dims(o_map,axis=0)
            if batch_map is None:
                batch_map = o_map
            else:
                batch_map = np.vstack((o_map,batch_map))
        return batch_map

    def do(self,input,is_train=False,target=None):
        res_1,res_2,res_3,res_map = self.resnet18(input)
        x1 = res_1.cpu().detach().numpy()
        x2 = res_2.cpu().detach().numpy()
        x3 = res_3.cpu().detach().numpy()
        x_map = res_map.cpu().detach().numpy()
        batch = len(x1)

        map1 = self.get_map(0,x1,x_map,batch)
        map1_tensor = torch.from_numpy(map1)
        if is_cuda and args.gpu is not None:
            map1_tensor = map1_tensor.cuda(args.gpu, non_blocking=True)
        p1 = self.m1(map1_tensor)

        map2 = self.get_map(2,x1,x_map,batch)
        map2_tensor = torch.from_numpy(map2)
        if is_cuda and args.gpu is not None:
            map2_tensor = map2_tensor.cuda(args.gpu, non_blocking=True)
        p2 = self.m2(map2_tensor)

        map3 = self.get_map(4,x1,x_map,batch)
        map3_tensor = torch.from_numpy(map3)
        if is_cuda and args.gpu is not None:
            map3_tensor = map3_tensor.cuda(args.gpu, non_blocking=True)
        p3 = self.m3(map3_tensor)

        map4 = self.get_map(6,x1,x_map,batch)
        map4_tensor = torch.from_numpy(map4)
        if is_cuda and args.gpu is not None:
            map4_tensor = map4_tensor.cuda(args.gpu, non_blocking=True)
        p4 = self.m4(map4_tensor)

        center_x = x2[:,0]
        center_y = x2[:,1]
        w = np.exp(x3[:,0])
        h = np.exp(x3[:,1])
        map5 = self.get_big_map(center_x,center_y,w,h,x_map,batch)
        map5_tensor = torch.from_numpy(map5)
        if is_cuda and args.gpu is not None:
            map5_tensor = map5_tensor.cuda(args.gpu, non_blocking=True)
        map6_tensor = adaptive_max_pool(map5_tensor,(4,4))
        p5 = self.m5(map6_tensor)

        if is_train:
            x_list = []
            y_list = []
            #todo
            loss_resnet = self.criterion(x1,target[:,0:8]) + self.criterion(x2,target[:,8:10])
            self.loss = loss_resnet + loss_m1 + loss_m2 + loss_m3 + loss_m4 + loss_m5


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    normalize,
    ])

trainset = HandDataset(args.trainjson,transform = transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,shuffle=True, num_workers=4)
testset = HandDataset(args.testjson,transform = transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,shuffle=True, num_workers=4)

is_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if is_cuda and args.gpu is not None:
    torch.cuda.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = True

model = resnet18()
m1 = HandNet()
m2 = HandNet()
m3 = HandNet()
m4 = HandNet()
m5 = HandNet()

print("loaded model!")

if is_cuda and args.gpu is not None:
    model = model.cuda(args.gpu)
    m1 = m1.cuda(args.gpu)
    m2 = m2.cuda(args.gpu)
    m3 = m3.cuda(args.gpu)
    m4 = m4.cuda(args.gpu)
    m5 = m5.cuda(args.gpu)
    print("model to gpu")
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}'".format(args.checkpoint))
if os.path.isfile(args.m1):
    chk_m1 = torch.load(args.m1)
    m1.load_state_dict(chk_m1['state_dict'])
    print("loaded checkpoint m1 '{}'".format(args.m1))
if os.path.isfile(args.m2):
    chk_m2 = torch.load(args.m2)
    m2.load_state_dict(chk_m2['state_dict'])
    print("loaded checkpoint m2 '{}'".format(args.m2))
if os.path.isfile(args.m3):
    chk_m3 = torch.load(args.m3)
    m3.load_state_dict(chk_m3['state_dict'])
    print("loaded checkpoint m3 '{}'".format(args.m3))
if os.path.isfile(args.m4):
    chk_m4 = torch.load(args.m4)
    m4.load_state_dict(chk_m4['state_dict'])
    print("loaded checkpoint m4 '{}'".format(args.m4))
if os.path.isfile(args.m5):
    chk_m5 = torch.load(args.m5)
    m5.load_state_dict(chk_m5['state_dict'])
    print("loaded checkpoint m5 '{}'".format(args.m5))

criterion_mse = nn.MSELoss()
criterion_L1 = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

loss_train = []
loss_test = []

def train(trainloader, model, criterion_mse, criterion_L1,optimizer, epoch):
    batchtimes = AVGMeter()
    losses = AVGMeter()
    epochtime = time.time()
    batchtime = time.time()
    model.train()
    for i, (input, target) in enumerate(trainloader):
        if is_cuda and args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        output = model(input)
        loss = criterion_mse(output, target)+criterion_L1(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        batchtimes.update(time.time() - batchtime)

        global loss_train
        loss_train.append(loss.item())

        if i % 5 == 0:
            print('epoch: [{0}][{1}/{2}]\t'
                'batchtime {batchtimes.val:.3f} ({batchtimes.avg:.3f})\t'
                'losses {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(trainloader), batchtimes=batchtimes,loss=losses))
        batchtime = time.time()
    print("train epoch {} time is {}".format(epoch,time.time()-epochtime))
    return losses.avg

def test(testloader, model, criterion_mse, criterion_L1, epoch):
    batchtimes = AVGMeter()
    losses = AVGMeter()
    epochtime = time.time()
    batchtime = time.time()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            if is_cuda and args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(input)
            loss = criterion_mse(output, target)+criterion_L1(output, target)

            losses.update(loss.item(), input.size(0))
            batchtimes.update(time.time() - batchtime)

            global loss_test
            loss_test.append(loss.item())

            if i % 5 == 0:
                print('epoch: [{0}][{1}/{2}]\t'
                    'batchtime {batchtimes.val:.3f} ({batchtimes.avg:.3f})\t'
                    'losses {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(testloader), batchtimes=batchtimes,loss=losses))
            batchtime = time.time()
    print("test epoch {} time is {}".format(epoch,time.time()-epochtime))
    return losses.avg

def main():
    loss_sub = 1000
    loss = None
    
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train(trainloader, model, criterion_mse, criterion_L1, optimizer, epoch)
        test_loss = test(testloader, model, criterion_mse, criterion_L1, epoch)
        torch.save({'epoch': epoch + 1,'state_dict': model.state_dict()}, 'checkpoint_{}.pth'.format(epoch))
        if loss == None:
            loss = train_loss
        else:
            if loss_sub > abs(train_loss - test_loss) and loss > train_loss:
                loss = train_loss
                loss_sub = abs(train_loss - test_loss)
                shutil.copyfile('checkpoint_{}.pth'.format(epoch), 'hand_model.pth')
    write_ls_txt("loss_train.log",loss_train)
    write_ls_txt("loss_test.log",loss_test)

if __name__ == '__main__':
    main()
