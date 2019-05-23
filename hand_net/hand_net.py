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
from PIL import Image, ImageDraw
from resnet import resnet18


parser = argparse.ArgumentParser(description='HandNet')
parser.add_argument('--trainjson', default='data/train.json', type=str, metavar='PATH',help='json file path')
parser.add_argument('--testjson', default='data/test.json', type=str, metavar='PATH',help='json file path')
parser.add_argument('--batchsize', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',help='path to checkpoint')
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
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

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
    def __init__(self, ngpu):
        super(HandNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )
        self.fc = nn.Linear(289, 62)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 289)
        output = self.fc(output)
        output = self.sig(output)
        return output

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

torch.manual_seed(1)
if args.gpu is not None:
    torch.cuda.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = True

model = resnet18()
print("loaded model!")

if args.gpu is not None:
    model = model.cuda(args.gpu)
    print("model to gpu")
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}'".format(args.checkpoint))

criterion = nn.MSELoss().cuda(args.gpu)
optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

loss_train = []
loss_test = []

def train(trainloader, model, criterion, optimizer, epoch):
    batchtimes = AVGMeter()
    losses = AVGMeter()
    epochtime = time.time()
    batchtime = time.time()
    model.train()
    for i, (input, target) in enumerate(trainloader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
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

def test(testloader, model, criterion, epoch):
    batchtimes = AVGMeter()
    losses = AVGMeter()
    epochtime = time.time()
    batchtime = time.time()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

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
        train_loss = train(trainloader, model, criterion, optimizer, epoch)
        test_loss = test(testloader, model, criterion, epoch)
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