
import paddle
from paddle.metric import Accuracy
from paddle.nn import CrossEntropyLoss
from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import RandomHorizontalFlip, Compose, RandomCrop, Normalize

import math
import time
import logging
import argparse
import numpy as np
import wide_resnet

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def config():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--epoch', default=200, type=int, help='epoch of model')
    parser.add_argument('--batchsize', default=128, type=int, help='epoch of model')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    return parser.parse_args()


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)


class ToArray(object):
    """Convert a ``PIL.Image`` to ``numpy.ndarray``.
    Converts a PIL.Image or numpy.ndarray (H x W x C) to a paddle.Tensor of shape (C x H x W).
    If input is a grayscale image (H x W), it will be converted to a image of shape (H x W x 1). 
    And the shape of output tensor will be (1 x H x W).
    If you want to keep the shape of output tensor as (H x W x C), you can set data_format = ``HWC`` .
    Converts a PIL.Image or numpy.ndarray in the range [0, 255] to a paddle.Tensor in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, 
    RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.
    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


def build_transform():
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465] 
    CIFAR_STD = [0.2023, 0.1994, 0.2010]  
    train_transforms = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transforms = Compose([ToArray(), Normalize(CIFAR_MEAN, CIFAR_STD)])
    return train_transforms, test_transforms


# Training
def train(epoch,model,train_loader,criterion,cfg):
    
    epoch_loss = 0
    epoch_acc = 0
    metric = Accuracy()
    model.train()

    opt = paddle.optimizer.SGD(learning_rate=learning_rate(cfg.lr, epoch), parameters = net.parameters())

    for batch_id,(img, label) in enumerate(train_loader):

        logits = model(img)
        
        loss = criterion(logits, label)
        acc = metric.update(metric.compute(logits, label))

        if batch_id % 10 == 0:
            logger.info("epoch: {}, batch_id: {}, train_loss: {}, train_acc: {}".format(epoch, batch_id, loss.item(),acc))

        loss.backward()
        opt.step()
        opt.clear_grad()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def test(epoch,model,val_loader,criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    metric = Accuracy()
    for batch_id,(img, label) in enumerate(val_loader):
        
        logits = model(img)
        loss = criterion(logits, label)
        acc = metric.update(metric.compute(logits, label))

        if batch_id % 10 == 0:
            logger.info("epoch: {}, batch_id: {}, val_loss: {}, val_acc: {}".format(epoch, batch_id, loss.item(),acc))

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(val_loader), epoch_acc / len(val_loader)




if __name__ == '__main__':

    #加载参数
    cfg = config()

    #加载数据
    train_transforms,val_transforms = build_transform()
    train_set = Cifar10(mode='train', transform=train_transforms,download=True)
    test_set = Cifar10(mode='test', transform=val_transforms)
    train_loader = paddle.io.DataLoader(train_set,batch_size=cfg.batchsize,num_workers=2,return_list=True)
    val_loader = paddle.io.DataLoader(test_set,batch_size=cfg.batchsize)

    #定义模型
    net = wide_resnet.Wide_ResNet(depth=cfg.depth, widen_factor=cfg.widen_factor, dropout_rate=cfg.dropout,num_classes=10)
    criterion = CrossEntropyLoss()

    # 训练
    best_acc = 0
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    for epoch in range(cfg.epoch):
        start_time = time.time()

        train_loss, train_acc  = train(epoch,net,train_loader,criterion,cfg)
        valid_loss, valid_acc  = test(epoch,net,val_loader,criterion)

        if best_acc < valid_acc:
            best_acc = valid_acc

        logger.info(f'Epoch: {epoch:02}, Best Acc: {best_acc * 100:.2f}%')
