import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from dataset import TrainDataset,TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from vgg16 import VGG


"Simple Implementation of Training cifar10 with vgg-mini"

if __name__ == "__main__":
    train_augmentation=transforms.Compose([
          transforms.Resize((112,112),interpolation=Image.BICUBIC),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
          ])
    train_cifar10 = TrainDataset(transform=train_augmentation)
    test_cifar10 = TestDataset(transform=train_augmentation)

    train_loader = DataLoader(dataset=train_cifar10,batch_size=128,shuffle=True)
    test_loader = DataLoader(dataset=test_cifar10,batch_size=128,shuffle=False)
    model=VGG()
    print(model)
    model=model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device="cuda"
    best_acc = 0

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


        # Save checkpoint.
        acc = 100. * correct / total
        print("acc: ",acc)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

    for epoch in range(0,200):
        train(epoch)
        test(epoch)