from net_inference import VGG as Int8Net
from vgg16 import VGG
from torchvision import transforms
import torch
from dataset import TestDataset
from torch.utils.data import DataLoader
from PIL import Image

'''test int8 acc
   compare to fp32
'''


def test(testmodel):
    device = 'cuda'
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = testmodel(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

if __name__ == "__main__":

    accbefore=torch.load('./checkpoint/ckpt.pth')['acc']

    int8model=Int8Net()
    int8model.load_state_dict(torch.load('./checkpoint/int8.pth'))
    int8net=int8model.cuda()

    test_augmentation = transforms.Compose([
        transforms.Resize((112, 112), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_cifar10 = TestDataset(transform=test_augmentation)
    test_loader = DataLoader(dataset=test_cifar10, batch_size=64, shuffle=False)


    int8net.eval()
    accint8=test(int8net)

    print("before int8: ",accbefore)
    print("after int8: ",accint8)


    '''
    test part
    for image,label in test_loader:
        out = net(image)
        print(out)
        print(out.shape)

        int8out = int8net(image)
        print(int8out)
        print(int8out.shape)
        break'''



