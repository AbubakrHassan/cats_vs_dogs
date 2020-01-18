import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from dataset import CvDDataset
from model import CNN, CNN4Layers
import argparse


def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


input_size = 224 * 244 * 3  # images are 224*224*3 pixels
output_size = 2  # there are 2 classes

train_loader = torch.utils.data.DataLoader(
    CvDDataset('./dogs-vs-cats/train/',
               transform=transforms.Compose([
                   transforms.ToPILImage(),
                   transforms.Resize((224, 224)),
                   transforms.ToTensor(),

               ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    CvDDataset('./dogs-vs-cats/test/',
               transform=transforms.Compose([
                   transforms.ToPILImage(),
                   transforms.Resize((224, 224)),
                   transforms.ToTensor(),

               ])),
    batch_size=1000, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

accuracy_list = []

# Training settings
n_features = 6  # number of feature maps

parser = argparse.ArgumentParser(description='Train and test one of two CNN architectures')
parser.add_argument('--model', type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()
if args.model=="CNN4Layers":
    model_cnn = CNN4Layers(input_size, n_features, output_size)
else:
    model_cnn = CNN(input_size, n_features, output_size)

optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))


def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)

        test_loss += F.nll_loss(output, target,
                                reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


if __name__ == "__main__":


    for epoch in range(0, 1):
        train(epoch, model_cnn)
        test(model_cnn)
