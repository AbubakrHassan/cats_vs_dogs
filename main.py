import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from dataset import CvDDataset
from model import CNN, CNN4Layers
import argparse
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log_dir/experiment1')
import matplotlib.pyplot as plt


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
n_features = 25  # number of feature maps

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


## log graph for tensorboard

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
writer.add_graph(model_cnn, images)

classes = ("Cat", "Dog")

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def train(epoch, model):
    model.train()
    running_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            writer.add_scalar('training loss',
                              loss.item(),
                              epoch * len(train_loader) + batch_idx)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, data, labels),
                              global_step=epoch * len(train_loader) + batch_idx)
            running_loss = 0.0
        writer.close()


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


    for epoch in range(0, 10):
        train(epoch, model_cnn)
        test(model_cnn)
