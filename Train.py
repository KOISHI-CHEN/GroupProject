import argparse
import os
import torch

from torch import nn
from torchvision import datasets, transforms

from Network import SegmentationNetwork

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--last-checkpoint', type=str, default=None)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

args = parser.parse_args()

train_batch_size = 1
test_batch_size = 1
learning_rate = 0.1
num_epoches = 30
lr = 0.015
momentum = 0.5


def train(model, train_loader, test_loader, optimizer, loss_fn, device):
    ###################### Begin #########################

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()

        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            some, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        model.eval()
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_fn(out, label)
            eval_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))

        print('epoch: {}, Train_loss: {:.4f}, Train Acc: {:.4f},Test Loss: {:,.4f}, Test Acc:{:.4f}'.format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            train_acc / len(
                                                                                                                train_loader),
                                                                                                            eval_loss / len(
                                                                                                                test_loader),
                                                                                                            eval_acc / len(
                                                                                                                test_loader)))
        torch.save(model.state_dict(), args.checkpoint_dir + f'epoch-{epoch}.pth')

    ######################  End  #########################


if __name__ == '__main__':
    ###################### Begin #########################
    # You can run your train() here
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

    model = SegmentationNetwork().to(device=args.device)
    if args.last_checkpoint is not None:
        model.load_state_dict(torch.load(args.last_checkpoint, map_location=args.device))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(model, train_loader, test_loader, optimizer, loss_fn, args.device)

    ######################  End  #########################
