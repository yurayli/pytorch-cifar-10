## Import library
import os, sys, argparse, json
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from models import *

## File path
filepath = "/train_files/"
model_path = "/models/"
output_path = "/output/"

## Settings
parser = argparse.ArgumentParser(description='CIFAR-10 classifier')
parser.add_argument('--model', default='vgg')
parser.add_argument('--optim', default='adam')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.1, type=float)
parser.add_argument('--step_size', default=20, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_train', default=49000, type=int)
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='cifar_best.pth.tar')
args = parser.parse_args()

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

preload = True  # pre-loading trained model or not


## Functions

def prepare_loader(path=filepath, batch_size=args.batch_size, num_train=args.num_train):
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)
    transform_train = T.Compose([
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                    T.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std)
                ])
    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std)
                ])

    cifar10_train = dset.CIFAR10(path, train=True, download=True, transform=transform_train)
    loader_train = DataLoader(cifar10_train, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(num_train)))

    cifar10_val = dset.CIFAR10(path, train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

    cifar10_test = dset.CIFAR10(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)
    return loader_train, loader_val, loader_test


# solver of model
class NetSolver(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_acc = 0
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), output_path+args.checkpoint_name)
        checkpoint = {
            'model': args.model,
            'optimizer': args.optim,
            'scheduler': str(type(self.scheduler)),
            'lr_init': args.lr,
            'lr_decay': args.lr_decay,
            'step_size': args.step_size,
            'batch_size': args.batch_size,
            'epoch': epoch,
        }
        with open(output_path+'hyper_param_optim.json', 'w') as f:
            json.dump(checkpoint, f)

    def train(self, train_loader, val_loader, epochs):
        self.model = self.model.to(device=device)  # move the model parameters to device (CPU/GPU)

        # Log the initial performance
        x, y = next(iter(train_loader))
        n_correct, n_samples, bch_loss = self.foward_pass(x, y)
        val_acc, val_loss = self.check_accuracy(val_loader)
        self.loss_history, self.val_loss_history = [bch_loss], [val_loss]
        self.train_acc_history, self.val_acc_history = [float(n_correct)/n_samples], [val_acc]
        self.best_val_acc = val_acc

        # Start training for epochs
        for e in range(epochs):
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            self.model.train()  # set the model in "training" mode
            self.scheduler.step()  # update the scheduler
            for t, (x, y) in enumerate(train_loader):
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)

                scores = self.model(x)
                loss = F.cross_entropy(scores, y)
                if (t + 1) % args.print_every == 0:
                    print('t = %d, loss = %.4f' % (t+1, loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Checkpoint and record/print metrics at epoch end
            _, preds = scores.max(1)
            bch_acc = float((preds == y).sum()) / preds.size(0)
            bch_loss = loss.item()
            val_acc, val_loss = self.check_accuracy(val_loader)

            self.loss_history.append(bch_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(bch_acc)
            self.val_acc_history.append(val_acc)
            # for floydhub metric graphs
            print('{"metric": "Acc.", "value": %.4f}' % (bch_acc,))
            print('{"metric": "Val. Acc.", "value": %.4f}' % (val_acc,))
            print('{"metric": "Loss", "value": %.4f}' % (bch_loss,))
            print('{"metric": "Val. Loss", "value": %.4f}' % (val_loss,))

            if val_acc > self.best_val_acc:
                print('Saving model...')
                self.best_val_acc = val_acc
                self._save_checkpoint(e)
            print()

    def foward_pass(self, x, y):
        self.model.eval()  # set model to "evaluation" mode
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = self.model(x)
        l = F.cross_entropy(scores, y)
        _, preds = scores.max(1)
        n_correct = (preds == y).sum()
        n_samples = preds.size(0)
        loss = l.item()
        return n_correct, n_samples, loss

    def check_accuracy(self, loader):
        if loader.dataset.train:
            print('Checking the validation set...')
        else:
            print('Checking the test set...')
        num_correct, num_samples, losses = 0, 0, []
        with torch.no_grad():
            for x, y in loader:
                n_corr, n_samples, l = self.foward_pass(x, y)
                num_correct += n_corr
                num_samples += n_samples
                losses.append(l)
        acc = float(num_correct) / num_samples
        loss = np.mean(losses)
        print('Got loss %.4f, and %d / %d correct (%.2f)' % (loss, num_correct, num_samples, 100 * acc))
        return acc, loss


def reload_best(model, path):
    model.load_state_dict(torch.load(path+args.checkpoint_name))


# plot log of loss and acc during training
def plot_history(history, fname):
    accs, val_accs, losses, val_losses = history
    epochs = range(len(accs))
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, accs, '-o')
    ax1.plot(epochs, val_accs, '-o')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'val'], loc='lower right')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses, '-o')
    ax2.plot(epochs, val_losses, '-o')
    ax2.set_ylim(bottom=-0.1)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'val'], loc='upper right')
    fig.savefig(fname)


def main():
    loader_train, loader_val, loader_test = prepare_loader()

    if args.model == 'vgg':
        model = VGG_like(num_fmaps_base=64)
    elif args.model == 'resnet':
        model = ResNet(Bottleneck, [2, 3, 5, 2])
    if preload:
        reload_best(model, model_path)
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    solver = NetSolver(model, optimizer, scheduler)

    t0 = time()
    solver.train(loader_train, loader_val, epochs=args.epochs)
    print('Training goes with %.2f seconds.' %(time()-t0))
    history = [solver.train_acc_history, solver.val_acc_history, solver.loss_history, solver.val_loss_history]
    plot_history(history, output_path+'%s_%s_%d_curve.png' %(args.model, args.optim, args.epochs))

    reload_best(model, output_path)
    print()
    test_acc, test_loss = solver.check_accuracy(loader_test)


if __name__ == "__main__":
    main()
