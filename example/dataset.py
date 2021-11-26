# import libraries
import torch, torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
import torch.nn as nn
import numpy as np
import csv
import inspect
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support


class ImageDataset(Dataset):
    '''
    A Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        train_loader, test_loader = self.load_data(self.dataset)

        return (train_loader, test_loader)
    
    def load_data(self):
        BATCH_SIZE_TRAIN = 100
        BATCH_SIZE_TEST = 64

        dataset = self.dataset
        train_loader = None
        test_loader = None

        print("loading dataset ....")

        '''
        Return MNIST train/test data and labels as numpy arrays
        '''
        if dataset == 'MNIST':
            # Loading MNIST using torchvision.datasets
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=BATCH_SIZE_TEST, shuffle=False)

        elif dataset == 'CIFAR':
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])),
                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

            # Normalize the test set same as training set without augmentation
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])),
                batch_size=BATCH_SIZE_TEST, shuffle=False)

        return train_loader, test_loader


    def test_model(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_losses = []

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]  # get class from network's prediction
                correct += pred.eq(target.data.view_as(pred)).sum()
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # accuracy = 100. * correct / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        accuracy = float(accuracy)

        return accuracy

    
    def test_model_detail (self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_losses = []
        target_true = 0
        predicted_true = 0
        correct_true = 0
        precision = 0
        recall = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]   # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum()

                precision+=precision_score(target.data.view_as(pred).cpu(),pred.cpu(), average='macro')
                recall+=recall_score(target.data.view_as(pred).cpu(),pred.cpu(), average='macro')

                # modifications for precision and recall
                predicted_classes = torch.argmax(output, dim=1) == 0
                
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        precision /= len(test_loader.dataset)

        accuracy = correct / len(test_loader.dataset)
        accuracy = float(accuracy)

        return accuracy, precision, recall


    def test_precision(self, model, test_loader):

        pred = []
        true = []
        sm = nn.Softmax(dim = 1)
        with torch.no_grad():
            model.eval()

            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                output = sm(output)
                _, preds = torch.max(output, 1)
                preds = preds.cpu().numpy()
                target = target.cpu().numpy()
                preds = np.reshape(preds,(len(preds),1))
                target = np.reshape(target,(len(preds),1))

                for i in range(len(preds)):
                    pred.append(preds[i])
                    true.append(target[i])

        precision = precision_score(true,pred,average='macro')
        recall = recall_score(true,pred,average='macro')
        accuracy = accuracy_score(true,pred)
        return accuracy, precision, recall

    
    def test_validations(self, model, test_loader):

        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        precision, recall = 0, 0
        
        # set model to evaluating (testing)
        y_pred_list = []
        target_list = []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                y_test_pred = model(data)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                target_list.append(target)

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        precision = precision_score(target_list, y_pred_list, average='macro')
        recall = recall_score(target_list, y_pred_list, average='macro')

        return precision, recall


    def train_model(self, model, train_loader, learning_rate = 0.01, momentum = 0.5, epochs = 1, iteration=1):
        '''
        Update the client model on client data
        '''
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # define optimizer

        # unfreeze layers
        for param in model.parameters():
            param.requires_grad = True


        # train model
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                print('Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
            print("Training completed for local model for iteration {}, epoch {}".format(iteration, epoch))



    def train_attack(self, classifer, train_loader, learning_rate = 0.01, momentum = 0.5, epochs = 1, iteration=1):
        '''
        Update the client model on client data
        '''
        optimizer = optim.SGD(classifer.parameters(), lr=learning_rate, momentum=momentum) # define optimizer
        
        # freeze layers
        for param in classifer.parameters():
            param.requires_grad = False

        # train model
        classifer.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = classifer(data)
                loss = F.nll_loss(output, target)
                # loss.backward()
                # optimizer.step()
                print('Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
            print("Training completed for local model for iteration {}, epoch {}".format(iteration, epoch))


    def calculate_metric(self, metric_fn, true_y, pred_y):
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y, average="macro")
        
    def print_scores(self, p, r, f1, a, batch_size):
        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

    



class CSVWriter():

    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename