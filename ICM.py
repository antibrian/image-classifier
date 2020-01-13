import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import time
from PIL import Image
import json
import os

class ICM: 
    def __init__(self, network = "resnet50", inputs = 2048, outputs = 102, learning_rate = 0.001, hidden_units = 2, gpu = False):

        self.set_gpu(gpu)
        self.epochs = 0
        self.network = network
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_units = 2
        self.learning_rate = learning_rate
        
    def build_network(self, network = None, inputs = None, outputs = None, learning_rate = None):
        ''' Initializes network for training and predictions according to instance properties 
        '''
        
        network = self.network if network is None else network
        inputs = self.inputs if inputs is None else inputs
        outputs = self.outputs if outputs is None else outputs
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        
        self.model = getattr(models, network)(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(inputs, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, outputs),
                                        nn.LogSoftmax(dim=1))
        
        self.model.fc = self.classifier
        
        if(hasattr(self, 'class_to_idx')):
            self.__configure_idx()
            
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
    
    def __configure_transforms(self):
        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])

    def __configure_idx(self):
        ''' Configures classifier indexes, and invert them for easy lookup
        '''
#         self.model.class_to_idx = self.train_data.class_to_idx
        self.model.class_to_idx = self.class_to_idx
        self.idx_to_class = {value: key for key, value in self.model.class_to_idx.items()}
        
    def set_category_names(self, json_path):
        ''' Load category names from provided JSON file
        '''
        with open(json_path, 'r') as f:
            self.cat_to_name = json.load(f)
    
    def set_gpu(self, gpu):
        ''' Configures CUDA or CPU device for instance
        '''
        cuda = torch.cuda.is_available()
        self.gpu = gpu and cuda
        self.device = torch.device("cuda" if self.gpu  else "cpu")

        print("GPU Enabled" if self.gpu else "Warning: CUDA Unavailable" if gpu and not cuda else "GPU Disabled")

    def load_data(self, data_dir):
        ''' Loads and transforms image data into instance 
        '''
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        self.__configure_transforms()
        
        self.train_data = datasets.ImageFolder(train_dir, transform=self.train_transforms)
        self.validation_data = datasets.ImageFolder(valid_dir, transform=self.test_transforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=self.test_transforms)

        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.validationloader = torch.utils.data.DataLoader(self.validation_data, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=64)
        
        self.class_to_idx = self.train_data.class_to_idx;
        if(hasattr(self, 'model')):
            self.__configure_idx()
        
    def train(self, epochs):
        ''' Trains model for provided epochs, printing error and accuracy to screen 
        '''
        
        running_loss = 0

        start = time.time()
        print('Training started')
        for epoch in range(epochs):

            for images, labels in self.trainloader:

                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                log_ps = self.model.forward(images)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            else:
                validation_loss = 0
                accuracy = 0
                self.model.eval()

                with torch.no_grad():
                    for images, labels in self.validationloader:
                        images, labels = images.to(self.device), labels.to(self.device)

                        log_ps = self.model.forward(images)
                        loss += self.criterion(log_ps, labels)
                        validation_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                time_elapsed = time.time() - start
                print("Epoch: {}/{}.. ".format(self.epochs + epoch + 1 , self.epochs + epochs),
                     "Training Loss: {:.3f}.. ".format(running_loss/len(self.trainloader)),
                     "Validation Loss: {:.3f}.. ".format(validation_loss/len(self.validationloader)),
                     "Validation Accuracy: {:.3f}.. ".format(accuracy/len(self.validationloader) * 100),
                     "Elapsed time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

                running_loss = 0
                self.model.train()

        self.epochs += epochs
        print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    def test(self):
        ''' Prints testing results to screen
        '''
        testing_loss = 0
        accuracy = 0
        self.model.eval()

        with torch.no_grad():
            for images, labels in self.testloader:

                images, labels = images.to(self.device), labels.to(self.device)

                log_ps = self.model.forward(images)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Testing Accuracy: {:.3f}".format(accuracy/len(self.testloader) * 100))

        self.model.train();

    def checkpoint(self, save_dir = None):
        ''' Stores checkpoint with instance parameters 
        '''
        
        save_dir_path = ''
        
        if save_dir is not None:
            save_dir_path = save_dir + '/'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            
        file_path = save_dir_path + 'checkpoint.pth'
        
        self.model.class_to_idx = self.train_data.class_to_idx

        checkpoint = {'network': self.network,
                      'inputs': self.inputs,
                      'outputs': self.outputs,
                      'learning_rate': self.learning_rate,
                      'hidden_units': self.hidden_units,
                      'epochs': self.epochs,
                      'classifier': self.model.fc,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'class_to_idx': self.model.class_to_idx}

        torch.save(checkpoint, file_path)
        
    def load_checkpoint(self, filepath = './checkpoint.pth'):
        ''' Configures network for training and predictions according to checkpoint properties
        '''
        
        checkpoint = torch.load(filepath, map_location=None if self.gpu and torch.cuda.is_available() else "cpu")

        self.network = checkpoint['network']
        self.inputs = checkpoint['inputs']
        self.outputs = checkpoint['outputs']
        self.learning_rate = checkpoint['learning_rate']
        self.hidden_units = checkpoint['hidden_units']
        self.epochs = checkpoint['epochs']
        self.classifier = checkpoint['classifier']

        self.class_to_idx = checkpoint['class_to_idx']
        
        self.model = getattr(models, self.network)()
        self.model.fc = self.classifier
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.__configure_idx()
        
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.criterion = nn.NLLLoss()
            
    def __process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a Tensor
        '''
        
        self.__configure_transforms()
        
        im = Image.open(image)

        return self.test_transforms(im).to(self.device)

    def predict(self, image_path, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
            returns numpy array, numpy array
        '''

        image = self.__process_image(image_path)

        self.model.eval()
        with torch.no_grad():
            image.unsqueeze_(0)
            log_ps = self.model.forward(image)
            ps = torch.exp(log_ps)
            probs, classes = ps.topk(topk, dim=1)           

            return probs.cpu().numpy()[0], list(map(lambda c: self.idx_to_class.get(c), classes.cpu().numpy()[0])), list(map(lambda c: self.cat_to_name[self.idx_to_class.get(c)], classes.cpu().numpy()[0])) if hasattr(self, 'cat_to_name') else np.empty(topk, dtype="S0")
        
        
        
    
