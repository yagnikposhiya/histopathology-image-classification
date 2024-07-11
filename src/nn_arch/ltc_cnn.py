"""
author: Dr. Sanskruti Patel, Yagnik Poshiya, Rupesh Garsondiya
github: @yagnikposhiya, @Rupeshgarsondiya
organization: Charotar University of Science and Technology
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

from config import Config

config = Config() # create config  object of Config class

class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeConstantCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_rec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_rec, a=math.sqrt(5))
        nn.init.zeros_(self.b)
        nn.init.ones_(self.tau)

    def forward(self, x, h):
        S = torch.tanh(torch.mm(x, self.W_in) + torch.mm(h, self.W_rec) + self.b)
        tau = self.tau
        h_new = h + (S - h) / tau
        return h_new

class CNN_LTC(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super(CNN_LTC, self).__init__()
        self.lr = learning_rate # set learning rate
        self.save_hyperparameters() # used to automatically save all the hyperparams passed to the '__init__' method of LightningModule
        """
        'save_hyperparameters()' function is available in the PyTorch Lightning module.
        access saved hyperparams using: self.hparams.<arg_name>
        i.e. 
            [1] self.hparams.num_classes
            [2] self.hparams.learning_rate
        """

        # Define CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after CNN layers
        self.feature_size = 64 * 128 * 128  # Adjust according to the input image size

        # Set hidden size automatically
        self.hidden_size = self.feature_size // 2  # Or any other heuristic
        
        # Define LTC layers
        self.ltc_cell = LiquidTimeConstantCell(self.feature_size, self.hidden_size)
        
        # Define fully connected layer
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Initialize hidden state
        h = torch.zeros(x.size(0), self.ltc_cell.hidden_size, device=x.device)
        
        # Pass the features through LTC cell
        h = self.ltc_cell(x, h)
        
        # Classification
        out = self.fc(h)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        # compute loss
        loss = F.cross_entropy(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_LOSS, loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # log the loss for visualization

        # compute accuracy
        acc = self.accuracy(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_ACC  , acc, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # log the accuracy for visualization

        # calculate and log additional metrics
        precision, recall, f1, tp, fp, fn, tn = self.calculate_metrics(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_PRECISION, precision, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean precisoin across all classes
        self.log(config.LOG_NAME_TRAIN_RECALL, recall, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean recall across all classes
        self.log(config.LOG_NAME_TRAIN_F1, f1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean f1-score across classes
        self.log(config.LOG_NAME_TRAIN_TP, tp.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # true positives across all classes
        self.log(config.LOG_NAME_TRAIN_FP, fp.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # false positives across all classes
        self.log(config.LOG_NAME_TRAIN_FN, fn.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # false negatives across all classes
        self.log(config.LOG_NAME_TRAIN_TN, tn.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # true negatives across all classes

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        # compute loss
        loss = F.cross_entropy(outputs,labels)
        self.log(config.LOG_NAME_VALID_LOSS, loss, on_step=True, on_epoch=True, prog_bar=True) # log the loss for visualization

        # compute accuracy
        acc = self.accuracy(outputs,labels)
        self.log(config.LOG_NAME_VALID_ACC, acc, on_step=True, on_epoch=True, prog_bar=True) # log the accuracy for visualization

        # calculate and log additional metrics
        precision, recall, f1, tp, fp, fn, tn = self.calculate_metrics(outputs,labels)
        self.log(config.LOG_NAME_VALID_PRECISION, precision, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean precision across all classes
        self.log(config.LOG_NAME_VALID_RECALL, recall, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean recall across all classes
        self.log(config.LOG_NAME_VALID_F1, f1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # mean f1-score across all classes
        self.log(config.LOG_NAME_VALID_TP, tp.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # true positives across all classes
        self.log(config.LOG_NAME_VALID_FP, fp.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # false positives across all classes
        self.log(config.LOG_NAME_VALID_FN, fn.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # false negatives across all classes
        self.log(config.LOG_NAME_VALID_TN, tn.sum(), on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # true negatives across classes

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = (predictions == labels).sum().item()
        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def calculate_metrics(self,logits,labels):

        predictions = torch.argmax(logits, dim=1)
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        # Calculate precision, recall, and F1-score for each class
        tp = confusion_matrix.diag()
        fp = confusion_matrix.sum(dim=0) - tp
        fn = confusion_matrix.sum(dim=1) - tp
        tn = confusion_matrix.sum() - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Average precision, recall, and F1-score across all classes
        avg_precision = precision.mean()
        avg_recall = recall.mean()
        avg_f1_score = f1_score.mean()

        return avg_precision, avg_recall, avg_f1_score, tp, fp, fn, tn