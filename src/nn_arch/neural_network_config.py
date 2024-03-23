"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import torch
import pytorch_lightning as pl

from config import Config

config = Config() # create an object of class Config

class CustomModule(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, num_classes) -> None:
        super().__init__() # call constructor of parent class
        '''
        It ensures that the initialization logic of the parent class is executed before the initialization logic of the current class.
        '''

        self.model = model # assign model
        self.criterion = criterion # assign criterion
        self.optimizer = optimizer # assign optimizer
        self.num_classes = num_classes # assign total number of categories

    def forward(self, x):
        return self.model(x) # return output represents the results of the forward pass of the neural network
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        # compute loss
        loss = self.criterion(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_LOSS, loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # log the loss for visualization

        # compute accuracy
        acc = self.accuracy(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_ACC  , acc, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # log the accuracy for visualization

        # calculate and log additional metrics
        precision, recall, f1, tp, fp, fn, tn = self.calculate_metrics(outputs,labels)
        self.log(config.LOG_NAME_TRAIN_PRECISION, precision, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_RECALL, recall, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_F1, f1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_TP, tp, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_FP, fp, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_FN, fn, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_TRAIN_TN, tn, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        # compute loss
        loss = self.criterion(outputs,labels)
        self.log(config.LOG_NAME_VALID_LOSS, loss, on_step=True, on_epoch=True, prog_bar=True) # log the loss for visualization

        # compute accuracy
        acc = self.accuracy(outputs,labels)
        self.log(config.LOG_NAME_VALID_ACC, acc, on_step=True, on_epoch=True, prog_bar=True) # log the accuracy for visualization

        # calculate and log additional metrics
        precision, recall, f1, tp, fp, fn, tn = self.calculate_metrics(outputs,labels)
        self.log(config.LOG_NAME_VALID_PRECISION, precision, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_RECALL, recall, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_F1, f1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_TP, tp, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_FP, fp, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_FN, fn, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)
        self.log(config.LOG_NAME_VALID_TN, tn, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer

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