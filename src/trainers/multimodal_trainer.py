import torch
from tqdm import tqdm
import os
import numpy as np
import logging
from src.trainers.callbacks import EarlyStopping

class MultiModalTrainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, metrics, epochs, early_stopping, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics.to(device)
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.history = []
        self.scheduler = scheduler

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            
            labels = batch['label']
            
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            
            labels = batch['label']
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            self.metrics.update(preds, labels.long())

        avg_loss = total_loss / len(self.val_loader)
        val_metrics_results = self.metrics.compute()
        return avg_loss, val_metrics_results

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, val_metrics = self._validate_one_epoch()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch} | LR: {current_lr:.6f}")
                
            logging.info(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            epoch_log = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
            log_str = ""
            for name, value in val_metrics.items():
                metric_val = value.item()
                log_str += f" | Val {name.capitalize()}: {metric_val:.4f}"
                epoch_log[f'val_{name}'] = metric_val
            logging.info(log_str.strip(" |"))
            self.history.append(epoch_log)

            self.early_stopping(val_loss, self.model)
            
            if self.early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break
        
        logging.info("Training finished. Loading best model weights from checkpoint.")
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        return self.history

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        all_labels = []
        all_preds_proba = []
        
        for batch in data_loader:
            for key, value in batch.items():
                batch[key] = value.to(self.device)
                
            outputs = self.model(batch)
            preds_proba = torch.sigmoid(outputs)
            
            all_labels.append(batch['label'].cpu())
            all_preds_proba.append(preds_proba.cpu())
            
        return torch.cat(all_labels).detach().numpy().flatten(), torch.cat(all_preds_proba).detach().numpy().flatten()