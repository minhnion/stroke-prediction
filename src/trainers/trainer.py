import torch
from tqdm import tqdm
import os
import logging

class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, metrics, epochs, early_stopping):
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

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for x_cat, x_cont, y in tqdm(self.train_loader, desc="Training"):
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_cat, x_cont)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        for x_cat, x_cont, y in tqdm(self.val_loader, desc="Validating"):
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
            
            outputs = self.model(x_cat, x_cont)
            loss = self.criterion(outputs, y)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            self.metrics.update(preds, y.long())

        avg_loss = total_loss / len(self.val_loader)
        val_metrics_results = self.metrics.compute()
        return avg_loss, val_metrics_results

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, val_metrics = self._validate_one_epoch()
            
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
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_labels = []
        all_preds_proba = []
        for x_cat, x_cont, y in data_loader:
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
            outputs = self.model(x_cat, x_cont)
            preds_proba = torch.sigmoid(outputs)
            all_labels.append(y.cpu())
            all_preds_proba.append(preds_proba.cpu())
        return torch.cat(all_labels).detach().numpy().flatten(), torch.cat(all_preds_proba).detach().numpy().flatten()
