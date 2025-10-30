import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, metrics, epochs, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics.to(device)
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        
        self.best_val_metric = -1 
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
            
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for name, value in val_metrics.items():
                print(f"  Val {name.capitalize()}: {value.item():.4f}")
            
            current_f1 = val_metrics['f1_score'].item()
            if current_f1 > self.best_val_metric:
                self.best_val_metric = current_f1
                self._save_checkpoint(epoch, val_loss, current_f1)
                print(f"  -> New best model saved with F1-score: {current_f1:.4f}")

    def _save_checkpoint(self, epoch, val_loss, val_metric):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_metric': val_metric
        }
        filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save(state, filepath)