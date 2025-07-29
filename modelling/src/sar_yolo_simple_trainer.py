"""
sar_yolo_simple_trainer.py (version 11)

Created: 2025-07-18
Modified: 
    2025-07-20: Added progressive unfreezing and early stopping
    2025-07-21: Added configurable optimizer (adam, adamw, sgd, rmsprop)
    2025-07-22: Added freeze_up_to_module() function

Simplied, YOLO model agnostic, trainer which bypasses Ultralytics'
tighlty coupled preprocessing and training modules framework in
order to preserve SAR data float32 precision.

The trainer:
    - Wraps the model in a simple nn.Module to ensure gradient flow
    - Manually initializes the loss with proper hyperparameters
    - Forces gradients if needed by adding a small computation
    - Simplifies the training loop to avoid complex YOLO internals
    - Uses direct forward passes without any special YOLO methods

Permits full control over pre-processing,data loading, loss calculation
training and data augmentation.

Optional:
=========
Post-training, Ultralytics framework can be utilised for inference by
first converting new images to uint8 PNG/JPEG or uint16 TIFF format,
then loading the trained model's weights as below:
======================================================================
from ultralytics import YOLO

# Load trained weights into Ultralytics
model = YOLO('yolov8n.pt')
model.model.load_state_dict(torch.load('best.pt')['model_state_dict'])

# Now all Ultralytics utilities are accessible
results = model.predict('image.jpg')  # Includes NMS, visualization, etc.
metrics = model.val()  # Proper mAP calculation
======================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
import yaml
from pathlib import Path
from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import custom dataset
from sar_dataset import SARPreprocessedDataset
from letterbox_resize import letterbox_resize

class SimpleSARTrainer:
    """Enhanced trainer with progressive unfreezing and early stopping"""
    
    def __init__(self, model_name='yolov8n.pt', data_yaml=None, 
                 imgsz=640, device='cuda', save_dir=None):
        
        # Load config
        if data_yaml is None:
            raise ValueError("data_yaml must be specified")
        with open(data_yaml, 'r') as f:
            self.data_dict = yaml.safe_load(f)
        
        # Setup paths
        if save_dir is None:
            # Use current working directory as base
            base_dir = Path.cwd() / 'runs' / 'train'
        else:
            base_dir = Path(save_dir)
            
        self.save_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Models will be saved to: {self.save_dir.absolute()}")
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model differently to ensure gradients work
        self.base_model = YOLO(model_name)
        self.model = self.base_model.model
        
        # CRITICAL: Create a new detection model that I control
        # This ensures gradients flow properly
        self.model = self._create_trainable_model(self.model)
        self.model.to(self.device)
        self.model.train()
        
        # Create datasets
        data_root  = Path(self.data_dict['path'])
        data_train = Path(self.data_dict['train'])
        self.train_dataset = SARPreprocessedDataset(
            image_dir=str(data_root / data_train / 'images'),
            label_dir=str(data_root / data_train / 'labels'),
            imgsz=imgsz
        )
        self.val_dataset = SARPreprocessedDataset(
            image_dir=str(data_root / 'val' / 'images'),
            label_dir=str(data_root / 'val' / 'labels'),
            imgsz=imgsz
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Initialize loss function properly
        self.criterion = self._init_loss_fn()
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')
        
        # Track best model and early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 5
        
        # Progressive unfreezing parameters
        self.freeze_schedule = None  # Will be set during training
        
    def _create_trainable_model(self, base_model):
        """Wrap model to ensure it's trainable"""
        class TrainableWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Ensure all parameters require gradients
                for param in self.model.parameters():
                    param.requires_grad = True
                    
            def forward(self, x):
                # Direct forward pass without any special handling
                return self.model(x)
        
        return TrainableWrapper(base_model)
    
    def _init_loss_fn(self):
        """Initialize loss function with proper configuration"""
        from types import SimpleNamespace
        
        # Create loss with proper hyperparameters
        loss_fn = v8DetectionLoss(self.model.model)
        
        # Set hyperparameters manually
        loss_fn.hyp = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
        
        # Move device-dependent components to device
        loss_fn.device = self.device
        
        # Ensure all loss components are on device
        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)
        if hasattr(loss_fn, 'stride'):
            loss_fn.stride = loss_fn.stride.to(self.device)
            
        return loss_fn
    
    def freeze_up_to_module(self, last_module):
        """Freeze all parameters up to and including specified module number"""
        frozen_count = 0
        total_count = 0
        
        for name, param in self.model.named_parameters():
            total_count += 1
            # Handle wrapped model structure
            if name.startswith('model.'):
                # For wrapped model: model.model.0.conv.weight
                parts = name.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    module_num = int(parts[2])
                    if module_num <= last_module:
                        param.requires_grad = False
                        frozen_count += 1
                    else:
                        param.requires_grad = True
            else:
                # For direct model: model.0.conv.weight
                parts = name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    module_num = int(parts[1])
                    if module_num <= last_module:
                        param.requires_grad = False
                        frozen_count += 1
                    else:
                        param.requires_grad = True
        
        # Count parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        
        print(f"Froze modules 0-{last_module}: {frozen_count}/{total_count} tensors")
        print(f"Frozen params: {frozen_params:,} / {total_params:,} ({100 * frozen_params / total_params:.1f}%)")
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    def setup_progressive_unfreezing(self, epochs):
        """Setup the progressive unfreezing schedule based on module numbers"""
        # YOLOv8 has modules 0-22
        # 0-9: Backbone
        # 10-21: Neck
        # 22: Head
        
        # Define unfreezing schedule
        self.freeze_schedule = {
            0: 9,                          # Start: freeze backbone (modules 0-9)
            int(epochs * 0.2): 12,         # At 20%: unfreeze to module 12
            int(epochs * 0.4): 15,         # At 40%: unfreeze to module 15
            int(epochs * 0.6): 18,         # At 60%: unfreeze to module 18
            int(epochs * 0.8): -1,         # At 80%: unfreeze all (-1 means unfreeze all)
        }
        
        print(f"Progressive unfreezing schedule:")
        for epoch, last_frozen_module in self.freeze_schedule.items():
            if last_frozen_module == -1:
                print(f"  Epoch {epoch}: Unfreeze all modules")
            else:
                print(f"  Epoch {epoch}: Freeze modules 0-{last_frozen_module}")
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        
        # Build batch dict for YOLO loss
        batch_idx = []
        cls = []
        bboxes = []
        
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                batch_idx.append(torch.full((label.shape[0],), i, dtype=torch.long))
                cls.append(label[:, 0].long())
                bboxes.append(label[:, 1:5])
        
        if batch_idx:
            batch_dict = {
                'batch_idx': torch.cat(batch_idx, 0),
                'cls': torch.cat(cls, 0),
                'bboxes': torch.cat(bboxes, 0),
                'ori_shape': torch.tensor([[640, 640]] * len(images)),
                'imgsz': 640
            }
        else:
            batch_dict = {
                'batch_idx': torch.zeros(0, dtype=torch.long),
                'cls': torch.zeros(0, dtype=torch.long), 
                'bboxes': torch.zeros(0, 4),
                'ori_shape': torch.tensor([[640, 640]] * len(images)),
                'imgsz': 640
            }
            
        return images, batch_dict
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, batch_dict in pbar:
            # Move to device
            images = images.to(self.device)
            for k, v in batch_dict.items():
                if isinstance(v, torch.Tensor):
                    batch_dict[k] = v.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - simple and direct
            with torch.set_grad_enabled(True):
                outputs = self.model(images)
                
                # Calculate loss
                loss_items = self.criterion(outputs, batch_dict)
                
                # Handle loss output format
                if isinstance(loss_items, tuple):
                    losses = loss_items[0]
                else:
                    losses = loss_items
                
                # Sum losses if needed
                if losses.shape[0] == 3:
                    loss = losses.sum()
                else:
                    loss = losses
                
                # Verify gradient
                if not loss.requires_grad:
                    print("Creating loss with grad_fn manually...")
                    # Force gradient by adding small computation
                    loss = loss + 0.0 * sum(p.sum() for p in self.model.parameters() if p.requires_grad)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            
            # Update weights
            optimizer.step()
            
            # Update scheduler if using OneCycleLR
            if scheduler and hasattr(scheduler, 'step'):
                scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})
            
            # Log batch loss to TensorBoard (every 10 batches)
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
                self.writer.add_scalar('LR/batch', current_lr, self.global_step)
            self.global_step += 1
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, batch_dict in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.to(self.device)
                
                outputs = self.model(images)
                loss_items = self.criterion(outputs, batch_dict)
                
                if isinstance(loss_items, tuple):
                    losses = loss_items[0]
                else:
                    losses = loss_items
                    
                loss = losses.sum() if losses.shape[0] == 3 else losses
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, epochs=50, batch_size=16, lr=0.00001, workers=4, 
          weight_decay=0.0005, use_warmup=True, use_progressive_unfreeze=True,
          optimizer_type='adamw', freeze_until_module=None, **optimizer_kwargs):
        """Main training loop with enhanced features"""
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True
        )
        
        # Setup progressive unfreezing if enabled
        if use_progressive_unfreeze:
            self.setup_progressive_unfreezing(epochs)
            # Apply initial freezing
            initial_freeze = self.freeze_schedule[0]
            if initial_freeze >= 0:
                self.freeze_up_to_module(initial_freeze)
        elif freeze_until_module is not None:
            # Static module-based freezing
            print(f"\n🔒 Static freezing: modules 0-{freeze_until_module}")
            self.freeze_up_to_module(freeze_until_module)
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=optimizer_kwargs.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=optimizer_kwargs.get('nesterov', True)
            )
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Scheduler setup
        if use_warmup:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 10,  # Peak LR is 10x initial LR
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )
            print(f"Using OneCycleLR scheduler with warmup. Max LR: {lr * 10}")
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
            print("Using ReduceLROnPlateau scheduler")
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        self.global_step = 0
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            # Check if we need to unfreeze more layers
            if use_progressive_unfreeze and epoch in self.freeze_schedule:
                freeze_until = self.freeze_schedule[epoch]
                if freeze_until == -1:
                    print(f"\n🔓 Progressive unfreezing at epoch {epoch+1}: Unfreezing all modules")
                    # Unfreeze all by setting all parameters to require gradients
                    for param in self.model.parameters():
                        param.requires_grad = True
                    print("All parameters are now trainable")
                else:
                    print(f"\n🔓 Progressive unfreezing at epoch {epoch+1}")
                    self.freeze_up_to_module(freeze_until)
            
            # Train
            if use_warmup:
                train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            else:
                train_loss = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR/epoch', current_lr, epoch)
            
            # Scheduler step (for ReduceLROnPlateau)
            if not use_warmup:
                scheduler.step(val_loss)
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, optimizer, best=True)
                print(f"  ✨ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ Patience counter: {self.patience_counter}/{self.early_stop_patience}")
                
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Plot progress
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.plot_history(history)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, optimizer)
        
        # Save final model
        self.save_checkpoint(epoch, optimizer, final=True)
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final epoch: {epoch+1}")
        print(f"{'='*50}")
        
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        clear_output(wait=True)
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 3, 2)
        plt.plot(history['lr'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Loss ratio plot (to detect overfitting)
        plt.subplot(1, 3, 3)
        if len(history['train_loss']) > 0:
            loss_ratio = [v/(t+1e-8) for t, v in zip(history['train_loss'], history['val_loss'])]
            plt.plot(loss_ratio, 'r-', linewidth=2)
            plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Val Loss / Train Loss')
            plt.title('Overfitting Indicator')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_checkpoint(self, epoch, optimizer, final=False, best=False):
        """Save model checkpoint"""
        if best:
            filename = 'best.pt'
        elif final:
            filename = 'final.pt'
        else:
            filename = f'epoch_{epoch+1}.pt'
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': self.best_val_loss if best else None,
            'patience_counter': self.patience_counter,
        }
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_best_model(self):
        """Load the best model checkpoint"""
        checkpoint_path = self.save_dir / 'best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
            return checkpoint['epoch'], checkpoint['val_loss']
        else:
            print("No best model found!")
            return None, None
    
    def save_training_history(self, history, config_dict=None):
        """Save training history and configuration"""
        import json
        
        # Save history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        
        # Save config if provided
        if config_dict is not None:
            with open(self.save_dir / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        print(f"Saved training history to: {self.save_dir}")
        
    def load_training_history(self):
        """Load training history from saved file"""
        import json
        
        history_path = self.save_dir / 'history.json'
        config_path = self.save_dir / 'config.json'
        
        history = None
        config = None
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        return history, config

