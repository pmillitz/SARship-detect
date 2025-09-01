"""
MLP Vessel Length Regressor
A modular implementation for training and loading MLP models for vessel length prediction.
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional, Dict, Any


def create_mlp_model(input_dim: int = 3, hidden_layers: list = [32, 16, 8]) -> tf.keras.Sequential:
    """
    Create the MLP model architecture.
    
    Args:
        input_dim: Input feature dimension
        hidden_layers: List of hidden layer sizes
        
    Returns:
        Compiled TensorFlow model
    """
    layers = [tf.keras.layers.Input(shape=[input_dim])]
    
    for units in hidden_layers:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        
    layers.append(tf.keras.layers.Dense(1))  # Output layer
    
    return tf.keras.Sequential(layers)


def setup_target_scaler(y_train: np.ndarray) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Setup target variable scaling.
    
    Args:
        y_train: Training target values
        
    Returns:
        Fitted scaler and scaled training targets
    """
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    return y_scaler, y_train_scaled


def create_custom_loss(y_scaler: MinMaxScaler):
    """
    Create custom absolute relative error loss function.
    
    Args:
        y_scaler: Fitted MinMaxScaler for targets
        
    Returns:
        Custom loss function
    """
    # Store scaler parameters as TensorFlow constants
    Y_MIN = tf.constant(y_scaler.data_min_[0], dtype=tf.float32)
    Y_RANGE = tf.constant(y_scaler.data_max_[0] - y_scaler.data_min_[0], dtype=tf.float32)
    
    def absolute_relative_error_scaled(y_true_scaled, y_pred_scaled):
        """Relative error loss on original scale"""
        # Convert to original scale
        y_true_original = y_true_scaled * Y_RANGE + Y_MIN
        y_pred_original = y_pred_scaled * Y_RANGE + Y_MIN
        
        # Safe division
        y_true_safe = tf.maximum(y_true_original, tf.keras.backend.epsilon())
        
        return tf.reduce_mean(tf.abs((y_pred_original - y_true_original) / y_true_safe))
    
    return absolute_relative_error_scaled


def compile_model(model: tf.keras.Sequential, loss_function, 
                 learning_rate: float = 0.0003, momentum: float = 0.9):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: TensorFlow model to compile
        loss_function: Custom loss function
        learning_rate: Learning rate for SGD optimizer
        momentum: Momentum for SGD optimizer
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, 
        momentum=momentum, 
        nesterov=True
    )
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae'])


def create_callbacks(checkpoint_path: str, X_val: np.ndarray, y_val: np.ndarray, 
                    y_scaler: MinMaxScaler, vla_freq: int = 5) -> list:
    """
    Create training callbacks.
    
    Args:
        checkpoint_path: Path to save best model
        X_val: Validation features
        y_val: Validation targets (original scale)
        y_scaler: Fitted MinMaxScaler for targets
        vla_freq: Frequency to print VLA scores
        
    Returns:
        List of callbacks
    """
    vla_callback = VLACallback(X_val, y_val, y_scaler, freq=vla_freq)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False
    )
    
    return [vla_callback, model_checkpoint_callback]


def train_model(model: tf.keras.Sequential, X_train: np.ndarray, y_train_scaled: np.ndarray,
               X_val: np.ndarray, y_val_scaled: np.ndarray, callbacks: list,
               epochs: int = 120, batch_size: int = 32, verbose: int = 0,
               random_seed: int = 42):
    """
    Train the MLP model.
    
    Args:
        model: Compiled TensorFlow model
        X_train: Training features
        y_train_scaled: Scaled training targets
        X_val: Validation features
        y_val_scaled: Scaled validation targets
        callbacks: List of training callbacks
        epochs: Number of training epochs
        batch_size: Training batch size
        verbose: Training verbosity
        random_seed: Random seed for reproducibility
        
    Returns:
        Training history
    """
    tf.random.set_seed(random_seed)
    
    history = model.fit(
        X_train, y_train_scaled,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(X_val, y_val_scaled),
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def load_mlp_model(checkpoint_path: str, y_scaler: MinMaxScaler) -> tf.keras.Sequential:
    """
    Load a complete saved MLP model.
    
    Args:
        checkpoint_path: Path to saved model
        y_scaler: MinMaxScaler used during training
        
    Returns:
        Loaded TensorFlow model
    """
    # Recreate the custom loss function for loading
    loss_function = create_custom_loss(y_scaler)
    
    # Load complete model with custom objects
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'absolute_relative_error_scaled': loss_function}
    )
    
    print(f"Complete model loaded from {checkpoint_path}")
    return model


def make_predictions(model: tf.keras.Sequential, X: np.ndarray, 
                    y_scaler: MinMaxScaler, return_scaled: bool = False) -> np.ndarray:
    """
    Make predictions with the model.
    
    Args:
        model: Trained TensorFlow model
        X: Input features
        y_scaler: Fitted MinMaxScaler for targets
        return_scaled: If True, return scaled predictions
        
    Returns:
        Predictions (original scale unless return_scaled=True)
    """
    y_pred_scaled = model.predict(X, verbose=0)
    
    if return_scaled:
        return y_pred_scaled.flatten()
    else:
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_original.flatten()


class VLACallback(tf.keras.callbacks.Callback):
    """Callback for monitoring Vessel Length Accuracy during training."""
    
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, 
                 y_scaler: MinMaxScaler, freq: int = 10):
        """
        Initialize VLA callback.
        
        Args:
            X_val: Validation features
            y_val: Validation targets (original scale)
            y_scaler: Fitted MinMaxScaler for targets
            freq: Frequency (in epochs) to print VLA
        """
        self.X_val = X_val
        self.y_val = y_val
        self.y_scaler = y_scaler
        self.freq = freq
    
    def on_epoch_end(self, epoch, logs=None):
        """Print VLA at specified frequency."""
        if epoch % self.freq == 0:
            y_pred_scaled = self.model.predict(self.X_val, verbose=0)
            y_pred_original = self.y_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            
            relative_errors = np.abs(y_pred_original - self.y_val) / self.y_val
            vla = 1 - np.clip(np.mean(relative_errors), 0, 1)
            
            print(f"\nEpoch {epoch+1} - Validation VLA: {vla:.3f}")


def calculate_metrics_by_class(y_true: np.ndarray, y_pred: np.ndarray, 
                             classes: np.ndarray, split_name: str = "") -> Dict[str, Any]:
    """
    Calculate metrics separately for each class with clean formatting.
    
    Args:
        y_true: True vessel lengths
        y_pred: Predicted vessel lengths
        classes: Class labels (0=vessel, 1=fishing)
        split_name: Name for printing (e.g., "Train", "Val", "Test")
    
    Returns:
        Dictionary with overall and per-class metrics
    """
    # Helper function to calculate metrics
    def _calculate_metrics(y_t, y_p):
        relative_errors = np.abs(y_p - y_t) / y_t
        mean_relative_error = np.mean(relative_errors)
        clipped_error = np.clip(mean_relative_error, 0, 1)
        vla = 1 - clipped_error
        
        return {
            'mae': mean_absolute_error(y_t, y_p),
            'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
            'r2': r2_score(y_t, y_p),
            'vla': vla
        }
    
    # Overall metrics
    overall_metrics = _calculate_metrics(y_true, y_pred)
    
    # Initialize results
    results = {
        'overall': overall_metrics,
        'by_class': {}
    }
    
    # Calculate metrics for each class
    unique_classes = np.unique(classes)
    class_names = {0: 'is_vessel', 1: 'is_fishing'}
    
    for cls in unique_classes:
        mask = classes == cls
        if np.sum(mask) > 0:  # Check if class has samples
            cls_metrics = _calculate_metrics(y_true[mask], y_pred[mask])
            cls_metrics['n_samples'] = int(np.sum(mask))
            results['by_class'][int(cls)] = cls_metrics
    
    # Print formatted results
    if split_name:
        print(f"\n=== {split_name} Metrics ===")
        print(f"{'Overall':<12} {'MAE:':<4} {overall_metrics['mae']:>6.2f}, {'RMSE:':<5} {overall_metrics['rmse']:>6.2f}, {'R²:':<3} {overall_metrics['r2']:>6.3f}, {'VLA:':<4} {overall_metrics['vla']:>6.3f}")
        
        for cls, metrics in results['by_class'].items():
            cls_name = class_names.get(cls, f"Class_{cls}")
            print(f"{cls_name:<12} {'MAE:':<4} {metrics['mae']:>6.2f}, {'RMSE:':<5} {metrics['rmse']:>6.2f}, {'R²:':<3} {metrics['r2']:>6.3f}, {'VLA:':<4} {metrics['vla']:>6.3f} (n={metrics['n_samples']})")
    
    return results