import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib
from typing import Dict, Tuple, Optional, Any, Union


class VesselLengthRegressor:
    """
    A vessel length regressor for predicting vessel lengths from bounding box features.
    Supports training, evaluation, visualization, and model persistence.
    """
    
    def __init__(self, 
                 model_params: Optional[Dict[str, Any]] = None,
                 feature_cols: Optional[list] = None,
                 sample_weights: Optional[Dict[int, float]] = None):
        """
        Initialize the regressor.
        
        Args:
            model_params: Parameters for the RandomForestRegressor
            feature_cols: List of feature column names
            sample_weights: Dictionary mapping class labels to weights (e.g., {0: 0.8, 1: 1.0})
        """
        self.model_params = model_params or {'n_estimators': 100, 'random_state': 42}
        self.feature_cols = feature_cols or ['width', 'height', 'class']
        self.sample_weights = sample_weights or {0: 1.0, 1: 1.0}  # Default: equal weights
        self.target_col = 'vessel_length_m'
        self.model = None
        
    def load_and_prepare_data(self, 
                            train_csv: str, 
                            val_csv: str, 
                            test_csv: str) -> Tuple[np.ndarray, ...]:
        """
        Load and prepare training, validation, and test datasets.
        
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV  
            test_csv: Path to test CSV
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Load datasets and remove any NaNs
        df_train = pd.read_csv(train_csv).dropna(subset=[self.target_col])
        df_val = pd.read_csv(val_csv).dropna(subset=[self.target_col])
        df_test = pd.read_csv(test_csv).dropna(subset=[self.target_col])
        
        # Extract features and targets
        X_train = df_train[self.feature_cols].values
        y_train = df_train[self.target_col].values
        
        X_val = df_val[self.feature_cols].values
        y_val = df_val[self.target_col].values
        
        X_test = df_test[self.feature_cols].values
        y_test = df_test[self.target_col].values
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """
        Train the regressor model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained regressor model
        """
        # Filter out sample_weights from model_params if present
        model_init_params = {k: v for k, v in self.model_params.items() if k != 'sample_weights'}
        self.model = RandomForestRegressor(**model_init_params)
        
        # Create sample weights based on class
        if 'class' in self.feature_cols:
            class_idx = self.feature_cols.index('class')
            classes = X_train[:, class_idx]
            # Map class labels to weights using the configured sample_weights
            sample_weights = np.array([self.sample_weights.get(int(cls), 1.0) for cls in classes])
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)
        
        return self.model
    
    
    def calculate_metrics_by_class(self, y_true: np.ndarray, y_pred: np.ndarray, 
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
        
        # Print formatted results with cleaner output
        if split_name:
            print(f"\n=== {split_name} Metrics ===")
#            print(f"{'Overall':<12} {'MAE:':<4} {overall_metrics['mae']:>6.2f}, {'RMSE:':<5} {overall_metrics['rmse']:>6.2f}, {'R²:':<3} {overall_metrics['r2']:>6.3f}, {'VLA:':<4} {overall_metrics['vla']:>6.3f}")
            print(f"{'Overall':<12} {'MAE:':<4} {overall_metrics['mae']:>6.2f}, {'R²:':<3} {overall_metrics['r2']:>6.3f}, {'VLA:':<4} {overall_metrics['vla']:>6.3f}")
            
            for cls, metrics in results['by_class'].items():
                cls_name = class_names.get(cls, f"Class_{cls}")
#                print(f"{cls_name:<12} {'MAE:':<4} {metrics['mae']:>6.2f}, {'RMSE:':<5} {metrics['rmse']:>6.2f}, {'R²:':<3} {metrics['r2']:>6.3f}, {'VLA:':<4} {metrics['vla']:>6.3f} (n={metrics['n_samples']})")
                print(f"{cls_name:<12} {'MAE:':<4} {metrics['mae']:>6.2f}, {'R²:':<3} {metrics['r2']:>6.3f}, {'VLA:':<4} {metrics['vla']:>6.3f} (n={metrics['n_samples']})")
        
        return results
    
    def save_model(self, save_path: str, log_dir: str) -> str:
        """
        Save the trained model.
        
        Args:
            save_path: Model save path (filename or full path)
            log_dir: Log directory for relative paths
            
        Returns:
            Full path where model was saved
        """
        if not self.model:
            raise ValueError("No trained model to save. Train the model first.")
            
        # If no directory specified, save to log directory
        if not os.path.dirname(save_path):
            full_save_path = os.path.join(log_dir, save_path)
            os.makedirs(log_dir, exist_ok=True)
        else:
            full_save_path = save_path
        
        joblib.dump(self.model, full_save_path)
        print(f"Regressor saved to {full_save_path}")
        return full_save_path
    
    def train_and_evaluate(self,
                          train_csv: str,
                          val_csv: str,
                          test_csv: str,
                          log_dir: str = 'runs/vessel_length',
                          save_model_path: Optional[str] = None,
                          return_plot_data: bool = False):
        """
        Complete training and evaluation pipeline.
        
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            test_csv: Path to test CSV
            log_dir: Directory for saving model and logs
            save_model_path: Optional path to save model
            return_plot_data: If True, return data for creating plots
            
        Returns:
            If return_plot_data=False: Trained regressor model
            If return_plot_data=True: Tuple of (trained model, plot_data_dict)
        """
        # Load and prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_prepare_data(
            train_csv, val_csv, test_csv
        )
        
        # Extract class information for plotting
        train_classes = X_train[:, self.feature_cols.index('class')] if 'class' in self.feature_cols else None  # added later
        val_classes = X_val[:, self.feature_cols.index('class')] if 'class' in self.feature_cols else None
        test_classes = X_test[:, self.feature_cols.index('class')] if 'class' in self.feature_cols else None
        
        # Train model
        self.train_regressor(X_train, y_train)
        
        # Make predictions  
        y_train_pred = self.model.predict(X_train)  # added later
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)  # added later
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Print results
        print(f"[Train] MAE: {train_metrics['mae']:.2f}, RMSE: {train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.3f}, VLA: {train_metrics['vla']:.3f}")  # added later
        print(f"[Test]  MAE: {test_metrics['mae']:.2f}, RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.3f}, VLA: {test_metrics['vla']:.3f}")
        
        
        # Save model if requested
        if save_model_path:
            self.save_model(save_model_path, log_dir)
        
        # Return model and optionally plot data
        if return_plot_data:
            plot_data = {
                'train_true': y_train,  # added later 
                'train_pred': y_train_pred,  # added later                 
                'test_true': y_test,
                'test_pred': y_test_pred,
                'train_metrics': train_metrics,  # added later
                'test_metrics': test_metrics,
                'train_classes': train_classes,  # added later
                'test_classes': test_classes
            }
            return self.model, plot_data
        else:
            return self.model
    
    def _create_single_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          metrics: Dict[str, float], title_prefix: str, 
                          classes: Optional[np.ndarray] = None) -> None:
        """
        Create and display a single scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Metrics dictionary
            title_prefix: Prefix for plot title (e.g., 'Validation', 'Test')
            classes: Optional class labels for coloring
        """
        plt.figure(figsize=(8, 8))
        
        if classes is not None:
            # Color by class
            colors = ['red' if c == 0 else 'lime' for c in classes]
            plt.scatter(y_true, y_pred, c=colors, alpha=0.6, edgecolors='k')
            
            # Add legend
            import matplotlib.patches as mpatches
            vessel_patch = mpatches.Patch(color='red', label='is_vessel')
            fishing_patch = mpatches.Patch(color='lime', label='is_fishing')
            plt.legend(handles=[vessel_patch, fishing_patch,
                               plt.Line2D([0], [0], color='red', linestyle='--', label='Ideal')])
        else:
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
            plt.legend(['Ideal'])

        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Vessel Length', fontsize=14)
        plt.ylabel('Predicted Vessel Length', fontsize=14)
        plt.title(f"{title_prefix}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}, VLA={metrics['vla']:.3f}")
        
        # Set fixed axis limits (square figure size ensures square appearance)
        plt.xlim(0, 350)
        plt.ylim(0, 350)
        
        plt.grid(True)
        plt.show()

    def plot_results(self, plot_data: Dict, save_path: Optional[str] = None, log_dir: str = 'runs/vlength') -> None:
        """
        Create and display plots for notebook use.
        
        Args:
            plot_data: Dictionary containing true/pred values and metrics
            save_path: Optional path to save the plot (e.g., 'results.png')
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        datasets = [
            ('train', 'Training'),
            ('test', 'Test')
        ]
        
        for i, (split, title) in enumerate(datasets):
            ax = axes[i]
            plt.sca(ax)  # Set current axis
            
            y_true = plot_data[f'{split}_true']
            y_pred = plot_data[f'{split}_pred']
            metrics = plot_data[f'{split}_metrics']
            classes = plot_data.get(f'{split}_classes')
            
            if classes is not None:
                # Color by class
                colors = ['red' if c == 0 else 'lime' for c in classes]
                ax.scatter(y_true, y_pred, c=colors, alpha=0.6, edgecolors='k')

                # Add legend
                import matplotlib.patches as mpatches
                vessel_patch = mpatches.Patch(color='red', label='is_vessel')
                fishing_patch = mpatches.Patch(color='lime', label='is_fishing')
                ax.legend(handles=[vessel_patch, fishing_patch])
            else:
                ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')

            ax.plot([0, 350], [0, 350], 'r--')
            ax.set_xlabel('True Vessel Length', fontsize=14)
            ax.set_ylabel('Predicted Vessel Length', fontsize=14)
            ax.set_title(f"{title}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}, VLA={metrics['vla']:.3f}", fontsize=10.5)
            
            # Set fixed axis limits
            ax.set_xlim(0, 350)
            ax.set_ylim(0, 350)
            ax.grid(True)
        
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            # If no directory specified, save to log directory (same logic as model saving)
            if not os.path.dirname(save_path):
                full_save_path = os.path.join(log_dir, save_path)
                os.makedirs(log_dir, exist_ok=True)
        else:
            full_save_path = save_path
    
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {full_save_path}") 

        plt.show()
