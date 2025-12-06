"""
Energy Consumption Prediction Model
Bidirectional LSTM with Attention Mechanism
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for time-series data
    Implements self-attention mechanism to focus on important time steps
    """

    def __init__(self, units=32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_vector',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Attention scores calculation
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)

        # Context vector calculation
        context_vector = attention_weights * x
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class EnergyPredictionModel:
    """
    Advanced Energy Consumption Prediction Model
    Uses Bidirectional LSTM with Attention Mechanism
    """

    def __init__(
            self,
            sequence_length=30,
            n_features=10,
            lstm_units_1=128,
            lstm_units_2=64,
            attention_units=32,
            dense_units_1=32,
            dense_units_2=16,
            dropout_rate=0.2,
            learning_rate=0.001
    ):
        """
        Initialize the model with configurable parameters

        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            attention_units: Units in attention layer
            dense_units_1: Units in first dense layer
            dense_units_2: Units in second dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.attention_units = attention_units
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.history = None

    def build_model(self):
        """
        Build the Bidirectional LSTM with Attention architecture

        Architecture:
        Input → Bi-LSTM(128) → Dropout → Bi-LSTM(64) → Dropout →
        Attention → Dense(32) → Dense(16) → Output
        """

        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))

        # First Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units_1,
                return_sequences=True,
                kernel_initializer='glorot_uniform'
            )
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)

        # Second Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units_2,
                return_sequences=True,
                kernel_initializer='glorot_uniform'
            )
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Attention mechanism
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention_layer'
        )(x)

        # Dense layers for final prediction
        x = layers.Dense(
            self.dense_units_1,
            activation='relu',
            kernel_initializer='glorot_uniform'
        )(attention_output)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(
            self.dense_units_2,
            activation='relu',
            kernel_initializer='glorot_uniform'
        )(x)

        # Output layer (single value prediction)
        outputs = layers.Dense(1, activation='linear')(x)

        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        return self.model

    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def train(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            batch_size=32,
            verbose=1
    ):
        """
        Train the model with early stopping and model checkpointing

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            verbose: Verbosity mode

        Returns:
            Training history
        """

        if self.model is None:
            self.build_model()

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )

        model_checkpoint = callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        tensorboard = callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],
            verbose=verbose
        )

        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        results = self.model.evaluate(X_test, y_test, verbose=0)

        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2],
            'rmse': results[3]
        }

        # Calculate R² score
        y_pred = self.predict(X_test)
        ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        metrics['r2_score'] = r2_score

        return metrics

    def plot_training_history(self, save_path=None):
        """
        Plot training history

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            raise ValueError("No training history available.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # RMSE
        axes[1, 0].plot(self.history.history['rmse'], label='Train RMSE')
        axes[1, 0].plot(self.history.history['val_rmse'], label='Val RMSE')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # MAPE
        axes[1, 1].plot(self.history.history['mape'], label='Train MAPE')
        axes[1, 1].plot(self.history.history['val_mape'], label='Val MAPE')
        axes[1, 1].set_title('Mean Absolute Percentage Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_model(self, path='models/energy_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path='models/energy_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(
            path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"Model loaded from {path}")
        return self.model

    def get_attention_weights(self, X):
        """
        Extract attention weights for visualization

        Args:
            X: Input features

        Returns:
            Attention weights
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # Create a model that outputs attention weights
        attention_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('attention_layer').output[1]
        )

        attention_weights = attention_model.predict(X)
        return attention_weights


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = EnergyPredictionModel(
        sequence_length=30,
        n_features=10,
        lstm_units_1=128,
        lstm_units_2=64
    )

    # Build model
    model.build_model()

    # Print summary
    print(model.summary())

    # Generate dummy data for testing
    X_train = np.random.randn(1000, 30, 10)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 30, 10)
    y_val = np.random.randn(200)

    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32
    )

    # Evaluate
    X_test = np.random.randn(100, 30, 10)
    y_test = np.random.randn(100)

    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")