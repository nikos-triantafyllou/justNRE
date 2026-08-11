import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


def build_smooth_mlp(input_dim, dropout=0.3, lr=1e-3, loss='binary_crossentropy',
                      last_activation='sigmoid', L2_reg=1e-8):
    """Pure function to compile a highly regularized, smooth MLP."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(L2_reg)),
        BatchNormalization(),
        Dropout(dropout),  # High dropout pushes the model to default to 0.5 if features are noise
        Dense(1, activation=last_activation)
    ])
    model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    return model


def train_nre_ensemble(X_tr, y_tr, X_va, y_va, num_models=5, base_seed=42,
                        epochs=150, batch_size=128, patience=40, verbose=0,
                        dropout=0.3, lr=1e-3, loss='binary_crossentropy',
                        last_activation='sigmoid', plot=True):
    """
    Trains an ensemble of independent models.

    Returns:
        ensemble: list of trained Keras models
        loss_figs: list of matplotlib Figures (one per member's train/val loss
            curve), or [] if plot=False. Caller decides whether to show/save
            them (figures are not auto-displayed here).
    """
    ensemble = []
    loss_figs = []

    for i in range(num_models):
        print(f"\n--- Training Ensemble Member {i+1}/{num_models} ---")
        K.clear_session()

        # Set unique seed per member for diverse initialization and shuffling.
        # Note: tf.random.set_seed / np.random.seed here are intentionally
        # global -- Keras layer init and shuffling read the global RNG, so
        # this is the normal way to seed a training run reproducibly.
        tf.random.set_seed(base_seed + i)
        np.random.seed(base_seed + i)

        # Build and fit
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        model = build_smooth_mlp(input_dim=X_tr.shape[1], dropout=dropout, lr=lr,
                                  loss=loss, last_activation=last_activation)
        history = model.fit(
            X_tr, y_tr,
            epochs=epochs,
            validation_data=(X_va, y_va),
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stopping],
            shuffle=True,
        )
        ensemble.append(model)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], color='black', linestyle="-", label='train loss')
            ax.plot(history.history['val_loss'], color='teal', linestyle="-", label='val loss')
            ax.set_ylim([0, 1])
            ax.set_xlabel('epochs')
            ax.set_ylabel('error')
            ax.set_title(f'Ensemble member {i+1}/{num_models}')
            ax.grid()
            ax.legend()
            loss_figs.append(fig)

    return ensemble, loss_figs


def ensemble_predict(nre_ensemble, X_test, X_test_norm=None, normalize_fn=None):
    if X_test_norm is None:
        X_test_norm = normalize_fn(X_test)
    X_clean = np.asarray(X_test_norm, dtype=np.float32)

    all_preds = []
    for model in nre_ensemble:
        preds = model.predict(X_clean, verbose=0)
        all_preds.append(preds)

    return np.mean(all_preds, axis=0)
