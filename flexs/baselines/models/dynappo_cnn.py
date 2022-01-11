"""Define a baseline CNN Model."""
import tensorflow as tf

import flexs
import sklearn.metrics
from . import keras_model

class CNNEnsemble(flexs.Model):
    """A baseline CNN ensemble model with 2 conv layers and 1 dense layer each."""
    def __init__(self, seq_len: int, alphabet: str):
        self.seq_len = seq_len
        self.alphabet = alphabet

        super().__init__(
            name='DynaPPOCNNEnsemble',
            hparam_tune=True,
            hparams_to_search={
                'lr': [1e-2, 1e-3, 1e-4],
                'epochs': [5, 10, 20, 40],
            },
            nfolds=5,
        )

    def _create_ensemble(self, lr):
        models = [self._create_cnn(seq_len, alphabet) for _ in range(10)]
        self.model = tf.keras.layers.Average([m.outputs[0] for m in models])

        self.model.compile(
            loss='MSE',
            optimizer=tf.keras.optimizer.Adam(learning_rate=lr),
            metrics=['mse']
        )

    def _seqs_to_tensor(self, seqs):
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in sequences
        ])

        return tf.convert_to_tensor(one_hots, dtype=tf.float32)

    def _train_hparam_setting(
        self,
        train_seq,
        train_label,
        val_seq,
        val_label,
        epochs=20,
        lr=1e-3
    ):
        self._create_ensemble(lr)

        train_one_hots = self._seqs_to_tensor(train_seq)
        val_one_hots = self._seqs_to_tensor(val_seq)

        train_label = tf.convert_to_tensor(train_label)
        val_label = tf.convert_to_tensor(val_label)

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)]
        self.model.fit(
            x=train_one_hots,
            y=train_label,
            batch_size=256,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(val_one_hots, val_label)
        )

        val_preds = self.model.predict(val_one_hots, batch_size=256).squeeze(axis=1)
        return sklearn.metrics.r2_score(val_label.numpy(), val_preds.numpy())

    def _train(self, sequences, labels, epochs=20, lr=1e-3):
        self._create_ensemble(lr)

        one_hots = = self._seqs_to_tensor(sequences)
        labels = tf.convert_to_tensor(labels)

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)]
        self.model.fit(
            x=train_one_hots,
            y=train_label,
            batch_size=256,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(val_one_hots, val_label)
        )

    def _create_cnn(self, seq_len, alphabet):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(
                32,
                5,
                padding="valid",
                activation="relu",
                strides=1,
                input_shape=(seq_len, len(alphabet)),
            ),
            tf.keras.layers.Conv1D(
                32,
                5,
                padding="same",
                activation="relu",
                strides=1,
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def _fitness_function(self, sequences):
        one_hots = = self._seqs_to_tensor(sequences)

        return np.nan_to_num(
            self.model.predict(one_hots, batch_size=self.batch_size).squeeze(axis=1)
        )
