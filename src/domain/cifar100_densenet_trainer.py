"""Train cifar100 densenet"""
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from src.app.dataset_reader import Cifar100DataSet
from src.app.net_inputs import prepare_dataset
from src.domain.models.datasets import DataSet
from src.domain.models.densenet import DenseNet
from src.domain.models.neural_network import NeuralNet

Callback = Tuple[ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler]


@dataclass
class Cifar100DenseNet:
    """Densenet for cifar100"""

    depth: int
    growth_rate: int
    num_dense_blocks: int
    batch_size: int = 32
    epochs: int = 200

    def get_dataset(self) -> DataSet:
        """load cifar100 dataset"""
        dataset = Cifar100DataSet()

        return dataset.load()

    def get_model(self, data: DataSet) -> DenseNet:
        """get densenet model"""
        input_shape = data.x_train.shape[1:]
        densenet = DenseNet(
            depth=self.depth,
            growth_rate=self.growth_rate,
            num_dense_blocks=self.num_dense_blocks,
            num_classes=len(np.unique(data.y_train)),
            input_shape=input_shape,
            compression_rate=0.5,
        )

        return densenet

    def get_callbacks(self, net: NeuralNet) -> Callback:
        """Get callbacks"""
        check_point = ModelCheckpoint(
            filepath=net.filepath,
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
        )
        lr_scheduler = LearningRateScheduler(net.lr_schedule)
        lr_reducer = ReduceLROnPlateau(
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6,
        )

        callbacks = (
            check_point,
            lr_reducer,
            lr_scheduler,
        )

        return callbacks

    def run(self):
        """Run DenseNet"""
        data = self.get_dataset()
        densenet = self.get_model(data=data)
        model = densenet.train_net()

        net = NeuralNet(
            model=model,
            model_type=densenet.model_type,
            dataset_name="Cifar100",
        )
        net.compile()
        net.summary()
        net.plot_model(to_file="data/dense_net_cifar100.png")
        callbacks = list(self.get_callbacks(net=net))
        x_train, y_train, x_test, y_test = prepare_dataset(dataset=data)
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )
        datagen.fit(x_train)
        steps_per_epoch = math.ceil(len(x_train) / self.batch_size)
        model = net.model
        model.fit(
            x=datagen.flow(x_train, y_train, batch_size=self.batch_size),
            verbose=1,
            epochs=self.epochs,
            validation_data=(x_test, y_test),
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
        )

        scores = model.evaluate(
            x_test,
            y_test,
            batch_size=self.batch_size,
            verbose=0,
        )
        print(f"Test loss: {scores[0]}")
        print(f"Test accuracy: {scores[1]}")
        model.save(filepath="data/cifar100_densenet.keras")
