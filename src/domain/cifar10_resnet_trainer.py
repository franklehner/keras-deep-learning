"""Train cifar10 net"""
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from src.app.net_inputs import prepare_dataset
from src.app.dataset_reader import Cifar10DataSet
from src.domain.models.resnet import Model, NeuralNetwork, ResNet
from src.domain.models.datasets import DataSet


Callback = Tuple[ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler]


@dataclass
class Cifar10Resnet:
    """Resnet for Cifar10"""

    version: int
    depth: int
    batch_size: int = 32
    epochs: int = 200

    def get_dataset(self) -> DataSet:
        """Get the data from keras"""
        dataset = Cifar10DataSet()

        return dataset.load_dataset()

    def get_model(self, data: DataSet) -> Tuple[ResNet, Model]:
        """Get resnet model"""
        input_shape = data.x_train.shape[1:]
        resnet = ResNet(
            version=self.version,
            depth=self.depth,
            input_shape=input_shape,
            num_classes=len(np.unique(data.y_train)),
        )
        if self.version == 2:
            model = resnet.resnet_v2()
        else:
            model = resnet.resnet_v1()

        return resnet, model

    def get_callbacks(self, net: NeuralNetwork) -> Callback:
        """Get callbacks"""
        check_point = ModelCheckpoint(
            filepath=net.filepath,
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
        )
        lr_scheduler = LearningRateScheduler(net.lr_scheduler)
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
        """Run Resnet"""
        data = self.get_dataset()
        resnet, model = self.get_model(data=data)
        net = NeuralNetwork(
            model=model,
            model_type=resnet.model_type,
            dataset_name="Cifar10",
        )
        net.compile()
        net.summary()
        net.plot_model(to_file="data/resnet_cifar10.png")
        callbacks = list(self.get_callbacks(net=net))
        x_train, y_train, x_test, y_test = prepare_dataset(
            dataset=data,
            subtract_pixel_mean=True,
        )
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

        model.save(filepath="data/cifar10_resnet_v1.keras")
