from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint

from src.keras.dataset import DataGenerator
from src.keras.model import _3d_cnn_model

if __name__ == "__main__":

    num_classes = 10
    input_shape = (20, 80, 40, 1)
    epochs = 30

    #train_iter = DataGenerator('dummy_dir', 10)
    train_iter = DataGenerator('/var/www/html/record/data', 10)

    model = _3d_cnn_model(input_shape, num_classes)

    opt = Adam()
    loss = categorical_crossentropy

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

   
    # save better result 
    #filepath = "lifesize_3d_cnn_{epoch:02d}-{val_acc:.2f}.h5"
    filepath = "lifesize_3d_cnn_{epoch:02d}.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(train_iter,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        steps_per_epoch = 32 
                        )

    model.save('lifesize_3d_cnn.h5')
   

