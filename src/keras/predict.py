from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy

from src.keras.dataset import DataGenerator
from src.keras.model import _3d_cnn_model

from keras.models import load_model
from src.common.helper import preproc
import numpy as np

import sys

if __name__ == "__main__":

    test_file = sys.argv[1]
 
    num_classes = 10
    input_shape = (20, 80, 40, 1)

    #model = _3d_cnn_model(input_shape, num_classes)
    model = load_model('lifesize_3d_cnn.h5')

    info=[20, 80, 40]
    feat = preproc(test_file, info)
    feat = np.expand_dims(feat, axis=0)

    # softmax output
    result = model.predict( feat )

    print( result )

    best_index = np.argmax(result)

    user_mapping = np.load("user_mapping.npy").tolist()
    best_user = user_mapping[best_index] 

    print( "best index:", best_index, " best user:", best_user)   

