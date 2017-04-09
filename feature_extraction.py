import pickle
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Input, Flatten
from keras.applications import VGG16,VGG19,InceptionV3,ResNet50
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

dataset = "traffic"     # cifar10, traffic
arch = "resnet"      # resnet, inception, vgg

input_sizes = {
    "vgg":( 224, 224),
    "inception":(299,299),
    "resnet":(224, 224)
}

train_file = "./{}-100/{}_{}_100_bottleneck_features_train.p".format(arch,arch,dataset)
valid_file = "./{}-100/{}_{}_bottleneck_features_validation.p".format(arch,arch,dataset)

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', train_file, "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', valid_file, "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String 
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data

    train_path = FLAGS.training_file if FLAGS.training_file else train_file
    valid_path = FLAGS.validation_file if FLAGS.training_file else valid_file
    epochs = FLAGS.epochs if FLAGS.training_file else train_file
    batch_size = FLAGS.batch_size if FLAGS.training_file else valid_file
    print(train_path,valid_path)
    X_train, y_train, X_valid, y_valid = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    n_train = len(X_train)
    n_valid = len(X_valid)
    n_classes = len(np.unique(y_train))
    input_type = type(X_train[0, 0, 0, 0])
    label_type = type(y_train[0])

    print("n_train", n_train)
    print("n_valid",n_valid)
    print("n_classes",n_classes)
    print("input_type",input_type)
    print("label_type",label_type)


    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)


    # TODO: define your model and hyperparams here
    learn_rate = 0.0001

    input = Input(shape=X_train.shape[1:])
    x = Flatten()(input)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(input, x)

    print(123)
    # TODO: train your model here
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(4)
    score = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid,y_valid), shuffle=True)
    print(score)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
