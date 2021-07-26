""" 
DISCLAIMER:
This code has been written in the optic
of my 'quick-n-dirty' Deep Learning series
on Medium (@juliendespois) to show the 
concepts. Please do not judge me by the 
quality of the code. 
¯\_(ツ)_/¯
"""

import sys
import time

from config import latent_dim, models_path, img_size
from keras.optimizers import RMSprop
from model import get_models
from visuals import visualizeDataset, visualizeReconstructedImages, computeTSNEProjectionOfLatentSpace, computeTSNEProjectionOfPixelSpace, visualizeInterpolation, visualizeArithmetics
from dataset_tools import load_Dataset
import numpy as np
import tensorflow as tf
from random import randint

# Handy parameters
nb_epochs = 20
batch_size = 256
model_name = "autoencoder_mnist.h5"

#Run ID for tensorboard, timestamp is for ordering
run_id = "{} - Autoencoder - MNIST".format(1. / time.time())

# Returns the string of remaining training time
def get_eta(batch_time, nb_batch, batch_index, nb_epochs, epoch):
    seconds = int(batch_time * (nb_batch - batch_index - 1) + batch_time * nb_batch * (nb_epochs - epoch - 1))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d"%(h,m,s)

# Trains the Autoencoder, resume training with start_epoch > 0
def train_model(start_epoch=0):
    # Create models
    print("Creating Autoencoder...")
    autoencoder, _, _ = get_models()
    autoencoder.compile(optimizer=RMSprop(lr=0.00025), loss="mse")

    # Resuming training
    if start_epoch != 0:
        # Load Autoencoder weights
        print("Loading weights...")
        autoencoder.load_weights(models_path+model_name)

    print("Loading dataset...")
    X_train, X_test = load_dataset()

    # Compute number of batches
    nb_batch = int(X_train.shape[0]/batch_size)

    # Train the Autoencoder on dataset
    print("Training Autoencoder for {} epochs with {} batches per epoch and {} samples per batch.".format(nb_epochs, nb_batch, batch_size))
    print("Run id: {}".format(run_id))

    # Debug utils writer 
    writer = tf.train.SummaryWriter("/tmp/logs/" + run_id)
    batch_times = [0. for i in range(5)]

    # For each epoch
    for epoch in range(start_epoch, nb_epochs):  
        # For each batch
        for batch_index in range(nb_batch):
            batch_start_time = time.time()
            # Get batch
            X = X_train[batch_index * batch_size:(batch_index + 1) * batch_size]
            
            # Train on batch
            autoencoder_loss = autoencoder.train_on_batch(X, X)
            training_summary = tf.Summary.Value(tag="Loss", simple_value=float(autoencoder_loss))

            # Compute ETA 
            batch_time = time.time() - batch_start_time
            batch_times = batch_times[1:] + [batch_time]
            eta = get_eta(sum(batch_times)/len(batch_times), nb_batch, batch_index, nb_epochs, epoch)
            
            # Save reconstructions on train/test samples
            if batch_index%2 == 0:
                visualizeReconstructedImages(X_train[:16],X_test[:16],autoencoder, save=True, label="{}_{}".format(epoch,batch_index))

            # Validation & Tensorboard Debug
            if batch_index%20 == 0:
                validation_loss = autoencoder.evaluate(X_test[:512], X_test[:512], batch_size=256, verbose=0)
                validation_summary = tf.Summary.Value(tag="Validation Loss", simple_value=float(validation_loss))
                summary = tf.Summary(value=[training_summary, validation_summary])
                print("Epoch {}/{} - Batch {}/{} - Loss: {:.3f}/{:.3f} - ETA:".format(epoch + 1, nb_epochs, batch_index + 1, nb_batch, autoencoder_loss, validation_loss), eta)
            else:
                print("Epoch {}/{} - Batch {}/{} - Loss: {:.3f} - ETA:".format(epoch+1,nb_epochs,batch_index+1,nb_batch,autoencoder_loss), eta)
                summary = tf.Summary(value=[training_summary,])
            writer.add_summary(summary, epoch*nb_batch + batch_index)

        #Save model every epoch
        print("Saving autoencoder...")
        autoencoder.save_weights(models_path+model_name, overwrite=True)

# Generates images and plots
def testModel():
    # Create models
    print("Creating Autoencoder, Encoder and Generator...")
    autoencoder, encoder, decoder = get_models()

    # Load Autoencoder weights
    print("Loading weights...")
    autoencoder.load_weights(models_path+model_name)

    # Load dataset to test
    print("Loading dataset...")
    X_train, X_test = load_dataset()

    # Visualization functions
    #visualizeReconstructedImages(X_train[:16],X_test[:16], autoencoder)
    #computeTSNEProjectionOfPixelSpace(X_test[:1000], display=True)
    #computeTSNEProjectionOfLatentSpace(X_test[:1000], encoder, display=True)
    while 1: visualizeInterpolation(X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], encoder, decoder, save=False, nbSteps=5)
    #while 1 :visualizeArithmetics(X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], encoder, decoder)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) == 2 else None 
    if arg is None:
        print("Need argument")
    elif arg == "train":
        train_model(start_epoch=0)
    elif arg == "test":
        testModel()
    else:
        print("Wrong argument")




