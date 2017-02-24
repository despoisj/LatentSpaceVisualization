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

from config import latent_dim, modelsPath, imageSize
from keras.optimizers import RMSprop
from model import getModels
from visuals import visualizeDataset, visualizeReconstructedImages, computeTSNEProjectionOfLatentSpace, computeTSNEProjectionOfPixelSpace, visualizeInterpolation, visualizeArithmetics
from datasetTools import loadDataset
import numpy as np
import tensorflow as tf
from random import randint

# Handy parameters
nbEpoch = 20
batchSize = 256
modelName = "autoencoder_mnist.h5"

#Run ID for tensorboard, timestamp is for ordering
runID = "{} - Autoencoder - MNIST".format(1./time.time())

# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
    seconds = int(batchTime*(nbBatch-batchIndex-1) + batchTime*nbBatch*(nbEpoch-epoch-1))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d"%(h,m,s)

# Trains the Autoencoder, resume training with startEpoch > 0
def trainModel(startEpoch=0):
    # Create models
    print("Creating Autoencoder...")
    autoencoder, _, _ = getModels()
    autoencoder.compile(optimizer=RMSprop(lr=0.00025), loss="mse")

    # From which we start
    if startEpoch > 0:
        # Load Autoencoder weights
        print("Loading weights...")
        autoencoder.load_weights(modelsPath+modelName)

    print("Loading dataset...")
    X_train, X_test = loadDataset()

    # Compute number of batches
    nbBatch = int(X_train.shape[0]/batchSize)

    # Train the Autoencoder on dataset
    print "Training Autoencoder for {} epochs with {} batches per epoch and {} samples per batch.".format(nbEpoch,nbBatch,batchSize)
    print "Run id: {}".format(runID)

    # Debug utils writer 
    writer = tf.train.SummaryWriter("/tmp/logs/"+runID)
    batchTimes = [0. for i in range(5)]

    # For each epoch
    for epoch in range(startEpoch,nbEpoch):  
        # For each batch
        for batchIndex in range(nbBatch):
            batchStartTime = time.time()
            # Get batch
            X = X_train[batchIndex*batchSize:(batchIndex+1)*batchSize]
            
            # Train on batch
            autoencoderLoss = autoencoder.train_on_batch(X, X)
            trainingSummary = tf.Summary.Value(tag="Loss", simple_value=float(autoencoderLoss))

            # Compute ETA 
            batchTime = time.time() - batchStartTime
            batchTimes = batchTimes[1:] + [batchTime]
            eta = getETA(sum(batchTimes)/len(batchTimes), nbBatch, batchIndex, nbEpoch, epoch)
            
            # Save reconstructions on train/test samples
            if batchIndex%2 == 0:
                visualizeReconstructedImages(X_train[:16],X_test[:16],autoencoder, save=True, label="{}_{}".format(epoch,batchIndex))

            # Validation & Tensorboard Debug
            if batchIndex%20 == 0:
                validationLoss = autoencoder.evaluate(X_test[:512], X_test[:512], batch_size=256, verbose=0)
                validationSummary = tf.Summary.Value(tag="Validation Loss", simple_value=float(validationLoss))
                summary = tf.Summary(value=[trainingSummary,validationSummary])
                print "Epoch {}/{} - Batch {}/{} - Loss: {:.3f}/{:.3f} - ETA:".format(epoch+1,nbEpoch,batchIndex+1,nbBatch,autoencoderLoss,validationLoss), eta
            else:
                print "Epoch {}/{} - Batch {}/{} - Loss: {:.3f} - ETA:".format(epoch+1,nbEpoch,batchIndex+1,nbBatch,autoencoderLoss), eta
                summary = tf.Summary(value=[trainingSummary,])
            writer.add_summary(summary, epoch*nbBatch + batchIndex)

        #Save model every epoch
        print("Saving autoencoder...")
        autoencoder.save_weights(modelsPath+modelName, overwrite=True)

# Generates images and plots
def testModel():
    # Create models
    print("Creating Autoencoder, Encoder and Generator...")
    autoencoder, encoder, decoder = getModels()

    # Load Autoencoder weights
    print("Loading weights...")
    autoencoder.load_weights(modelsPath+modelName)

    # Load dataset to test
    print("Loading dataset...")
    X_train, X_test = loadDataset()

    # Visualization functions
    #visualizeReconstructedImages(X_train[:16],X_test[:16], autoencoder)
    #computeTSNEProjectionOfPixelSpace(X_test[:1000], display=True)
    #computeTSNEProjectionOfLatentSpace(X_test[:1000], encoder, display=True)
    while 1: visualizeInterpolation(X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], encoder, decoder, save=False, nbSteps=5)
    #while 1 :visualizeArithmetics(X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], encoder, decoder)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) == 2 else None 
    if arg is None:
        print "Need argument"
    elif arg == "train":
        trainModel(startEpoch=0)
    elif arg == "test":
        testModel()
    else:
        print "Wrong argument"




