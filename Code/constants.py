# Defaults/constants.

# Model parameters
MAX_LENGTH = 350  # Length of DNA seq ,i.e., no. recurrent units
EMBEDDING_DIM = 4
HIDDEN_DIM = 128
HIDDEN_LAYERS = 3

# Training parameters
NUM_EPOCHS = 20
RANDOM_SEED = 1
BATCH_SIZE = 16
REG_LAMBDA = 0.1
DROPOUT = 0.3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0 #use weight decay later
TRAINING_SAVE_FREQUENCY = 10  # Measured in 'global steps'.
#Todo: Then you have to load from that state too