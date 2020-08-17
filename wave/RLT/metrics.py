import numpy as np

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all videos.

    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels:  (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(outputs.size)
