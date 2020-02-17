import cv2
import numpy as np

def handle_asl(output):
    '''
    Handles the output of the asl recognition model.
    Returns one integer = prob: the argmax of softmax output.
    '''
    output_asl = output.flatten()
    asl_pred = np.argmax(output_asl)    

    return asl_pred


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, 1, height, width)
        
    return image