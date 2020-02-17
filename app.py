import argparse
import cv2

from handle_models import handle_asl, preprocessing
from inference import Network

# The 100 words that are supported by Intel's ASL model
ASL = ['hello', 'nice', 'teacher', 'eat', 'no', 'happy', 'like', 'orange', 'want', 'deaf', 'school', 'sister', 'finish', 'white', 'chicken/bird', 'what', 'tired', 'friend', 'sit', 'mother', 'yes', 'student', 'learn', 'spring', 'good', 'fish', 'again', 'sad', 'table', 'must/need/should', 'where', 'father', 'milk', 'cousin', 'brother', 'paper', 'forget', 'nothing', 'book', 'girl', 'fine', 'black', 'boy', 'lost', 'family', 'hearing', 'bored/boring', 'please', 'water', 'computer', 'help', 'doctor', 'yellow', 'write', 'hungry/wish', 'but/different', 'drink', 'bathroom', 'man', 'how', 'understand', 'red', 'beautiful', 'sick', 'blue', 'green', 'english', 'name', 'you', 'who', 'same', 'nurse', 'day', 'now', 'brown', 'thank you', 'hurt', 'here', 'grandmother', 'pencil', 'walk', 'bad', 'read', 'when', 'dance', 'play', 'sign', 'go', 'big', 'sorry', 'work', 'draw', 'grandfather', 'woman', 'right', 'france', 'pink', 'know', 'live', 'night', 'apple']

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("ASL Recognition App with Voice recording")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input file"
    m_desc = "The location of the model XML file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-m", help=m_desc, default="models/asl-recognition-0003.xml")
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def save_audio(asl):
    # Import the required module for text  
    # to speech conversion 
    from gtts import gTTS 
   
    # Language in which you want to convert 
    language = 'en'
      
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=asl, lang=language, slow=False) 
      
    # Saving the converted audio in a mp3 file named as out
    myobj.save("out.mp3")   


def create_output_image(image, output):
    '''
    creates an output image showing the result of inference.
    '''    
    asl = ASL[output]
    save_audio(asl)
    # Scale the output text by the image shape
    scaler = max(int(image.shape[0] / 1000), 1)
    # Write the text of asl onto the image
    image = cv2.putText(image, 
        "{}".format(asl), 
        (40 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
        1 * scaler, (0, 0, 0), 2 * scaler)
    return image

def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, args.c)
    net_input_shape = plugin.get_input_shape()

    # Read the input image    
    image = cv2.imread(args.i)
    
    # Preprocess the input image
    preprocessed_image = preprocessing(image, net_input_shape[3], net_input_shape[4])

    # Perform synchronous inference on the image
    plugin.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = plugin.extract_output()
    processed_output = handle_asl(output)
    
    # Create an output image based on network
    output_image = create_output_image(image, processed_output)

    # Save down the resulting image
    cv2.imwrite("outputs/output.png", output_image)


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()
