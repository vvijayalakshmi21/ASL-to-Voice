# I can listen to your sign!
Intel &amp; Udacity - Social Good - Project Showcase <br />
ASL stands for [American Sign language](https://en.wikipedia.org/wiki/American_Sign_Language) which is a predominant method of communication used by deaf (or hard of hearing) communities. The speaker uses hand gestures to express a word/letter. Although there is a pre-trained model available in OpenVINO toolkit (V2020) to convert ASL to text in real-time, there is not a provision to listen to the text in audio format. 
This project proposes to include audio feature in addition to the text display. This is crucial because this will help those that do not see (i.e., visually challenged) or are illiterate to be actively engaged in the communication. <br /> <br /> Please note that this project is NOT complete and only a prototype that currently works for image files. The aim of this submission is to propose to include audio feature for the input (which would be a video file or webcam).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
OpenVINO Toolkit (V2020) <br />
This can be downloaded from [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html).<br />
Please ensure that you download the 2020 version of the toolkit. 

### Running the tests
The test can be run from command line.

**Example**: 
```python app.py -m <path-to-asl-recognition-model>.xml -i <path-to-input-file>.jpg``` 

**-m** refers to the path of the model that will be converting ASL to text. Note that this is an **optional** parameter and the default will be Intel's pre-trained model -**asl-recognition-0003.xml**  <br />
**-i** refers to the path of the input file (here it is hello.jpg). This is a mandatory parameter

**Other optional arguments**
<br />**-c** file location of CPU extension, if using CPU (not required for OpenVINO Version>=2020.x)<br />
**-d** intel hardware used if not CPU (GPU, FPGA, MYRIAD)

## Future Work/Scope for extension
1) Convert speech to ASL - The speech output could be converted to ASL and displayed in real-time so that communication between an ASL speaker and a normal speaker becomes seamless <br />
2) Support sign language in multiple languages (like BSL, CSL, etc.,)

### Authors
Vijaya Lakshmi Venkatraman
