# Sign Language to Text and Speech Conversion System

## Overview
The **Sign Language to Text and Speech Conversion System** is a machine learning-based application that translates American Sign Language (ASL) gestures into text and speech. This system leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to recognize sign language gestures and convert them into readable and audible formats. The project aims to bridge the communication gap between the hearing impaired and non-signers, promoting inclusivity and accessibility.

## Features
- **Sign Recognition:** Uses a pre-trained VGG16 CNN model to recognize ASL gestures from static images.
- **Text Conversion:** Identifies and translates the recognized signs into corresponding text.
- **Speech Synthesis:** Uses `gtts` to convert text output into audible speech.
- **Web-Based Interface:** Built with Streamlit for a user-friendly and interactive experience.
- **High Accuracy:** Trained on an extensive dataset to ensure precise gesture recognition.

## Technologies Used
- **Python 3.10**
- **TensorFlow & Keras** (for deep learning and model training)
- **OpenCV** (for image preprocessing)
- **Streamlit** (for web-based interaction)
- **gtts** (for text-to-speech conversion)
- **NumPy & Pandas** (for data manipulation)

## Installation
### Prerequisites
Ensure you have Python 3.10 installed along with the required dependencies.

### Clone the Repository
```bash
 git clone https://github.com/souravms256/asl-sign-language-conversion-to-text-and-speech.git
 cd asl-sign-language-conversion-to-text-and-speech
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Running the Application
```bash
streamlit run app.py
```

### Uploading an Image
- Upload a static image containing a sign language gesture.
- The system will process the image and display the recognized text.
- The text will be converted into speech output.

## Future Enhancements
- Real-time gesture recognition using webcam input.
- Support for multiple sign languages beyond ASL.
- Improved accuracy with larger and more diverse training datasets.
- Mobile and wearable device integration for portability.

## Contribution
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to the branch and create a Pull Request.


