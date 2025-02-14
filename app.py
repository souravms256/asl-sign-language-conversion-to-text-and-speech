import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import streamlit as st
from gtts import gTTS
import os

# Load the pre-trained model
model = tf.keras.models.load_model('saved_vgg16_model.h5')

# Define class labels (mapping indices to class names)
class_labels = [
    'blank',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Streamlit app title
st.title("Sign Language Conversion to Text and Speech")

# Add a description of the system
st.markdown("""
    ## Description
    This application uses a pre-trained deep learning model to recognize sign language letters from images. 
    You can upload an image of a sign language gesture, and the model will predict the corresponding letter. 
    Additionally, the predicted letter can be spoken aloud using text-to-speech functionality.
""")

# Create two columns for horizontal layout
#col1, col2 = st.columns(2)

# File uploader for image input in the first column

st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize prediction variable
prediction = None

if uploaded_file is not None:
    # Load and preprocess the test image
    test_image = image.load_img(uploaded_file, target_size=(224, 224))  # Resize to match model input size
    
    # Create a small box for the image

    st.subheader("Uploaded Image")
    st.image(test_image, caption='Uploaded Image', width=200)  # Display the image

    # Convert image to array and preprocess
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize to match training preprocessing

    # Make prediction
    result = model.predict(test_image)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(result, axis=1)[0]

    # Map the predicted class index to the corresponding label
    prediction = class_labels[predicted_class_index]

# Display prediction and speak button in the second column

if prediction is not None:
    st.header("Prediction Result")
    st.write(f"**Predicted Letter:** {prediction}")

        # Add a button to speak the predicted letter
    if st.button("Speak Predicted Letter"):
        tts = gTTS(text=prediction, lang='en')
        audio_file = "predicted_letter.mp3"
        tts.save(audio_file)
            
            # Play the audio file
        st.audio(audio_file, format='audio/mp3')

            # Optionally, remove the audio file after playing
        os.remove(audio_file)

# Add some footer information
