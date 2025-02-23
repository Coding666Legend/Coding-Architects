import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("AI CROP DISEASE DETECTION & TREATMENT ADVISOR")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to AI Crop Disease Detection & Treatment Advisor! 🌿🔍
    
    Our mission is to help in identifying crop diseases efficiently. Upload an image of a crop, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
""")

    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")  
            result_index = model_prediction(test_image)
            #Define Class
            class_name = [
    'Apple(crop) - Apple scab(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Apple(crop) - Black rot(disease)\n\nPRECAUTIONS: \n Remove mummified fruit, ensure proper air circulation. \n\n CURE: \n Cut out infected areas, apply fungicides (Mancozeb, Captan).',
    'Apple(crop) - Cedar apple rust(disease)\n\nPRECAUTIONS: \n Remove nearby juniper trees, plant resistant varieties. \n\n CURE: \n Use sulfur or copper fungicides, prune infected leaves.',
    'Apple(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Blueberry(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Cherry(crop) - Powdery mildew(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Cherry(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Corn(crop) - Cercospora leaf spot Gray leaf spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Corn(crop) - Common rust(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Corn(crop) - Northern Leaf Blight(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Corn(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Grape(crop) - Black rot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Grape(crop) - Esca (Black Measles)(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Grape(crop) - Leaf blight (Isariopsis Leaf Spot)(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Grape(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Orange(crop) - Haunglongbing (Citrus greening)(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Peach(crop) - Bacterial spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Peach(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Pepper(crop) - bell Bacterial spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Pepper(crop) - bell Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Potato(crop) - Early blight(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Potato(crop) - Late blight(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Potato(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Raspberry(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Soybean(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Squash(crop) - Powdery mildew(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Strawberry(crop) - Leaf scorch(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Strawberry(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Bacterial spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Early blight(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Late blight(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Leaf Mold(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Septoria leaf spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Spider mites Two-spotted spider mite(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Target Spot(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Tomato Yellow Leaf Curl Virus(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Tomato mosaic virus(disease)\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.',
    'Tomato(crop) - Healthy\n\nPRECAUTIONS: \n 1. Remove and destroy infected leaves. \n 2. Avoid watering the foliage. \n 3. Apply fungicides to protect new growth. \n\n CURE: \n 1. Apply fungicides to protect new growth. \n 2. Remove and destroy infected leaves. \n 3. Avoid watering the foliage. \n 4. Prune trees to increase air circulation.']

        st.success("Model Has Detected : {}".format(class_name[result_index]))
        

