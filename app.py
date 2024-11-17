"""import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

print("Path to dataset files:", path)

st.header('Fashion Recommendation System')

Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                   GlobalMaxPool2D()
                                   ])
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader('Uploaded Image')
    st.image(upload_file)
    input_img_features = extract_features_from_images(upload_file, model)
    distance,indices = neighbors.kneighbors([input_img_features])
    st.subheader('Recommended Images')
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])"""
"""import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st
import kagglehub

# Download the latest dataset version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print("Path to dataset files:", path)

st.header('Fashion Recommendation System')

# Load precomputed image features and filenames
Image_features = pkl.load(open('images_features.pkl', 'rb'))
filenames = pkl.load(open('images_features.pkl', 'rb'))

# Function to extract features from a single image using the pre-trained model
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Initialize ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Fit nearest neighbors on the precomputed image features
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Define the upload folder path within the Downloads directory
upload_folder = os.path.join(os.path.expanduser('~'), 'Downloads', 'images')
os.makedirs(upload_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Streamlit file uploader widget
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    # Save the uploaded file to the Downloads/images folder
    upload_path = os.path.join(upload_folder, upload_file.name)
    with open(upload_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features from the uploaded image
    input_img_features = extract_features_from_images(upload_path, model)
    
    # Find the nearest neighbors based on the uploaded image's features
    distances, indices = neighbors.kneighbors([input_img_features])
    
    st.subheader('Recommended Images')
    
    # Display recommended images in a row of columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])"""

"""import os
import numpy as np
import sqlite3
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st

# Define paths
images_directory = '/Users/ayushyadv/Downloads/Fashion_Recom_Model/images'
db_path = '/Users/ayushyadv/Downloads/Fashion_Recom_Model/images_features.db'

# Initialize Streamlit app
st.header('Fashion Recommendation System')

# Load ResNet50 model for feature extraction
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
    return model

model = load_model()

# Function to extract features from a single image
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_expand_dim)
    features = model.predict(img_preprocessed).flatten()
    return features / norm(features)

# Save image features to the SQLite database
def save_features_to_db(images_directory, db_path, model):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS image_features (filename TEXT, feature BLOB)")
    conn.commit()

    for img_name in os.listdir(images_directory):
        img_path = os.path.join(images_directory, img_name)
        features = extract_features(img_path, model)
        features_blob = pkl.dumps(features)
        cursor.execute("INSERT INTO image_features (filename, feature) VALUES (?, ?)", (img_name, features_blob))
    conn.commit()
    conn.close()

# Load features and filenames from SQLite database
def load_features_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, feature FROM image_features")
    records = cursor.fetchall()
    conn.close()
    
    filenames = []
    image_features = []
    for filename, feature_blob in records:
        filenames.append(os.path.join(images_directory, filename))
        image_features.append(pkl.loads(feature_blob))
    
    return np.array(image_features), filenames

# Initialize or load data from the database
Image_features, filenames = None, None
try:
    Image_features, filenames = load_features_from_db(db_path)
    if len(Image_features) == 0:
        st.info("Database is empty. Populating with image features...")
        save_features_to_db(images_directory, db_path, model)
        Image_features, filenames = load_features_from_db(db_path)
except Exception as e:
    st.error("Failed to load or initialize database. Error: " + str(e))

# Check if features are loaded properly
if Image_features is None or len(Image_features) == 0:
    st.error("No image features loaded. Please check the database and image paths.")
else:
    # Fit the nearest neighbors model on image features
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(Image_features)

    # Streamlit file uploader for user-uploaded image
    upload_file = st.file_uploader("Upload Image")
    if upload_file is not None:
        upload_path = os.path.join('/Users/ayushyadv/Downloads/Fashion_Recom_Model/uploads', upload_file.name)
        with open(upload_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        st.subheader('Uploaded Image')
        st.image(upload_file)

        # Extract features from the uploaded image
        input_img_features = extract_features(upload_path, model)
        
        # Find nearest neighbors
        distances, indices = neighbors.kneighbors([input_img_features])

        st.subheader('Recommended Images')
        
        # Display recommended images
        cols = st.columns(5)
        for i in range(1, 6):
            cols[i - 1].image(filenames[indices[0][i]])"""

import os
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
import streamlit as st

# Path to the images folder and database file
images_folder = '/Users/ayushyadv/Downloads/Fashion_Recom_Model/images'
db_path = '/Users/ayushyadv/Downloads/Fashion_Recom_Model/fashion_images.db'

# Function to create SQLite database and store image features
def create_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table for image features
    cursor.execute('''CREATE TABLE IF NOT EXISTS images
                    (id INTEGER PRIMARY KEY,
                     filename TEXT NOT NULL,
                     features BLOB NOT NULL)''')
    
    conn.commit()
    conn.close()

# Function to insert image features into the database
def insert_image_features(filename, features):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('INSERT INTO images (filename, features) VALUES (?, ?)', (filename, features))
    
    conn.commit()
    conn.close()

# Function to extract features from a single image using the pre-trained model
def extract_features_from_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Initialize ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Function to populate the database with image features
def populate_db():
    create_db()
    print(f"Populating database with features from images in: {images_folder}")
    
    # Debugging: List the images being processed
    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        
        if os.path.isfile(image_path):
            print(f"Processing: {filename}")
            features = extract_features_from_image(image_path, model)
            insert_image_features(filename, features)
            print(f"Features inserted for {filename}")

# Function to display recommendations based on the uploaded image
def display_recommendations(uploaded_image_path):
    # Extract features from the uploaded image
    input_img_features = extract_features_from_image(uploaded_image_path, model)
    
    # Connect to the database and find nearest neighbors
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all the stored image features and filenames
    cursor.execute('SELECT filename, features FROM images')
    all_images = cursor.fetchall()
    
    distances = []
    filenames = []
    
    for img in all_images:
        stored_features = np.frombuffer(img[1], dtype=np.float32)  # Convert from BLOB back to numpy array
        distance = np.linalg.norm(stored_features - input_img_features)
        distances.append(distance)
        filenames.append(img[0])
    
    # Sort by distance and get the top 5 closest images
    recommendations = sorted(zip(distances, filenames))[:5]
    
    st.subheader('Recommended Images')
    
    # Create columns for displaying the recommended images
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Display recommended images in a row of columns
    with col1:
        st.image(os.path.join(images_folder, filenames[0]), caption=f'{filenames[0]} - {distances[0]:.2f}')
    with col2:
        st.image(os.path.join(images_folder, filenames[1]), caption=f'{filenames[1]} - {distances[1]:.2f}')
    with col3:
        st.image(os.path.join(images_folder, filenames[2]), caption=f'{filenames[2]} - {distances[2]:.2f}')
    with col4:
        st.image(os.path.join(images_folder, filenames[3]), caption=f'{filenames[3]} - {distances[3]:.2f}')
    with col5:
        st.image(os.path.join(images_folder, filenames[4]), caption=f'{filenames[4]} - {distances[4]:.2f}')
    
    conn.close()

# Streamlit Interface
st.header('Fashion Recommendation System')

# Button to populate the database with features (Run this only once)
if st.button('Populate Database'):
    populate_db()
    st.write("Database populated with image features.")

# File uploader to upload a new image
upload_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Save the uploaded file to the images folder
    uploaded_image_path = os.path.join(images_folder, upload_file.name)
    
    with open(uploaded_image_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.image(upload_file, caption="Uploaded Image", use_column_width=True)
    
    # Show recommendations based on the uploaded image
    display_recommendations(uploaded_image_path)

