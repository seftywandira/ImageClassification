#
# Classify cats and dogs
# James
#

import os
import shutil
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

#
# Paths to the folders
#
#
input_folder = 'images'
# input_folder = 'bird/yes'
cat_folder   = 'images/cats'
dog_folder   = 'images/dogs'

#
# Load pre-trained ResNet50 model + higher level layers
#
model = ResNet50(weights='imagenet')

#
# List of known dog classes
#
cat_classes = [
    'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat', 
    'cougar', 'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion', 'cheetah'
]
cat_classes = [element.lower() for element in cat_classes]

dog_classes = [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 
    'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 
    'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 
    'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 
    'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 
    'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 
    'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 
    'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 
    'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 
    'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 
    'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 
    'curly-coated_retriever', 'Golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 
    'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 
    'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 
    'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 
    'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 
    'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 
    'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 
    'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 
    'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 
    'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 
    'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 
    'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
]
dog_classes = [element.lower() for element in dog_classes]

# Function to classify an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return decode_predictions(preds, top=1)[0][0][1]  # returns the class name

#
# Process each image in the input folder
#
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.JPG', '.jpeg', '.png')):  # Check if the file is an image
        img_path = os.path.join(input_folder, filename)
        print(f'Processing {filename}...')
        
        # Classify the image
        classification = classify_image(img_path)
        print(classification)
        classification = classification.lower()
        # Move the image to the appropriate folder
        if classification in cat_classes:
            shutil.move(img_path, os.path.join(cat_folder, filename))
            print(f'{filename} moved to {cat_folder}')
        elif classification in dog_classes:
            shutil.move(img_path, os.path.join(dog_folder, filename))
            print(f'{filename} moved to {dog_folder}')
        else:
            print(f'{filename} is not a cat or dog.')

print('Processing complete.')