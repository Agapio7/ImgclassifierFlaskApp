from flask import Flask,render_template,request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# from keras.applications.imagenet_utils import decode_predictions


# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np






app = Flask(__name__)
model = ResNet50(weights='imagenet')

@app.route("/" ,methods=['GET'])
def project():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1],label[2]*100)


    return render_template('index.html',prediction=classification)


if __name__=='__main_':
    app.route(port=4000,debug=True)
