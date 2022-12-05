import os
import numpy as np
import base64
import cv2

from keras.applications import vgg16
from tensorflow.keras.utils import img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    model = None

    def __init__(self):
        self.model = vgg16.VGG16(weights='imagenet')
        self.model = Model(inputs=self.model.input, outputs=self.model.get_layer("fc2").output)

    def get_model(self):
        if not self.model:
            raise Exception("Model not implemented")
        return self.model
    
    def predict_similarity(self, query):
        """ Predict similarity of features
        """
        model = self.get_model()
        processed_imgs = self.prepocess(query)
        imgs_features = model.predict(processed_imgs)
        cosSimilarities = cosine_similarity([imgs_features[0]], [imgs_features[1]])
        return cosSimilarities[0]

    def prepocess(self, images):
        """ Preprocessing of input images
        """
        imported_images = []
        for key in images:
            image = self.get_image(images[key])
            image = cv2.resize(image, (224,224), interpolation = cv2.INTER_NEAREST)
            numpy_image = img_to_array(image)
            image_batch = np.expand_dims(numpy_image, axis=0)
            imported_images.append(image_batch)
        images = np.vstack(imported_images)
        processed_imgs = preprocess_input(images.copy())
        return processed_imgs

    def get_image(self, image_txt):
        """ Decoding base64 image
        """
        nparr = np.fromstring(base64.b64decode(image_txt), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
