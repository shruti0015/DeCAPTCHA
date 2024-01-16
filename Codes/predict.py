
import pickle
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def decaptcha( filenames ):
    
  
    def vanish_lines(image, color_range=80):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv_image[:,:,2]
        min_val = np.min(v)
        mask = cv2.inRange(v, np.array(min_val - color_range), np.array(min_val + color_range))
        result_image = cv2.bitwise_and(image, image, mask=mask)
        result_image[np.where(mask == 0)] = 255
        return result_image


    def segment_last_letter_bw(image):

        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        
        last_contour = contours[-1]
        x, y, w, h = cv2.boundingRect(last_contour)
        last_letter = gray_image[y:y+h, x:x+w]

        
        _, last_letter_bw = cv2.threshold(last_letter, 127, 255, cv2.THRESH_BINARY)

        return last_letter_bw


    
    def process_image(filename):
        if filename.endswith('.png'):
            img_path = filename
            img = cv2.imread(img_path)
            img = vanish_lines(img)
            img = segment_last_letter_bw(img)
            img = cv2.resize(img, (50, 50))
            return img

    a = 1
    
    files = filenames
    
    resized_images = Parallel(n_jobs=-1)(delayed(process_image)(filename) for filename in files)

    
    X_test = np.array([img for img in resized_images if img is not None])

    


   
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    y_pred = model.predict(X_test_2d)
    return y_pred