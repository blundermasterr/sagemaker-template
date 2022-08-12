import tensorflow as tf
import os
import PIL
from numpy import asarray
from PIL import Image

if __name__ == "__main__":
    input_directory = "./opt/ml/processing/input/"
    output_directory = "./opt/ml/processing/output/"
    print("--------------STARTING----------------")
    print(f'---------INPUT DIRECTORY : {input_directory} -----------')
    print(f'---------OUTPUT DIRECTORY : {output_directory} ----------')
    #input_directory = "../input_images/"
    #output_directory = "../output_images/"
    
    for _,_,files in os.walk(input_directory):
        for name in files:
            if(name.endswith(".png")):
                image_path = os.path.join(input_directory,name)
                image = Image.open(image_path)
                image = asarray(image)
                inverted_image = tf.bitwise.invert(image)
                inverted_image = inverted_image.numpy()
                inverted_image = Image.fromarray(inverted_image)
                inverted_image = inverted_image.save(os.path.join(output_directory,name))