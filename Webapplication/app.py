from flask import Flask, render_template, request
from flask_mail import Mail, Message
import smtplib
import io
import numpy as np
from PIL import Image
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets


app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/send-email', methods=['POST'])
def send_email():
 
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    
    subject = 'Contact Form Submission from ' + name
    body = 'Name: ' + name + '\nEmail: ' + email + '\nMessage: ' + message

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('zahidulislam2225@gmail.com', 'valb mmmn awhg snpd')
    
    server.sendmail('zahidulislam2225@gmail.com', 'rafin3600@gmail.com', subject + '\n\n' + body)
    server.quit()

    return render_template('thank-you.html')

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

def load_content_img(image_pixels):
    if image_pixels.shape[-1] == 4:
        image_pixels = Image.fromarray(image_pixels)
        img = image_pixels.convert('RGB')
        img = np.array(img)
        img = tf.convert_to_tensor(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 3:
        img = tf.convert_to_tensor(image_pixels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 1:
        raise Error('Grayscale images not supported! Please try with RGB or RGBA images.')
    print('Exception not thrown')

def preprocess_image(image, target_dim):
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image


style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3-dynamic-shapes/int8/predict/1?lite-format=tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3-dynamic-shapes/int8/transfer/1?lite-format=tflite')



def run_style_predict(preprocessed_style_image):
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck


@app.route('/', methods=['GET', 'POST'])
def stylize_image():
    if request.method == 'POST':
        STYLE_IMAGE_NAME = request.form['style_image_name']
      
        corresponding_url = {
            'IMAGE_1': 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
            'IMAGE_2': 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg',
            'IMAGE_3': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1024px-Tsunami_by_hokusai_19th_century.jpg',
            'IMAGE_4': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
            'IMAGE_5': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/757px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
            'IMAGE_6': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/220px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg',
	    'IMAGE_7': 'https://images.squarespace-cdn.com/content/v1/5511fc7ce4b0a3782aa9418b/1429331653608-5VZMF2UT2RVIUI4CWQL9/abstract-art-style-by-thaneeya.jpg',
	    'IMAGE_8': 'https://www.artmajeur.com/medias/standard/l/a/laurent-folco/artwork/14871329_a75fb86e-1a71-4559-a730-5cd4df09f0c4.jpg',
	    'IMAGE_9': 'https://s3.amazonaws.com/gallea.arts.bucket/e36461e0-551c-11eb-b1d7-c544bb4e051b.jpg',
	    'IMAGE_10': 'https://www.homestratosphere.com/wp-content/uploads/2019/10/Raster-painting-example-woman-oct16.jpg',
	    'IMAGE_11': 'https://i.ibb.co/jLB89J6/IMAGE-11.jpg',
	    'IMAGE_12': 'https://static01.nyt.com/images/2020/10/23/arts/21lawrence/21lawrence-superJumbo.jpg'
        }

        style_image_path = tf.keras.utils.get_file(
            STYLE_IMAGE_NAME + ".jpg", corresponding_url[STYLE_IMAGE_NAME])
        global content_image
        content_image = Image.open(io.BytesIO(request.files['content_image'].read()))
        img = content_image.convert('RGB')
        img.thumbnail((256, 256))
        img.save('content.jpg')
        content_image = np.array(content_image)

        content_image = load_content_img(content_image)
        style_image = load_img(style_image_path)

        content_blending_ratio = -0.33
        content_image_size = 512
        def run_style_transform(style_bottleneck, preprocessed_content_image):
          interpreter = tf.lite.Interpreter(model_path=style_transform_path)
          input_details = interpreter.get_input_details()
          for index in range(len(input_details)):
            if input_details[index]["name"]=='content_image':
              index = input_details[index]["index"]
              interpreter.resize_tensor_input(index, [1, content_image_size, content_image_size, 3])
          interpreter.allocate_tensors()
          for index in range(len(input_details)):
            if input_details[index]["name"]=='Conv/BiasAdd':
              interpreter.set_tensor(input_details[index]["index"], style_bottleneck)
            elif input_details[index]["name"]=='content_image':
              interpreter.set_tensor(input_details[index]["index"], preprocessed_content_image)
          interpreter.invoke()
          stylized_image = interpreter.tensor(
              interpreter.get_output_details()[0]["index"]
              )()
          return stylized_image
        preprocessed_content_image = preprocess_image(content_image, content_image_size)
        preprocessed_style_image = preprocess_image(style_image, 256)
        def imshow(image, title=None):
          if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
            plt.imshow(image)
            if title:
              plt.title(title)
        def tensor_to_image(tensor):
          tensor = tensor*255
          tensor = np.array(tensor, dtype=np.uint8)
          if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
          return PIL.Image.fromarray(tensor)  
        style_bottleneck = run_style_predict(preprocessed_style_image)
        style_bottleneck_content = run_style_predict(preprocess_image(content_image, 256))
        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck
        stylized_image = run_style_transform(
            style_bottleneck_blended, preprocessed_content_image)
        stylized_image = tensor_to_image(stylized_image)
        stylized_image.save('static/NST_image.jpeg')
        
        return render_template('result.html', stylized_image="static/NST_image.jpeg")


if __name__ == '__main__':
    app.run(debug=True)
