from flask import Flask, request, render_template, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import load_img, img_to_array


app = Flask(__name__)

# Skin Classification Preprocessing
def skin_preprossing(image):
    image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr


skin_model = load_model('Augmentation_weights_model.h5')

# ================================================================================


# MPC Classification Preprocessing
def MPC_preprossing(image):
    image=Image.open(image)
    image = image.resize((200, 200))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 200, 200, 3)
    return image_arr

MPC_model = load_model('MPC_VGG16.h5')

# ================================================================================


# LGP Classification Preprocessing
def LGP_preprossing(image):
    image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

LGP_model = load_model('LGP_Augmentation_weights_model.h5')




@app.route('/')
def index():

    return render_template('index.html', appName="Skin Diseases Classification")


#  Skin Classification Diseases
@app.route('/predict_skin_Api', methods=["POST"])
def skin_api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= skin_preprossing(image)
        print("predicting ...")
        new_predict = skin_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        skin_classes = {
            0: 'Allergy', 
            1: 'Bacteria'
        }

        # print(classes[Class])
        prediction = skin_classes[Class]
        # print("Model predicting ...")
        # result = model.predict(image_arr)
        # print("Model predicted")
        # Class = np.round(result).astype('int32')
        # prediction = classes[Class]
        # print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict_skin', methods=['GET', 'POST'])
def skin_predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= skin_preprossing(img)
        print("predicting ...")
        new_predict = skin_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        skin_classes = {
            0: 'Allergy', 
            1: 'Bacteria'
        }

        # print(classes[Class])
        prediction = skin_classes[Class]

        return render_template('index.html', prediction=prediction, appName="Skin Diseases Classification")
    else:
        return render_template('index.html',appName="Skin Diseases Classification")





# ============================================================================================




# MPC Classification 

@app.route('/predict_MPC_Api', methods=["POST"])
def MPC_api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= MPC_preprossing(image)
        print("predicting ...")
        new_predict = MPC_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        MPC_classes = {
            0: 'Normal', 
            1: 'MPC'
        }

        # print(classes[Class])
        prediction = MPC_classes[Class]
        # print("Model predicting ...")
        # result = model.predict(image_arr)
        # print("Model predicted")
        # Class = np.round(result).astype('int32')
        # prediction = classes[Class]
        # print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict_MPC', methods=['GET', 'POST'])
def MPC_predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= MPC_preprossing(img)
        print("predicting ...")
        new_predict = MPC_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        MPC_classes = {
            0: 'Normal', 
            1: 'MPC'
        }

        # print(classes[Class])
        prediction = MPC_classes[Class]

        return render_template('index.html', prediction=prediction, appName="Skin Diseases Classification")
    else:
        return render_template('index.html',appName="Skin Diseases Classification")




# ============================================================================================



# LGP
@app.route('/predict_LGP_Api', methods=["POST"])
def LGP_api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= LGP_preprossing(image)
        print("predicting ...")
        new_predict = LGP_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        LGP_classes = {
            0: 'LGP', 
            1: 'Normal'
        }

        # print(classes[Class])
        prediction = LGP_classes[Class]
        # print("Model predicting ...")
        # result = model.predict(image_arr)
        # print("Model predicted")
        # Class = np.round(result).astype('int32')
        # prediction = classes[Class]
        # print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict_LGP', methods=['GET', 'POST'])
def LGP_predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= LGP_preprossing(img)
        print("predicting ...")
        new_predict = LGP_model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        LGP_classes = {
            0: 'Allergy', 
            1: 'Bacteria'
        }

        # print(classes[Class])
        prediction = LGP_classes[Class]

        return render_template('index.html', prediction=prediction, appName="Skin Diseases Classification")
    else:
        return render_template('index.html',appName="Skin Diseases Classification")



if __name__ == '__main__':
    app.run(debug=True)