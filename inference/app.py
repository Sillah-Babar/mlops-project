from distutils.log import debug
from fileinput import filename

import flask
from flask import *
import numpy 
from inference import Video,Decoder,labels_to_text,Spell,tokenizer,token,ctc_lambda_func,CTC

import os
import matplotlib
import tensorflow as tf
import keras
tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.core import Lambda
from scipy import ndimage
import numpy as np
import sys
import os
from keras import backend as K
import numpy as np
import math

app = Flask(__name__)



def inference(file):
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model= keras.models.load_model("weights\\best_val_model1.h5",compile=False)
    prediction_model = keras.models.Model(model.get_layer(name="the_input").input, model.get_layer(name="softmax").output)

	#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])
    #new_result = ""
    
    CURRENT_PATH = os.path.dirname('')
    PREDICT_GREEDY      = False
    PREDICT_BEAM_WIDTH  = 200
    PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'dictionaries','urdu_sentences.txt')
    print("predict dictionary: ",PREDICT_DICTIONARY )
    absolute_max_string_len=29
    output_size=43

    video=Video().from_frames(file)
    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                          postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) /255
    input_length = np.array([len(video.data)])
  
    y_pred         = prediction_model.predict(X_data)
    result         = decoder.decode(y_pred, input_length)
    new_result=result[0]
	#new_result='میں'
	#new_result=token(new_result)
 	#new_result = tokenizer(result[0])

    print("Result: ",new_result)

    return new_result.encode("utf-8") 
    
@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		# name = request.form['name']
		f.save(f.filename)
		print(f.filename)
		# print(name)
		result=inference(f.filename)
		return result

if __name__ == '__main__':
	app.run(debug=True)
