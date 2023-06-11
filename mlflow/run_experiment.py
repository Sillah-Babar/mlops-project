import numpy as np
import datetime
import os
from scipy import ndimage
import skvideo.io
import matplotlib.pyplot as plt
import mediapipe as mp
import  cv2
from keras import backend as K

import numpy as np
import re
import string
from collections import Counter


import editdistance

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    print("decode: ",decoded)
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    logprobs  = decoded[1].eval(session=K.get_session())

    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []
        for output in decoded:
            out = output
            for postprocessor in self.postprocessors:
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed

# Source: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

# Source: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# Source: http://norvig.com/spell-correct.html (with some modifications)
class Spell(object):
    def __init__(self, path):
        self.dictionary = Counter(list(string.punctuation) + self.words(open(path,encoding='utf-8').read()))

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        return untokenize(self.corrections(tokenize(sentence)))




#helpers
def text_to_labels(text):
    letters=['ں','آ','ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ','ع', 'غ', 'ف', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ﮩ', 'ﮨ', 'ھ', 'ء', 'ی', 'ے']
    #unicode value for the letters in Urdu
    unicode=[1722,1570,1575, 1576, 1662, 1578, 1657, 1579, 1580, 1670, 1581, 1582, 1583, 1672, 1584, 1585, 1586, 1688, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1601, 1602, 1705, 1711, 1604, 1605, 1606, 1608, 1729, 64425, 64424, 1726, 1569, 1740,1746]
    #mapping each unicode value to a numerical value that is linearly increasing
    maps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31, 32, 33, 34, 35, 36, 37, 38,39,40]
    ret = []
    # digits=[]
    # order
    for char in text:
        if char == ' ': #the numerical values for a space
            ret.append(41)
        else:
            for m in range(len(unicode)):
                if ord(char)==unicode[m]:
                    ret.append(maps[m])
    return ret

def labels_to_text(labels):
    letters=['ں','آ','ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ','ع', 'غ', 'ف', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ﮩ', 'ﮨ', 'ھ', 'ء', 'ی', 'ے']
    unicode=[1722,1570,1575, 1576, 1662, 1578, 1657, 1579, 1580, 1670, 1581, 1582, 1583, 1672, 1584, 1585, 1586, 1688, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1601, 1602, 1705, 1711, 1604, 1605, 1606, 1608, 1729, 64425, 64424, 1726, 1569, 1740,1746]
    maps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31, 32, 33, 34, 35, 36, 37, 38,39,40]
    ret = []
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c==41:
            text += ' '
        else:
            for i in range(len(maps)):
                if(c==maps[i]):
                    text+=letters[i]
    return text
import zarr
def Save_files_as_zarr(data,count,directory):
  home=r'preprocessedDataset\\'
  zarr.save(home+'\\'+directory+'_zarr\\'+count+'\\the_input.zarr',data[0]['the_input'])
  zarr.save(home+'\\'+directory+'_zarr\\'+count+'\\the_labels.zarr',data[0]['the_labels'])
  zarr.save(home+'\\'+directory+'_zarr\\'+count+'\\the_input_length.zarr',data[0]['input_length'])
  zarr.save(home+'\\'+directory+'_zarr\\'+count+'\\the_label_length.zarr',data[0]['label_length'])
  zarr.save(home + '\\' + directory + '_zarr\\' + count + '\\the_source.zarr', data[0]['source_str'])
  zarr.save(home+'\\'+directory+'_zarr\\'+count+'\\ctc.zarr',data[1]['ctc'])

def Load_files_from_zarr(count,directory):
  home = r'preprocessedDataset\\'
  input = zarr.load(home+'\\'+directory+'_zarr\\'+count+'\\the_input.zarr')
  label = zarr.load(home+'\\'+directory+'_zarr\\'+count+'\\the_labels.zarr')
  input_length = zarr.load(home+'\\'+directory+'_zarr\\'+count+'\\the_input_length.zarr')
  label_length = zarr.load(home+'\\'+directory+'_zarr\\'+count+'\\the_label_length.zarr')
  ctc = zarr.load(home+'\\'+directory+'_zarr\\'+count+'\\ctc.zarr')
  source_str = zarr.load(home + '\\' + directory + '_zarr\\' + count + '\\the_source.zarr')
  inputs = {'the_input': input,
                  'the_labels': label,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
            }
  outputs = {'ctc': ctc}  # dummy data for dummy loss function
  return(inputs,outputs)


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

# Actual loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

from keras.layers.core import Lambda
from keras.layers import Input

# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra prams on loss function)
def CTC(name, args):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)
 


def build(img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=29, output_size=43):
        if K.image_data_format() == 'channels_first':
            input_shape = (img_c, frames_n, img_w, img_h)
        else:
            input_shape = (frames_n, img_w, img_h, img_c)

        input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
        conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(zero1)
        batc1 = BatchNormalization(name='batc1')(conv1)
        actv1 = Activation('relu', name='actv1')(batc1)
        drop1 = SpatialDropout3D(0.5)(actv1)
        maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(drop1)

        zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(maxp1)
        conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(zero2)
        batc2 = BatchNormalization(name='batc2')(conv2)
        actv2 = Activation('relu', name='actv2')(batc2)
        drop2 = SpatialDropout3D(0.5)(actv2)
        maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(drop2)

        zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(maxp2)
        conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(zero3)
        batc3 = BatchNormalization(name='batc3')(conv3)
        actv3 = Activation('relu', name='actv3')(batc3)
        drop3 = SpatialDropout3D(0.5)(actv3)
        maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(drop3)

        resh1 = TimeDistributed(Flatten())(maxp3)

        gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
        gru_2 = Bidirectional(GRU(256, return_sequences=True , kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

        # transforms RNN output to character activations:
        dense1 = Dense(output_size, kernel_initializer='he_normal', name='dense1')(gru_2)

        y_pred = Activation('softmax', name='softmax')(dense1)
        print("self.y_pred: ",y_pred)

        labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
        print("self.y_pred: ",labels)
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC('ctc', [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        return model


import matplotlib
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
tf.compat.v1.experimental.output_all_intermediates(True)
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint




# new_data=None
# for i in range(1):
#   print(i+1)
#   data=Load_files_from_zarr(str(i+1),'0')
#   if(i==0):
#     new_data=data
#   else:
#     arr1=data[0]['the_input']
#     arr2=new_data[0]['the_input']
#     joined=np.concatenate((arr1,arr2),axis=0)
#     joined2=np.concatenate((data[0]['the_labels'],new_data[0]['the_labels']),axis=0)
#     joined3=np.concatenate((data[0]['input_length'],new_data[0]['input_length']),axis=0)
#     joined4=np.concatenate((data[0]['label_length'],new_data[0]['label_length']),axis=0)
#     joined5=np.concatenate((data[1]['ctc'],new_data[1]['ctc']),axis=0)
#     new_data[0]['the_input']=joined
#     new_data[0]['the_labels']=joined2
#     new_data[0]['input_length']=joined3
#     new_data[0]['label_length']=joined4
#     new_data[1]['ctc']=joined5
    

# val_data=None
# for i in range(1):
#   print(i+1)
#   data=Load_files_from_zarr(str(i+1),'0')
#   if(i==0):
#     val_data=data
#   else:
#     arr1=data[0]['the_input']
#     arr2=val_data[0]['the_input']
#     joined=np.concatenate((arr1,arr2),axis=0)
#     joined2=np.concatenate((data[0]['the_labels'],val_data[0]['the_labels']),axis=0)
#     joined3=np.concatenate((data[0]['input_length'],val_data[0]['input_length']),axis=0)
#     joined4=np.concatenate((data[0]['label_length'],val_data[0]['label_length']),axis=0)
#     joined5=np.concatenate((data[1]['ctc'],val_data[1]['ctc']),axis=0)
#     val_data[0]['the_input']=joined
#     val_data[0]['the_labels']=joined2
#     val_data[0]['input_length']=joined3
#     val_data[0]['label_length']=joined4
#     val_data[1]['ctc']=joined5

# print(new_data[0]['the_input'].shape)
# print(new_data[0]['the_labels'].shape)
# print(new_data[0]['input_length'].shape)
# print(new_data[0]['label_length'].shape)
# print(new_data[1]['ctc'].shape)

# print(val_data[0]['the_input'].shape)
# print(val_data[0]['the_labels'].shape)
# print(val_data[0]['input_length'].shape)
# print(val_data[0]['label_length'].shape)
# print(val_data[1]['ctc'].shape)



import numpy as np
import mlflow
import keras
from jiwer import cer,wer
import os
new_data=None
for i in range(1):
  print(i+1)
  data=Load_files_from_zarr(str(i+1),'0')
  if(i==0):
    new_data=data
  else:
    arr1=data[0]['the_input']
    arr2=new_data[0]['the_input']
    joined=np.concatenate((arr1,arr2),axis=0)
    joined2=np.concatenate((data[0]['the_labels'],new_data[0]['the_labels']),axis=0)
    joined3=np.concatenate((data[0]['input_length'],new_data[0]['input_length']),axis=0)
    joined4=np.concatenate((data[0]['label_length'],new_data[0]['label_length']),axis=0)
    joined5=np.concatenate((data[1]['ctc'],new_data[1]['ctc']),axis=0)
    new_data[0]['the_input']=joined
    new_data[0]['the_labels']=joined2
    new_data[0]['input_length']=joined3
    new_data[0]['label_length']=joined4
    new_data[1]['ctc']=joined5

def run_experiment(new_data,batch_size_,eps,learning_rate,beta1,beta2):
    # plot graph

    adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta1, epsilon=1e-08)

    model=build()
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])

    his=model.fit(
        x=new_data[0],
        y=new_data[1],
        batch_size=batch_size_,
        epochs=eps,
        initial_epoch=0,
        max_queue_size=10,
        workers=8,
        #validation_data=(val_data[0],val_data[1]),
        #use_multiprocessing=False,callbacks = [es,mc]
    )    
    return model,{'batch_size': batch_size_,'epochs':eps,'learning_rate':learning_rate,'beta_1':beta1,'beta_2':beta2}

def evaluate(model,new_data):        
    prediction_model = keras.models.Model(
        model.get_layer(name="the_input").input, model.get_layer(name="softmax").output
    )

    PREDICT_GREEDY      = False
    PREDICT_BEAM_WIDTH  = 200
    PREDICT_DICTIONARY  = os.path.join('dictionaries','urdu_sentences.txt')
    absolute_max_string_len=29
    output_size=43

    count_vids=0

    for i in range(len(new_data[0]['the_input'])):
        wer_error=0
        cer_error=0
        print(i)
        spell = Spell(path=PREDICT_DICTIONARY)
        decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                            postprocessors=[labels_to_text, spell.sentence])

        X_data       = np.array([new_data[0]['the_input'][i]]).astype(np.float32) 
        input_length = np.array([len(new_data[0]['the_input'][i])])
    
        y_pred         = prediction_model.predict(X_data)
        #print(y_pred.shape)
        result         = decoder.decode(y_pred, input_length)
        #new_res=tokenizer(result[0])
        new_res=result[0]
        referen=new_data[0]['source_str'][i]
        #referen=vid_path
        print("Reference: ",referen)
        print("Result: ",new_res)

        error = wer(referen, new_res)
        err = cer(referen, new_res)
        print("wer error: ",error)
        print("cer error: ",err)
        wer_error+=error
        cer_error+=err
        count_vids+=1


        return wer_error/float(count_vids) , cer_error/float(count_vids)

def log_experiment(experiment_name,run_name, wer_err,cer_err,model ,run_params=None):
   
    #mlflow.set_tracking_uri("http://localhost:5000") #uncomment this line if you want to use any database like sqlite as backend storage for model
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
  
        mlflow.log_metric("wer",wer_err)
        mlflow.log_metric("cer",cer_err)
        
       # mlflow.sklearn.log_model(model, "model")
        
    
            
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
    
    #run different experiments
batch_size_=[8,16,32]
learning_rate=[0.001,0.0001,0.00001]
eps=[1,2,3]
beta1=[0.2,0.5,0.7]
beta2=[0.1,0.2,0.3]
print(new_data)
for i in range(3):
    print(i)
    model,params=run_experiment(new_data,batch_size_[i],eps[i],learning_rate[i],beta1[i],beta2[i])
    wer_err,cer_err=evaluate(model,new_data)
    for param in params:
        print(params[param])
    log_experiment('lipnet',str(i), wer_err,cer_err,model ,params)