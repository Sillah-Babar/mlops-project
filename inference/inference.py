import numpy as np
import datetime
import os
from scipy import ndimage
import skvideo.io
import mediapipe as mp
import  cv2
from keras import backend as K
from moviepy.editor import *

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

class Align(object):
    def __init__(self, absolute_max_string_len=29, label_func=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()

        list_of_strings=[]
        for i in range(len(lines)):

            lines[i]=lines[i].strip()
           # print("linesss: ",lines[i])
            new_array = lines[i].split(' ')
            if(new_array[0]!=''):
                list_of_strings.append(new_array[0])
                
        #code to safe the start and end of the video after removing the silence
        if (len(lines) > 2):
            length = len(lines)
            firstline = lines[0].split(' ')
            lastline = lines[length - 1].split(' ')
            last=lastline[0]
            last=last[1:len(last)]
            self.start = firstline[2]
            self.end = last
          
        self.build(list_of_strings)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        # self.align = self.strip(align, ['sp','<sil>'])
        print(align[1:7])
        self.align=align[1:7]
        for i in range(len(self.align)) :
            print(self.align[i])
        self.sentence = self.get_sentence(align[1:7])
        print('sentence: ',self.sentence)
        self.label = self.get_label(self.sentence)
        print('label: ',self.label)
        self.padded_label = self.get_padded_label(self.label)
        # print("Sentencce: ",self.sentence)
        # print("label: ", self.label)
        # print("padded_label: ", self.padded_label)
        # print("padded_label len: ", len(self.label))
    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        sentence=''
        for i in range(len(self.align)) :
            if(i<len(self.align)-1):
                sentence+=self.align[i]+" "
            else:
                sentence += self.align[i]


        return sentence

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)


import math

class Video(object):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def getLipLandmarks(self, results):
        index = 0
        lips_x = []
        lips_y = []
        for lip_landmarks in results.landmark:
            # print("x: ",lip_landmarks.x)
            # print("y: ",lip_landmarks.y)
            if (index == 61 or index == 76 or index == 62 or index == 146 or index == 77 or index == 96
                    or index == 95 or index == 191 or index == 183 or index == 184 or index == 185 or index == 78
                    or index == 306 or index == 292 or index == 291 or index == 308 or index == 324 or index == 415
                    or index == 407 or index == 408 or index == 409 or index == 325 or index == 307 or index == 375
                    or index == 40 or index == 74 or index == 42 or index == 80 or index == 88 or index == 89 or index == 90
                    or index == 91 or index == 39 or index == 73 or index == 41 or index == 81 or index == 178 or index == 179
                    or index == 180 or index == 181 or index == 37 or index == 72 or index == 38 or index == 82 or index == 87
                    or index == 86 or index == 85 or index == 84 or index == 0 or index == 11 or index == 12 or index == 13
                    or index == 14 or index == 15 or index == 16 or index == 17 or index == 267 or index == 302 or index == 268
                    or index == 312 or index == 317 or index == 316 or index == 315 or index == 314 or index == 269 or index == 303
                    or index == 271 or index == 311 or index == 402 or index == 403 or index == 404 or index == 405 or index == 270
                    or index == 304 or index == 272 or index == 310 or index == 318 or index == 319 or index == 320 or index == 321):
                # Putting array of landmark attributes in test (Exluding res.visibility)
                lips_x.append(lip_landmarks.x)
                lips_y.append(lip_landmarks.y)

            index += 1  # (Update)To keep track
        return lips_x, lips_y

    def ApplyModel(self, ImagePath):
        frame = ImagePath

        # frame = cv2.resize(frame, (800, 600))

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.holistic_model.process(frame)
        lips_x = []
        lips_y = []

        if (results.face_landmarks):
            lips_x, lips_y = self.getLipLandmarks(results.face_landmarks)
            self.prev_lipsx = lips_x
            self.prev_lipsy = lips_y
            MOUTH_WIDTH = 100
            MOUTH_HEIGHT = 50
            HORIZONTAL_PAD = 0.05
            VERTICAL_PAD = 0.05
            # Converting back the RGB image to BGR
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Unormalizing lip landmarks
            for i in range(len(lips_x)):
                lips_x[i] = int(lips_x[i] * frame.shape[1])
                lips_y[i] = int(lips_y[i] * frame.shape[0])

            # code to draw landmarks on lips
            # for i in range(len(lips_x)):
            # cv2.circle(frame, (lips_x[i],lips_y[i]), 1, (255, 0, 0), -1)
            # plt.imshow(frame)

            mouth_left = min(lips_x)
            mouth_right = max(lips_x)
            mouth_top = min(lips_y)
            mouth_bottom = max(lips_y)

            crop_image = frame[int(mouth_top):int(mouth_bottom), int(mouth_left):int(mouth_right)]

            newsize = (100, 50)

            resized_img = cv2.resize(crop_image, newsize)
        else:
            crop_image = frame

            newsize = (100, 50)

            resized_img = cv2.resize(crop_image, newsize)

        return resized_img

    def from_frames(self, path):
        frames = self.get_video_frames(path)
        # frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        # frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self

    def from_video(self, path,start,end):
        #print("videp pathhhh", path)
        clip = VideoFileClip(path)
        duration=clip.duration

        # clipping of the video
        # getting video for only starting 10 seconds
        clip = clip.subclip(start, end)
        start=float(start)
        end=float(end)
        print("actual duration: ", duration)
        print("start: ",start)
        print("end: ",duration-end)
        start_duration=start
        end_duration=duration-end
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        percentage_start=(start_duration/(start_duration+end_duration) )*100
        percentage_end=(end_duration/(start_duration+end_duration) )*100
        print("percentage_start: ",percentage_start)
        print("percentage_end: ",percentage_end)
        total_frames_to_remove=(len(frames))-75
        remove_start=total_frames_to_remove*(percentage_start/100.0)
        remove_end=total_frames_to_remove*(percentage_end/100.0)
        print("total to remove: ",total_frames_to_remove)
        print("num remove start:",remove_start )
        print("num remove end:", remove_end)
        more_decimal_value_start=remove_start-int(remove_start)
        more_decimal_value_end=remove_end-int(remove_end)
        num_remove_frames_start=0
        num_remove_frames_end = 0
        if(more_decimal_value_start>more_decimal_value_end):
            num_remove_frames_start=math.ceil(remove_start)
            num_remove_frames_end=math.floor(remove_end)
        else:
            num_remove_frames_start = math.floor(remove_start)
            num_remove_frames_end = math.ceil(remove_end)
        print("num_remove_frames_start: ",num_remove_frames_start)
        print("num_remove_frames_end: ",num_remove_frames_end)
        frames=frames[num_remove_frames_start:75+num_remove_frames_start]
        print("processed frames length: ",len(frames))

        print("videp pathhhh", path)
        # clip = VideoFileClip(path)
        #
        # # clipping of the video
        # # getting video for only starting 10 seconds
        # clip = clip.subclip(start, end)
        #
        # # showing clip
        # #clip.ipython_display(width=280)
        # paths=path.split('\\')
        # new_path=paths[0:len(paths)-1]
        # filepath=new_path[0]
        # count=1
        # while(count<len(new_path)):
        #     filepath=os.path.join(filepath,new_path[count])
        #     count=count+1
        # print("paths: ",new_path)
        # newvidgeneratedpath=os.path.join(filepath,'__temp__.mp4')
        # clip.write_videofile(newvidgeneratedpath)
        #
        # frames = self.get_video_frames(newvidgeneratedpath)
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        self.process_frames_face(frames)

    def process_frames_face(self, frames):

        mouth_frames = self.get_frames_mouth(frames)
        print(len(mouth_frames))
        # if(len(mouth_frames)<75):
        #     sub=75-len(mouth_frames)
        #     odd=False
        #     new_sub = int(sub / 2)
        #     if sub%2==0:
        #         odd=False
        #     else:
        #         odd=True
        #     new_frames=[]
        #     for i in range(new_sub):
        #         new_frames.append(mouth_frames[0])
        #     for i in range(len(mouth_frames)):
        #         new_frames.append(mouth_frames[i])
        #
        #     if(odd==True):
        #         new_sub=new_sub+1
        #     for i in range(new_sub):
        #         new_frames.append(mouth_frames[len(mouth_frames)-1])
        #
        #     print("newframes: ", new_frames[0].shape)
        #     mouth_frames = new_frames
        # else:
        #     print(len(mouth_frames))
        #     mouth_frames = mouth_frames[0:75]
        #     print("........................................................THERE ARE MORE THAN 75 FRAMES........................................................................................................................................................")


        #print("Mouth frames: ",mouth_frames)
        #self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, frames):
        fname = []
        for i in range(len(frames)):
            # im_cv = cv2.imread(frames[i], cv2.IMREAD_COLOR)
            # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
            # frame2=self.edit_module(im_rgb)
            cropped_img = self.ApplyModel(frames[i])
            fname.append(cropped_img)
        return fname

    def get_video_frames(self, path):

        videogen = skvideo.io.vreader(path)

        frames = np.array([frame for frame in videogen])
        #print("frames: ",frames)
        return frames

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0, 1)  # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames)  # T x W x H x C
        #if K.image_data_format() == 'channels_first':
        #data_frames = np.rollaxis(data_frames, 3)  # C x T x W x H
        self.data = data_frames
        #print("self.data: ",self.data.shape)
        self.length = frames_n

from keras import backend as K
import numpy as np

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
import re
import string
from collections import Counter

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
        self.dictionary = Counter(list(string.punctuation) + self.words(open(path).read()))

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

import numpy as np
import editdistance
import keras
import csv
import os
import numpy
import doctest


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
# import the fuzzywuzzy module
from fuzzywuzzy import fuzz

# spellcheck main class
class SpellCheck:

    # initialization method
    def __init__(self, word_dict_file=None):
        # open the dictionary file
        self.file = open(word_dict_file, 'r')
        
        # load the file data in a variable
        data = self.file.read()
        
        # store all the words in a list
        data = data.split(",")
        
        # change all the words to lowercase
        data = [i.lower() for i in data]
        
        # remove all the duplicates in the list
        data = set(data)
        print("isdjisdh")
        
        # store all the words into a class variable dictionary
        self.dictionary = list(data)

    # string setter method
    def check(self, string_to_check):
        # store the string to be checked in a class variable
        self.string_to_check = string_to_check

    # this method returns the possible suggestions of the correct words
    def suggestions(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()

        # a list to store all the possible suggestions
        suggestions = []

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):
            
            # loop over words in the dictionary
            for name in self.dictionary:
                
                # if the fuzzywuzzy returns the matched value greater than 80
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:
                    
                    # append the dict word to the suggestion list
                    suggestions.append(name)

        # return the suggestions list
        return suggestions

    # this method returns the corrected string of the given input
    def correct(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()
        print(string_words)

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):
            
            # initiaze a maximum probability variable to 0
            max_percent = 0

            # loop over the words in the dictionary
            for name in self.dictionary:
                
                # calulcate the match probability
                percent = fuzz.ratio(string_words[i].lower(), name.lower())
                #print(percent)
                # if the fuzzywuzzy returns the matched value greater than 80
                if percent >= 80:
                    
                    # if the matched probability is
                    if percent > max_percent:
                        
                        # change the original value with the corrected matched value
                        string_words[i] = name
                    
                    # change the max percent to the current matched percent
                    max_percent = percent
        
        # return the cprrected string
        return " ".join(string_words)        


def tokenizer(string_val):
  split_words=string_val.split(' ')
  spell_check1 =  SpellCheck('/content/drive/MyDrive/spellcheck/words1.txt')
  spell_check2 =  SpellCheck('/content/drive/MyDrive/spellcheck/words2.txt')
  spell_check3 =  SpellCheck('/content/drive/MyDrive/spellcheck/words3.txt')
  spell_check4 =  SpellCheck('/content/drive/MyDrive/spellcheck/words4.txt')
  spell_check5 =  SpellCheck('/content/drive/MyDrive/spellcheck/words5.txt')
  spell_check6 =  SpellCheck('/content/drive/MyDrive/spellcheck/words6.txt')
  spell_check1.check(split_words[0])
  val1=spell_check1.correct()
  spell_check2.check(split_words[1])
  val2=spell_check2.correct()
  spell_check3.check(split_words[2])
  val3=spell_check3.correct()
  spell_check4.check(split_words[3])
  val4=spell_check4.correct()
  spell_check5.check(split_words[4])
  val5=spell_check5.correct()
  spell_check6.check(split_words[5])
  val6=spell_check6.correct()
  return val1+' '+val2+' '+val3+' '+val4+' '+val5+' '+val6


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
import re
import string
from collections import Counter

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
import re
from collections import Counter

def correct_word(word_list, target_word, n=1):
    target_ngrams = Counter([target_word[i:i+n] for i in range(len(target_word)-n+1)])
    best_match, best_match_score = None, 0
    for word in word_list:
        word_ngrams = Counter([word[i:i+n] for i in range(len(word)-n+1)])
        score = sum((word_ngrams & target_ngrams).values())
        if score > best_match_score:
            best_match, best_match_score = word, score
    return best_match

def token(value):
    split_paths=value.split(' ')
    new_result=''
    wordlist = ['ہے', 'کیسے', 'چار', 'دو', 'چھ', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نو', 'ہاں', 'میں', 'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']
    for i in range(len(split_paths)):
        closest_word =correct_word(wordlist,split_paths[i],n=2)
        
        if closest_word == None:
            closest_word = correct_word(wordlist, split_paths[i],n=1)
        new_result+=closest_word+" "
    return closest_word
# Source: http://norvig.com/spell-correct.html (with some modifications)
class Spell(object):
    def __init__(self, path):
        #self.file=path
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

import numpy as np
import editdistance
import keras
import csv
import os
import numpy
import doctest

def wer(r, h):
    """
    Source: https://martin-thoma.com/word-error-rate-calculation/
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def wer_sentence(r, h):
    return wer(r.split(), h.split())


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
# model = build()
# print(model.summary())
# # plot graph

# adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# # checkpoint_path = "drive/MyDrive/datasets/training/cp-{epoch:04d}.ckpt"
# # checkpoint_dir = os.path.dirname(checkpoint_path)

# # cp_callback = tf.keras.callbacks.ModelCheckpoint(
# #    checkpoint_path, verbose=1, save_weights_only=True,
# #    # Save weights, every epoch.
# #    save_freq='epoch')
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)
# mc = ModelCheckpoint('weights\\best_val_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
#     # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])
