# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import skvideo

from moviepy.editor import *

# ffmpeg_path = 'D:\\Programs\\ffmpeg\\bin\\'
# skvideo.setFFmpegPath(ffmpeg_path)

import skvideo.io
import mediapipe as mp
import cv2
from keras import backend as K


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


class Align(object):
    def __init__(self, absolute_max_string_len=26, label_func=None):
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
            # print("new_array: ", new_array)
            # print("new_array[0]: ", new_array[0])
            if(new_array[0]!=''):
                list_of_strings.append(new_array[0])
                #print(new_array)
                #list_of_strings.append(new_array[1])

        #code to safe the start and end of the video after removing the silence
        if (len(lines) > 2):
            length = len(lines)
            firstline = lines[0].split(' ')
            lastline = lines[length - 1].split(' ')
            last=lastline[0]
            last=last[1:len(last)]
            # print(firstline)
            # print(firstline[2])
            # print(last)

            self.start = firstline[2]
            self.end = last
            # print(self.start)
            # print(self.end)



        #print(list_of_strings)
        # new_string=''
        # for i in range(len(list_of_strings)):
        #     if(i<len(list_of_strings)-1):
        #         new_string+=list_of_strings[i]+" "
        #     else:
        #         new_string += list_of_strings[i]
        # print(new_string)



        #align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
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


import glob
import albumentations as albu
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class DataAugmenter():
    def __init__(self,Frames):
        self.frames=Frames
    def DataAugment(self):
        # transform1 = albu.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.6, hue=0.2, p=0.0)
        # transform4 = albu.ColorJitter(brightness=0.9, contrast=0.8, saturation=0.8, hue=0.8, p=0.0)
        transformVertical = albu.HorizontalFlip(p=0.5)
        verticalFrames=[]
        # transform1Frames=[]
        # transform4Frames=[]
        for image in self.frames:
            print(image.shape)
            augmented_image = transformVertical(image=image)['image']
            # transform1Image=  transform1(image=image)['image']
            #transform4Image=transform4(image=image)['image']
            verticalFrames.append(augmented_image)
            # transform1Frames.append(transform1Image)
            # transform4Frames.append(transform4Image)

        return np.array(verticalFrames)




def makePath(datapath,rangee):
    new_path=''
    for i in range(rangee):
        new_path+=datapath[i]+'\\'


    return new_path

class datasetMaker:
    def __init__(self, videos_path, align_path, img_c, img_w, img_h, frames_n, max_length):
        self.path = videos_path
        self.align_path = align_path
        self.blank_label = 39
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = max_length

    def enumerate_videos(self, path):
        video_list = []
        print(path)
        # print("dishfisdfh")
        for video_path in glob.glob(path):
           print(video_path)
           array=video_path.split('\\')
           print(array)
           new_Split=array[6].split('.')


           if(len(new_Split)==2 and new_Split[1]=='avi'):

                    try:
                        if os.path.isfile(video_path):
                            video_list.append(video_path)
                            #print("videopath",video_path)
                            #video = Video().from_video(video_path)
                        else:
                            print("video path not found")
                            #video = Video().from_frames(video_path)
                    except AttributeError as err:
                        raise err
                    except:
                        print("Error loading video: ") + video_path
                        continue
                    # print("video shape: ", video.data.shape)
                    # print("count: ", count)
                    # if K.image_data_format() == 'channels_first' and video.data.shape != ( self.img_c, self.frames_n, self.img_w, self.img_h):
                    #     #print("Video ") + video_path + " has incorrect shape " + str(video.data.shape) + ", must be " + str( (self.img_c, self.frames_n, self.img_w, self.img_h)) + ""
                    #     continue
                    # if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n, self.img_w, self.img_h, self.img_c):
                    #     #print("Video ") + video_path + " has incorrect shape " + str(video.data.shape) + ", must be " + str( (self.frames_n, self.img_w, self.img_h, self.img_c)) + ""
                    #     continue


        return video_list

    def get_align(self, _id):
        return self.align_hash[_id]

    def enumerate_align_hash(self, video_list):
        self.align_hash = {}
        for video_path in video_list:

            array=video_path.split('\\')
            new_path=makePath(array,6)
            print(array[6])
            vid_Split=array[6].split('.')
            new_path+=vid_Split[0]+'.align'
            video_id = vid_Split[0]
            #print("video_id"+video_id)
            #align_path = os.path.join(self.align_path, video_id) + ".align"
            #print("align_path: ",new_path)
            self.align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(new_path)
        return self.align_hash

    def get_batch(self, video_list, align_hash,count1,count2):
        X_data_path = video_list
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        id_list=[]
        j=count1
        while(j<count2):
            id_list.append(j)
            j+=1
        print("id_list: ",id_list)
        for path in X_data_path:
            # print("path: ",path)

            temp = path.split('\\')[-1]
            t = temp.split('.')
            print("t: ",t)
            get_path_id=int(t[0])
            if(get_path_id in id_list):

                align = self.get_align(t[0])
                video = Video().from_video(path,align.start,align.end)
                video_unpadded_length = video.length
                X_data.append(video.data)
                augmenter=DataAugmenter(video.data)
                verticalTransform=augmenter.DataAugment()
                print("video data: ",video.data.shape)
                print("Vertical transform: ",verticalTransform.shape)
                X_data.append(verticalTransform)
                # X_data.append(trans1)
                # X_data.append(trans2)
                # Y_data.append(align.padded_label)
                # Y_data.append(align.padded_label)
                Y_data.append(align.padded_label)
                Y_data.append(align.padded_label)
                print("padded label",align.padded_label)
                # label_length.append(align.label_length)  # CHANGED [A] -> A, CHECK!
                # label_length.append(align.label_length)
                label_length.append(align.label_length)
                label_length.append(align.label_length)
                # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
                input_length.append(
                    video.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
                input_length.append(
                    video.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
                # input_length.append(
                #     video.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
                # input_length.append(
                #     video.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)

                # source_str.append(align.sentence)  # CHANGED [A] -> A, CHECK!
                # source_str.append(align.sentence)
                print("align.sentence: ",align.sentence)
                source_str.append(align.sentence)
                source_str.append(align.sentence)


        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        for i in range(len(X_data)):
            print("X_Data: ",X_data[i].shape)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([len(source_str)])}  # dummy data for dummy loss function

        return (inputs, outputs)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
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
import os




# def Save_files_as_zarr(data,count,directory):
#   zarr.save('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_input.zarr',data[0]['the_input'])
#   zarr.save('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_labels.zarr',data[0]['the_labels'])
#   zarr.save('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_input_length.zarr',data[0]['input_length'])
#   zarr.save('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_label_length.zarr',data[0]['label_length'])
#   zarr.save('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/ctc.zarr',data[1]['ctc'])
#
# def Load_files_from_zarr(count,directory):
#   input = zarr.load('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_input.zarr')
#   label = zarr.load('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_labels.zarr')
#   input_length = zarr.load('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_input_length.zarr')
#   label_length = zarr.load('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/the_label_length.zarr')
#   ctc = zarr.load('drive/MyDrive/datasets/'+directory+'_zarr/'+count+'/ctc.zarr')
#   inputs = {'the_input': input,
#                   'the_labels': label,
#                   'input_length': input_length,
#                   'label_length': label_length,
#                   #'source_str': source_str  # used for visualization only
#             }
#   outputs = {'ctc': ctc}  # dummy data for dummy loss function
#   return(inputs,outputs)
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
# Press the gg ,9-reen button in the gutter to run the script.
if __name__ == '__main__':


    CURRENT_PATH = r'dataset\\0\\0_test\\*\\*'
    # CURRENT_PATH = os.path.dirname('C:\\Users\\usman\\Documents\\datasets\\s3_DOC\\')
    # DATASET_DIR = os.path.join(CURRENT_PATH, 's3\\video\\*')
    # align_path = os.path.join(CURRENT_PATH, 'align')
    OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
    LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

    PREDICT_GREEDY = False
    PREDICT_BEAM_WIDTH = 200
    PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'grid.txt')
    dataset = datasetMaker(CURRENT_PATH, CURRENT_PATH, 3, 100, 50, 75, 29)
    directory = "0"
    #
    # count1=0
    # count2=52
    # count=1
    count1=0
    count2=108
    count=1
    video_list = dataset.enumerate_videos(CURRENT_PATH)
    align_hash = dataset.enumerate_align_hash(video_list)
    data = dataset.get_batch(video_list, align_hash,count1,count2)

    Save_files_as_zarr(data, str(count), directory)
    data = Load_files_from_zarr(str(count), directory)
    print(data[0]['the_input'].shape)
    print(data[0]['the_labels'].shape)
    print(data[0]['input_length'].shape)
    print(data[0]['label_length'].shape)
    print(data[0]['source_str'].shape)
    print(data[1]['ctc'].shape)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
