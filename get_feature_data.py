
import sys
import time
import glob
import librosa
import python_speech_features
import tensorflow as tf
import numpy
import collections
import tools
import random
import multiprocessing
from pypinyin import pinyin,Style
if sys.version > '3':
	import queue as Queue
else:
	import Queue

label_index = {}
index = 1
max_label_length = 40
max_time_step = 1000
def get_label(file_name):
    global index, label_index
    if len(label_index) == 0:
        for line in open("./data/pinyin.data", 'r').readlines():
            py = line.strip()
            if py not in label_index:
                label_index[py] = index
                index += 1
    if type(file_name) == type(""):
        file_name = file_name.split('.')[0]
        fields = file_name.split('|')
    else:
        fields = file_name
    labels = []
    unkown_index = 0
    for field in fields:
        if field not in label_index:
            label_index[field] = unkown_index
        labels.append(label_index[field])
    return labels

def padding_label(labels):
    if len(labels) >= max_label_length:
        return labels[:max_label_length]
    labels.extend([0]*(max_label_length-len(labels)))
    return labels

def padding_feature(feature):
    step, dims = feature.shape
    if step < max_time_step:
        padded = numpy.concatenate((feature, numpy.zeros([max_time_step-step, dims], dtype=float)), axis=0)
    else:
        padded = feature[:max_time_step]
    padded = padded.flatten()
    #print(padded.shape)
    return padded

def load_words_audio(path, queue, lock):
    '''
    文件名是拼音，例如"pin1|yin1.mp3"，目录下只要音频文件即可，适合短句
    '''
    files = glob.glob(path)
    random.shuffle(files)
    for index in range(len(files)):
        name = files[index]
        try:
            sig, rate = librosa.load(name, sr=None)
        except Exception as e:
            print("error:",e)
            continue
        #sig = tools.add_noise(sig)
        filt_num, energy = python_speech_features.base.fbank(sig, nfilt=26)
        feature = numpy.concatenate((filt_num, numpy.expand_dims(energy, axis=-1)), axis=-1)
        labels = get_label(name.split('/')[-1])
        lock.acquire()
        queue.put((feature, labels))
        lock.release()

def load_aishell_audio(audio_path, label_path, queue, lock):
    '''
    按照data_aishell的数据组织方式，一个文件记录音频文件对应的文本
    '''
    print("start load ai_shell")
    label_map = {}
    for line in open(label_path, 'r').readlines():
        fields = line.strip().split()
        audio_name = fields[0]
        sentence = "".join(fields[1:])
        pys = pinyin(sentence, style=Style.TONE3, heteronym=False)    
        labels = []
        for item in pys:
            labels.append(item[0])
        labels = get_label(labels)
        label_map[audio_name] = labels
    files = glob.glob(audio_path)
    random.shuffle(files)
    print(len(files))
    for index in range(len(files)):
        name = files[index]
        key = name.split('/')[-1].split('.')[0]
        print(key)
        if key not in label_map:
            print("lack label: ", key)
            continue
        try:
            sig, rate = librosa.load(name, sr=None)
        except Exception as e:
            print("error:",e)
            continue
        #sig = tools.add_noise(sig)
        #librosa.output.write_wav("./tmp_audio/%s.wav" % key, sig, rate)
        filt_num, energy = python_speech_features.base.fbank(sig, nfilt=40)
        feature = numpy.concatenate((filt_num, numpy.expand_dims(energy, axis=-1)), axis=-1)
        labels = label_map[key]
        lock.acquire()
        queue.put((feature, labels))
        lock.release()

def comsume(queue, lock, mode):
    test_writer = tf.io.TFRecordWriter("test.tf_record")
    if mode == "test":
        train_writer = test_writer
        eval_writer = test_writer
    else:
        train_writer = tf.io.TFRecordWriter("train.tf_record")
        eval_writer = tf.io.TFRecordWriter("eval.tf_record")
    time.sleep(60)
    print("start write...")
    count = 0
    while True:
        if queue.empty() == True:
            time.sleep(10)
        lock.acquire()
        try:
            data = queue.get(block=True, timeout=3)
        except Exception:
            break
        lock.release()
        feature, labels = data
        count += 1
        if random.randint(0,10) > 8:
            write_to_file(eval_writer, feature, labels)
        else:
            write_to_file(train_writer, feature, labels)
    print(count)
    train_writer.close()
    eval_writer.close()

def write_to_file(writer, feature, labels):
    global count
    if len(labels) > max_label_length or len(feature) > max_time_step:
        return
    features = collections.OrderedDict()
    features["label_length"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(labels)]))
    features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=padding_label(labels)))
    features["feature_length"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(feature)]))
    features["feature"] = tf.train.Feature(float_list=tf.train.FloatList(value=padding_feature(feature)))
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

if __name__ == "__main__":
    tools.load_background_audio("./background/*")
    threads = []
    write_lock = multiprocessing.Lock()
    queue = multiprocessing.Manager().Queue()
    mode = sys.argv[1]
    if len(sys.argv) > 3:
        label_path = sys.argv[2]
        audio_path = sys.argv[3]
        threads.append(multiprocessing.Process(target=load_aishell_audio, args=(audio_path, label_path, queue, write_lock)))
        #load_aishell_audio(audio_path, label_path, train_writer, eval_writer)
    if len(glob.glob("./data/*.mp3")) > 0:
        threads.append(multiprocessing.Process(target=load_words_audio, args=("./data/*.mp3", queue, write_lock)))
    threads.append(multiprocessing.Process(target=comsume, args=(queue,write_lock, mode)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    #load_words_audio("./data/*.mp3", train_writer, eval_writer)
