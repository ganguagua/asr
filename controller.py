
import tensorflow as tf
from clstm import CLSTM
import time
from keras import backend as K
import sys

nframes = 1000
nfilter = 41
max_sentence_length = 40
nlabel = 1300

def getDataset(fileName, isTraining=True, batch=32, epoch=3):
    name_to_features = {
      "feature": tf.io.FixedLenFeature([nframes*nfilter], tf.float32),
      "label": tf.io.FixedLenFeature([max_sentence_length], tf.int64),
      "feature_length": tf.io.FixedLenFeature([1], tf.int64),
      "label_length": tf.io.FixedLenFeature([1], tf.int64)
    }
    data = tf.data.TFRecordDataset(fileName)
    if isTraining:
        data = data.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    data = data.repeat(epoch).map(lambda record: tf.io.parse_single_example(record, name_to_features)).batch(batch_size=batch, drop_remainder=True)
    return data

def test():
    global batch_size
    save_model_path = "./output/105098-2"
    index = 1
    symbols = {0:"UNK", 1299:'blank'}
    for line in open('pinyin.data', "r").readlines():
        symbols[index] = line.strip()
        index += 1
    batch_size = 1
    model = CLSTM(nframes, nfilter, nlabel, batch_size)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint("./output")).assert_existing_objects_matched()
    dataset = getDataset("test.tf_record", isTraining=False, batch=batch_size, epoch=1)
    step = 0 
    for data in dataset:
        step += 1
        print(step)
        prob, _ = feed_to_model(data, model)
        res = tf.argmax(prob, axis=-1).numpy()
        print(" ".join([symbols[i] for i in res[0]]))

def eval(model):
    dataset = getDataset("eval.tf_record", batch=32, epoch=1)
    total_loss = 0
    step = 0
    for batchData in dataset:
        step += 1
        feature = tf.reshape(batchData["feature"], [batch_size, nframes, -1])
        feature_length = tf.reshape(batchData["feature_length"], [-1])
        label_length = tf.reshape(batchData["label_length"], [-1])
        label = batchData["label"]
        pred = model(feature)
        loss = K.ctc_batch_cost(labels, pred, label_length=label_length, input_length=feature_length)
        #loss = tf.nn.ctc_loss(label, pred, label_length=label_length, logit_length=feature_length, logits_time_major=False)
        total_loss += tf.reduce_mean(loss).numpy()
    return total_loss / step

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def feed_to_model(batchData, model):
    feature = tf.reshape(batchData["feature"], [batch_size, nframes, -1])
    label = batchData["label"]
    feature_length = tf.reshape(batchData["feature_length"], [batch_size, 1])
    label_length = tf.reshape(batchData["label_length"], [batch_size, 1])
    pred = model(feature)
    #with tf.device('/cpu:0'):
    loss = K.ctc_batch_cost(label, pred, input_length=feature_length // 8, label_length=label_length)
    #loss = tf.nn.ctc_loss(label, pred, label_length=label_length, logit_length=feature_length//8, logits_time_major=False)
    loss = tf.reduce_mean(loss)
    return pred, loss

batch_size=32

def train():
    seq_max_length = tf.constant([128]*batch_size, dtype=tf.int64)
    print(seq_max_length.shape)
    dataset = getDataset("train.tf_record", batch=batch_size, epoch=20)
    eval_dataset = iter(getDataset("eval.tf_record", batch=batch_size, epoch=200))
    model = CLSTM(nframes, nfilter, nlabel, batch_size)
    #model = deep_speech_network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-05)
    step = 0
    step_per_epoch = 105098 / batch_size
    train_log_writer = tf.summary.create_file_writer("./logs/train.log")
    eval_log_writer = tf.summary.create_file_writer("./logs/eval.log")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, './output', max_to_keep=3)
    for batchData in dataset:
        step += 1
        feature = tf.reshape(batchData["feature"], [batch_size, nframes, -1])
        label = batchData["label"]
        feature_length = tf.reshape(batchData["feature_length"], [batch_size, 1])
        label_length = tf.reshape(batchData["label_length"], [batch_size, 1])
        with tf.GradientTape() as tape:
            pred = model(feature)
            #with tf.device('/cpu:0'):
            loss = K.ctc_batch_cost(label, pred, input_length=feature_length // 8, label_length=label_length)
            #loss = tf.nn.ctc_loss(label, pred, label_length=label_length, logit_length=feature_length//8, logits_time_major=False)
            loss = tf.reduce_mean(loss)
        if step % 10 == 0:
            eval_data = next(eval_dataset)
            _, eval_loss = feed_to_model(eval_data, model)
            with eval_log_writer.as_default():
                tf.summary.scalar("loss", eval_loss, step)
                eval_log_writer.flush()
            print(step, loss, eval_loss)
        train_variables = model.trainable_variables
        gradients = tape.gradient(target=loss, sources=train_variables)
        #print(gradients)
        optimizer.apply_gradients(zip(gradients, train_variables))
        with train_log_writer.as_default():
            tf.summary.scalar("loss", loss, step)
            if step % step_per_epoch == 0:
                manager.save(step)
            train_log_writer.flush()
    checkpoint.save("./output/final")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    #getDataset("train.tf_record")
