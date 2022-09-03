train_record_path = "REPLACE-BY-YOUR-PATH/datasets/dataset/LongSumm2021/abstractive/processed/eval.tfrecord"


import tensorflow as tf 

def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(record, name_to_features)

    document = example['document']
    summary = example['summary']

    document = tf.io.parse_tensor(document, out_type=tf.string)
    summary = tf.io.parse_tensor(summary, out_type=tf.string) 

    return document, summary


dataset = tf.data.TFRecordDataset(train_record_path)

dataset = dataset.map(_decode_record)

print(len(list(dataset)))

for sample in dataset.take(2):
  print(sample[0].numpy()) #the text data
  print(sample[1]) #the label
  print('type: ', type(sample[0].numpy()) )