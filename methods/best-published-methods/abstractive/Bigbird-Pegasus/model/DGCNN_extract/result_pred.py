import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
from rouge import Rouge
import json, pickle
import os
import random 

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()


def compute_rouge(source, target):
    #source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except RecursionError:
        scores = rouge.get_scores(hyps=source[:int(len(source)*0.95)], refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_main_metric(source, target):
    metrics = compute_rouge(source, target)
    metrics['main'] = (
            metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
            metrics['rouge-l'] * 0.4
    )
    return metrics['main']

class ResidualGatedConv1D(tf.keras.layers.Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = tf.keras.layers.LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = tf.keras.layers.Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        # 2*filters 相当于两组filters来 一组*sigmoid(另一组)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            #用于对象是否包含对应的属性值
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs


def bulid_extract_model(max_len,input_size,hidden_size):
    input_ = tf.keras.layers.Input((max_len,input_size))
    x = tf.keras.layers.Masking()(input_)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(hidden_size, use_bias=False)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
    x = tf.keras.layers.Dropout(0.1)(x) 
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=16)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out_put = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=input_, outputs=out_put)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def evaluate(model,data,data_x,threshold=0.2):
    evaluater = 0
    pred = model.predict(data_x)[:,:,0]
    # [sample_num,256]

    for d, yp in tqdm(zip(data, pred), desc='evaluating'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        pred_sum = ' '.join([d[0][i] for i in yp])
        try:
            evaluater += compute_main_metric(pred_sum, d[1])
        except Exception as e:
            evaluater += 0
            print(f"pred_sum: {pred_sum}")
            print(f"\nd[1]: {d[1]}")
            
    return evaluater/len(data)

class Evaluator(tf.keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self,threshold,valid_data,valid_x,fold,idx):
        self.best_metric = 0.0
        self.threshold = threshold
        self.valid_data = valid_data
        self.valid_x = valid_x
        self.fold = fold
        self.idx = idx

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%15 != 0:
            return
        eva = evaluate(self.model, self.valid_data, self.valid_x, self.threshold + 0.1)
        if  eva > self.best_metric:  # 保存最优
            self.best_metric = eva
            self.model.save_weights(f'REPLACE-BY-YOUR-PATH/Longsumm_code/model/DGCNN_saved_splits/split_{self.idx}/extract_model_{self.fold}.hdf5')
            print('eval raise to %s'%eva)
        else:
            print('eval is %s, not raise'%eva)

def data_split(data, fold, num_folds, mode):
    """
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]
        
    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D

def load_data(filename):
    """Load data
    Return：[(texts, labels, summary)]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D

def main(idx):

    input_size = 1024
    hidden_size = 512
    epochs = 30
    batch_size = 32
    threshold = 0.2
    num_folds = 5
    max_len = 400

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    data_x = np.load(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/processed/docs_roberta_avail.npy')
    
    # data_y = np.zeros( (token_x.shape[0], token_x.shape[1], 1)  )
    data_y = np.zeros_like(data_x[..., :1])

    with open(f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/processed/labels_avail.pickle", 'rb') as handle:
        labels = pickle.load(handle)

        
    with open(f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/processed/data_avail.pickle", 'rb') as handle:
        data = pickle.load(handle)

    for i, d in enumerate(labels):
        for j in d:
            if j < len(data_y[i]):
                data_y[i][j][0] = 1

    for fold in range(num_folds): # one fold multiple epochs?
        train_x = data_split(data_x, fold, num_folds, 'train')
        train_y = data_split(data_y, fold, num_folds, 'train')
        valid_x = data_split(data_x, fold, num_folds, 'valid')
        # valid_y = data_split(data_y, fold, num_folds, 'valid')

        valid_data = data_split(data, fold, num_folds, 'valid')
        if len(valid_data) == 0:
            print('val data: ', valid_data)


        K.clear_session()
        model = bulid_extract_model(max_len, input_size, hidden_size)
        evaluator = Evaluator(threshold, valid_data, valid_x, fold, idx)
        model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[evaluator]
        )

    # Evaluation on the blind test dataset

    dataset_to_predict = "abstractive"
    threshold = 0.5

    if dataset_to_predict == "extractive":
        data_x = np.load(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/processed/docs_roberta_test.npy')
        
        with open(f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/processed/data_test.pickle", 'rb') as handle:
            data = pickle.load(handle)

    elif dataset_to_predict == "abstractive":
        data_x = np.load(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive_test_for_extractive_prediction/docs_roberta_test.npy')

        with open(f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive_test_for_extractive_prediction/data_test.pickle", 'rb') as handle:
            data = pickle.load(handle)


    print("---- Evaluation Starts ----")

    predictions = []
    groudtruth = []
    probabilies = []

    for fold in range(num_folds):
        test_x = data_split(data_x, fold, num_folds, 'valid')

        test_data = data_split(data, fold, num_folds, 'valid')

        K.clear_session()
        model = bulid_extract_model(max_len, input_size, hidden_size)

        model.load_weights(f'REPLACE-BY-YOUR-PATH/Longsumm_code/model/DGCNN_saved_splits/split_{idx}/extract_model_{fold}.hdf5')
        pred = model.predict(test_x)[:,:,0]

        for d, yp in tqdm(zip(test_data, pred), desc='evaluating'):
            yp = yp[:len(d[0])]

            yp_copy = yp.copy()

            yp = np.where(yp > threshold)[0]    

            prob = 1

            pred_sum = ""
            for i in yp:
                if len(pred_sum.split()) + len(d[0][i].split()) > 600:
                    break
                pred_sum += " " + d[0][i]
                prob *= yp_copy[i]

            probabilies.append(prob)

            predictions.append(pred_sum.strip())
            groudtruth.append(d[1].strip())


    threshold_to_write = int(threshold*100)

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{idx}/baseline/DGCNN/system_{threshold_to_write}.txt", "w") as f:
        pred_to_write = "\n".join( predictions )
        f.write(  pred_to_write   )

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{idx}/baseline/DGCNN/reference_{threshold_to_write}.txt", "w") as f:
        truth_to_write = "\n\n".join( groudtruth )
        f.write(  truth_to_write   )

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{idx}/baseline/DGCNN/system_probs_{threshold_to_write}.txt", "w") as f:
        probs_to_write = "\n\n".join( str(prob) for prob in probabilies )
        f.write(  probs_to_write   )

    
if __name__ == "__main__":
    for idx in range(10):
        print(f"idx: ----- {idx} -----")
        main(idx)
