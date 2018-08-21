import keras
import numpy

from keras.layers import Input, Dense, merge, Lambda, BatchNormalization
from keras.layers import Activation, TimeDistributed
from keras.layers import Bidirectional, LSTM
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2


def decomposable_model_generation(shape_utt, shape_path, settings):
    max_length_utt, hidden_utt_num = shape_utt
    max_length_path, hidden_path_num = shape_path

    hidden_len = 300

    utterance_inp = Input(shape=(max_length_utt, hidden_utt_num,), dtype='float32', name='uttr_inp')
    path_inp = Input(shape=(max_length_path, hidden_path_num,), dtype='float32', name='path_inp')

    encode_utt = BiLSTM(max_length_utt, hidden_len, hidden_utt_num)
    encode_path = BiLSTM(max_length_path, hidden_len, hidden_path_num)
    attend = Attention(max_length_utt, max_length_path, hidden_len)
    align = Alignment(hidden_len)
    compare = CompareAndAggregate(hidden_len)
    output_score = Ranker(hidden_len)

    uttr_enc = encode_utt(utterance_inp)
    path_enc = encode_path(path_inp)

    attention = attend(uttr_enc, path_enc)

    align_uttr = align(path_enc, attention, max_length_utt)
    align_path = align(uttr_enc, attention, max_length_path, transpose=True)

    compare_uttr = compare(uttr_enc, align_uttr, max_length_utt, 0)
    compare_path = compare(path_enc, align_path, max_length_path, 1)

    ranks = output_score(compare_uttr, compare_path)

    decomposable_model = Model(input=[utterance_inp, path_inp], output=[ranks])

    decomposable_model.compile(
        optimizer=Adam(lr=settings['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return decomposable_model


class BiLSTM(object):
    def __init__(self, max_length, out_len, in_len):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(out_len, return_sequences=True),
                                     input_shape=(max_length, in_len)))
        self.model.add(TimeDistributed(Dense(out_len, activation='relu', init='he_normal')))

    def __call__(self, sentence):
        return self.model(sentence)


class Attention(object):
    def __init__(self, max_length_utt, max_length_path, hidden_len):
        self.max_length_utt = max_length_utt
        self.max_length_path = max_length_path
        self.model_utt = Sequential()
        self.model_utt.add(
            Dense(hidden_len, name='attend1', init='he_normal', input_shape=(hidden_len,), activation='relu'))
        self.model_utt.add(Dense(hidden_len, name='attend2',
                                 init='he_normal', activation='relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential()
        self.model_path.add(
            Dense(hidden_len, name='attend3', init='he_normal', input_shape=(hidden_len,), activation='relu'))

        self.model_path.add(Dense(hidden_len, name='attend4', init='he_normal', activation='relu'))

        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, utter, path):
        # attend step that skips the quadratic complexity of normal attention
        def merge_mode(utter_path):
            bitwise_attention = K.batch_dot(utter_path[1], K.permute_dimensions(utter_path[0], (0, 2, 1)))
            return K.permute_dimensions(bitwise_attention, (0, 2, 1))

        return merge(
            [self.model_utt(utter), self.model_path(path)],
            mode=merge_mode,
            output_shape=(self.max_length_utt, self.max_length_path))


class Alignment(object):
    def __init__(self, hidden_len):
        self.hidden_len = hidden_len

    def __call__(self, sentence, attentions_matrix, max_length, transpose=False):
        def normalization3d(attention_matrix_sentence):
            attention_matrix = attention_matrix_sentence[0]
            att_sentence = attention_matrix_sentence[1]
            if transpose:
                attention_matrix = K.permute_dimensions(attention_matrix, (0, 2, 1))
            # softmax attention
            exp_mat = K.exp(attention_matrix - K.max(attention_matrix, axis=-1, keepdims=True))
            sum_mat = K.sum(exp_mat, axis=-1, keepdims=True)
            softmax_attention = exp_mat / sum_mat
            return K.batch_dot(softmax_attention, att_sentence)

        return merge([attentions_matrix, sentence], mode=normalization3d,
                     output_shape=(max_length, self.hidden_len))


class CompareAndAggregate(object):
    def __init__(self, hidden_len):
        self.model_utt = Sequential()
        self.model_utt.add(Dense(hidden_len, name='compareutt', input_shape=(hidden_len * 2,),
                                 init='he_normal'))
        self.model_utt.add(Activation('relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential()
        self.model_path.add(Dense(hidden_len, name='comparepath', input_shape=(hidden_len * 2,),
                                  init='he_normal'))
        self.model_path.add(Activation('relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, sent, align, max_len, model_type, **kwargs):
        if model_type == 0:
            result = self.model_utt(merge([sent, align], mode='concat'))
        else:
            result = self.model_path(merge([sent, align], mode='concat'))

        aggregated = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(result)
        return aggregated


class Ranker(object):
    def __init__(self, hidden_len):
        self.model = Sequential()
        self.model.add(Dense(hidden_len, name='hidden_rank', input_shape=(hidden_len * 2,),
                             init='he_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1, name='ranker', activation='softmax', init='he_normal'))

    def __call__(self, compare_utter, compare_path):
        ranker = merge([compare_utter, compare_path], mode='concat')
        ranker = self.model(ranker)
        ranker = BatchNormalization()(ranker)
        return ranker
