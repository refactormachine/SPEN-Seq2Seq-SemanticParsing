import keras
import numpy

from keras.layers import Input, Dense, merge, Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, \
    Dropout, Embedding
from keras.layers import Activation, TimeDistributed
from keras.layers import Bidirectional, LSTM
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2


def decomposable_model_generation(max_len_utt, max_len_path, hidden_len, class_utt_batch_size,
                                  settings):
    # todo: work on batch
    utterance_inp = Input(shape=(max_len_utt,), dtype='float32', name='uttr_inp')
    path_inp = Input(shape=(max_len_path,), dtype='float32', name='path_inp')

    # Construct operations, which we'll chain together.
    vectors = numpy.ndarray((100, 8), dtype='float32')
    # todo: think of removing embedding layer
    embed_utt = StaticEmbedding(vectors, max_len_utt, hidden_len, dropout=0.2, nr_tune=5000)
    embed_path = StaticEmbedding(vectors, max_len_path, hidden_len, dropout=0.2, nr_tune=5000)
    encode_utt = BiLSTM(max_len_utt, hidden_len, dropout=settings['dropout'])
    encode_path = BiLSTM(max_len_path, hidden_len, dropout=settings['dropout'])
    attend = Attention(hidden_len, dropout=settings['dropout'])
    align = Alignment(hidden_len)
    compare = CompareAndAggregate(hidden_len, dropout=settings['dropout'])
    output_score = Ranker(hidden_len, class_utt_batch_size, dropout=settings['dropout'])

    utterance_embedded = embed_utt(utterance_inp)
    path_embedded = embed_path(path_inp)

    uttr_enc = encode_utt(utterance_embedded)
    path_enc = encode_path(path_embedded)

    attention = attend(uttr_enc, path_enc, max_len_utt, max_len_path)

    align_uttr = align(path_enc, attention, max_len_utt)
    align_path = align(uttr_enc, attention, max_len_path, transpose=True)

    compare_uttr = compare(uttr_enc, align_uttr, max_len_utt, True)
    compare_path = compare(path_enc, align_path, max_len_path, False)

    ranks = output_score(compare_uttr, compare_path)

    decomposable_model = Model(input=[utterance_inp, path_inp], output=[ranks])

    decomposable_model.compile(
        optimizer=Adam(lr=settings['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return decomposable_model


class StaticEmbedding(object):
    def __init__(self, vectors, max_length, nr_out, nr_tune=1000, dropout=0.0):
        self.nr_out = nr_out
        self.max_length = max_length
        self.embed = Embedding(
            vectors.shape[0],
            vectors.shape[1],
            input_length=max_length,
            weights=[vectors],
            name='embed_{}'.format(max_length),
            trainable=False)
        self.tune = Embedding(
            nr_tune,
            nr_out,
            input_length=max_length,
            weights=None,
            name='tune_{}'.format(max_length),
            trainable=True,
            dropout=dropout)
        self.mod_ids = Lambda(lambda sent: sent % (nr_tune - 1) + 1,
                              output_shape=(self.max_length,))

        self.project = TimeDistributed(
            Dense(
                nr_out,
                activation=None,
                bias=False,
                name='project'))

    def __call__(self, sentence):
        def get_output_shape(shapes):
            print(shapes)
            return shapes[0]

        mod_sent = self.mod_ids(sentence)
        tuning = self.tune(mod_sent)
        # tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.nr_out))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')
        return vectors


class BiLSTM(object):
    def __init__(self, max_length, out_len, dropout=0.0):
        self.model = Sequential(name='BiLSTM_{}'.format(max_length))
        self.model.add(Bidirectional(LSTM(out_len, return_sequences=True,
                                     dropout_W=dropout, dropout_U=dropout),
                                     input_shape=(max_length, out_len)))
        self.model.add(TimeDistributed(Dense(out_len, activation='relu', init='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)


class Attention(object):
    def __init__(self, hidden_len, dropout=0.0, L2=0.0):
        self.model_utter = Sequential(name='sequential_att_utt')
        self.model_utter.add(Dropout(dropout, input_shape=(hidden_len,)))
        self.model_utter.add(
            Dense(hidden_len, name='attend1',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(hidden_len,),
                  activation='relu'))
        self.model_utter.add(Dropout(dropout))
        self.model_utter.add(Dense(hidden_len, name='attend2',
                                   init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_utter = TimeDistributed(self.model_utter)

        self.model_path = Sequential(name='sequential_att_path')
        self.model_path.add(Dropout(dropout, input_shape=(hidden_len,)))
        self.model_path.add(
            Dense(hidden_len, name='attend3',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(hidden_len,),
                  activation='relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(hidden_len, name='attend4',
                                  init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, utter, path, max_len_utter, max_len_path):
        # attend step that skips the quadratic complexity of normal attention
        def merge_mode(utter_path):
            bitwise_attention = K.batch_dot(utter_path[1], K.permute_dimensions(utter_path[0], (0, 2, 1)))
            return K.permute_dimensions(bitwise_attention, (0, 2, 1))

        utter_model = self.model_utter(utter)
        path_model = self.model_path(path)

        return merge(
            [utter_model, path_model],
            mode=merge_mode,
            output_shape=(max_len_utter, max_len_path))


class Alignment(object):
    def __init__(self, hidden_len):
        self.hidden_len = hidden_len

    def __call__(self, sentence, attentions_matrix, max_length, transpose=False):
        def normalize_attention(attention_matrix_sentence):
            attention_matrix = attention_matrix_sentence[0]
            att_sentence = attention_matrix_sentence[1]
            if transpose:
                attention_matrix = K.permute_dimensions(attention_matrix, (0, 2, 1))
            # softmax attention
            exp_mat = K.exp(attention_matrix - K.max(attention_matrix, axis=-1, keepdims=True))
            sum_mat = K.sum(exp_mat, axis=-1, keepdims=True)
            softmax_attention = exp_mat / sum_mat
            return K.batch_dot(softmax_attention, att_sentence)

        return merge([attentions_matrix, sentence], mode=normalize_attention,
                     output_shape=(max_length, self.hidden_len))


class CompareAndAggregate(object):
    def __init__(self, hidden_len, L2=0.0, dropout=0.0):
        self.model_utt = Sequential(name='sequential_cmp_utt')
        self.model_utt.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model_utt.add(Dense(hidden_len, name='compare1',
                                 init='he_normal', W_regularizer=l2(L2)))
        self.model_utt.add(Activation('relu'))
        self.model_utt.add(Dropout(dropout))
        self.model_utt.add(Dense(hidden_len, name='compare2',
                                 init='he_normal', W_regularizer=l2(L2)))
        self.model_utt.add(Activation('relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential(name='sequential_cmp_path')
        self.model_path.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model_path.add(Dense(hidden_len, name='compare3',
                                  init='he_normal', W_regularizer=l2(L2)))
        self.model_path.add(Activation('relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(hidden_len, name='compare4',
                                  init='he_normal', W_regularizer=l2(L2)))
        self.model_path.add(Activation('relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, sent, align, max_len, is_model_utt, **kwargs):
        if is_model_utt:
            result = self.model_utt(merge([sent, align], mode='concat'))
        else:
            result = self.model_path(merge([sent, align], mode='concat'))
        avged = GlobalAveragePooling1D()(result, mask=max_len)
        maxed = GlobalMaxPooling1D()(result, mask=max_len)
        merged = merge([avged, maxed])
        result = BatchNormalization()(merged)
        return result


class Ranker(object):
    def __init__(self, hidden_len, out_len, dropout=0.0, L2=0.0):
        self.model = Sequential(name='sequential_ranker')
        self.model.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model.add(Dense(hidden_len, name='hidden_rank',
                             init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_len, name='entail2',
                             init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(out_len, name='entail_out', activation='softmax',
                             W_regularizer=l2(L2), init='zero'))

    def __call__(self, compare_utter, compare_path):
        ranker = merge([compare_utter, compare_path], mode='concat')
        # ranker = K.permute_dimensions(ranker, (1, 0))  # transpose (?, 20) -> (20, ?)
        # ranker = K.transpose(ranker)
        # ranker = merge(ranker, mode='transpose')
        ranker = self.model(ranker)
        return ranker
