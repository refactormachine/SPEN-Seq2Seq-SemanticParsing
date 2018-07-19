# Semantic similarity with decomposable attention (using spaCy and Keras)
# Practical state-of-the-art text similarity with spaCy and Keras
import numpy

from keras.layers import InputSpec, Layer, Input, Dense, merge, Masking
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import Bidirectional, GRU, LSTM
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ELU
import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


def build_model(shape_utt, shape_path, settings):
    """Compile the model."""
    max_length_utt, nr_hidden_utt, nr_class_utt = shape_utt
    max_length_path, nr_hidden_path, nr_class_path = shape_path
    nr_hidden_output = min(nr_hidden_utt,nr_hidden_path)
    
    # Declare inputs.
    utterance_inp = Input(shape=(max_length_utt,nr_hidden_utt,), dtype='float32', name='words1')
    path_inp = Input(shape=(max_length_path,nr_hidden_path,), dtype='float32', name='words2')

    # Construct operations, which we'll chain together.
    # embed = _StaticEmbedding(vectors, max_length, nr_hidden, dropout=0.2, nr_tune=5000)
    if settings['gru_encode']:
        encode_utt = _BiRNNEncoding(max_length_utt, nr_hidden_output,nr_hidden_utt, dropout=settings['dropout'])
        encode_path = _BiRNNEncoding(max_length_path, nr_hidden_output,nr_hidden_path, dropout=settings['dropout'])
    attend = _Attention(max_length_utt, max_length_path, nr_hidden_output, dropout=settings['dropout'])
    align = _SoftAlignment(nr_hidden_output)
    compare = _Comparison(nr_hidden_output, dropout=settings['dropout'])
    entail = _Entailment(nr_hidden_output, nr_class_utt, dropout=settings['dropout'])

    # Declare the model as a computational graph.
    # sent1 = embed(ids1) # Shape: (i, n)
    # sent2 = embed(ids2) # Shape: (j, n)

    if settings['gru_encode']:
        sent1 = encode_utt(utterance_inp)
        sent2 = encode_path(path_inp)

    attention = attend(sent1, sent2)  # Shape: (i, j)

    align1 = align(sent2, attention, max_length_utt)
    align2 = align(sent1, attention, max_length_path, transpose=True)

    feats1 = compare(sent1, align1, max_length_utt, 0)
    feats2 = compare(sent2, align2, max_length_path, 1)

    scores = entail(feats1, feats2)

    # Now that we have the input/output, we can construct the Model object...
    model = Model(input=[utterance_inp, path_inp], output=[scores])

    # ...Compile it...
    model.compile(
        optimizer=Adam(lr=settings['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    # ...And return it for training.
    return model


class _StaticEmbedding(object):
    def __init__(self, vectors, max_length, nr_out, nr_tune=1000, dropout=0.0):
        self.nr_out = nr_out
        self.max_length = max_length
        self.embed = Embedding(
                        vectors.shape[0],
                        vectors.shape[1],
                        input_length=max_length,
                        weights=[vectors],
                        name='embed',
                        trainable=False)
        self.tune = Embedding(
                        nr_tune,
                        nr_out,
                        input_length=max_length,
                        weights=None,
                        name='tune',
                        trainable=True,
                        dropout=dropout)
        self.mod_ids = Lambda(lambda sent: sent % (nr_tune-1)+1,
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
        #tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.nr_out))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')
        return vectors


class _BiRNNEncoding(object):
    def __init__(self, max_length, nr_out,nr_in, dropout=0.0):
        self.model = Sequential()
        # self.model.add(Masking(mask_value=-9999., input_shape=(max_length, nr_in)))
        self.model.add(Bidirectional(LSTM(nr_out, return_sequences=True,
                                          dropout_W=dropout, dropout_U=dropout),
                                     input_shape=(max_length, nr_in)))
        self.model.add(TimeDistributed(Dense(nr_out, activation='relu', init='he_normal', name = 'birnndense')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)


class _Attention(object):
    def __init__(self, max_length_utt, max_length_path, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
        self.max_length_utt = max_length_utt
        self.max_length_path = max_length_path
        self.model_utt = Sequential()
        self.model_utt.add(Dropout(dropout, input_shape=(nr_hidden,)))
        self.model_utt.add(
            Dense(nr_hidden, name='attend1',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(nr_hidden,), activation='relu'))
        self.model_utt.add(Dropout(dropout))
        self.model_utt.add(Dense(nr_hidden, name='attend2',
                                 init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential()
        self.model_path.add(Dropout(dropout, input_shape=(nr_hidden,)))
        self.model_path.add(
            Dense(nr_hidden, name='attend3',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(nr_hidden,), activation='relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(nr_hidden, name='attend4',
                                  init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, sent1, sent2):
        def _outer(AB):
            att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
            return K.permute_dimensions(att_ji,(0, 2, 1))
        return merge(
                [self.model_utt(sent1), self.model_path(sent2)],
                mode=_outer,
                output_shape=(self.max_length_utt, self.max_length_path))


class _SoftAlignment(object):
    def __init__(self, nr_hidden):
        self.nr_hidden = nr_hidden

    def __call__(self, sentence, attention, max_length, transpose=False):
        def _normalize_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            if transpose:
                att = K.permute_dimensions(att, (0, 2, 1))
            # 3d softmax
            e = K.exp(att - K.max(att, axis=-1, keepdims=True))
            s = K.sum(e, axis=-1, keepdims=True)
            sm_att = e / s
            return K.batch_dot(sm_att, mat)
        return merge([attention, sentence], mode=_normalize_attention,
                     output_shape=(max_length, self.nr_hidden)) # Shape: (i, n)


class _Comparison(object):
    def __init__(self, nr_hidden, L2=0.0, dropout=0.0):
        self.model_utt = Sequential()
        self.model_utt.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model_utt.add(Dense(nr_hidden, name='compare1',
                                 init='he_normal', W_regularizer=l2(L2)))
        self.model_utt.add(Activation('relu'))
        self.model_utt.add(Dropout(dropout))
        self.model_utt.add(Dense(nr_hidden, name='compare2',
                                 W_regularizer=l2(L2), init='he_normal'))
        self.model_utt.add(Activation('relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential()
        self.model_path.add(Dropout(dropout, input_shape=(nr_hidden * 2,)))
        self.model_path.add(Dense(nr_hidden, name='compare1',
                                  init='he_normal', W_regularizer=l2(L2)))
        self.model_path.add(Activation('relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(nr_hidden, name='compare2',
                                  W_regularizer=l2(L2), init='he_normal'))
        self.model_path.add(Activation('relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, sent, align, max_len, type, **kwargs):
        if type == 0:
            result = self.model_utt(merge([sent, align], mode='concat')) # Shape: (i, n)
        else:
            result = self.model_path(merge([sent, align], mode='concat')) # Shape: (i, n)

        # avged = GlobalAveragePooling1D()(result, mask=max_len)
        # maxed = GlobalMaxPooling1D()(result, mask=max_len)
        summed = _GlobalSumPooling1D()(result, mask=max_len)
        # merged = merge([avged, maxed])
        result = BatchNormalization()(summed)
        return result


# class GlobalAveragey
#  input_shape[0], input_shape[2]


# class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
#     def call(self, x, mask=None):
#         if mask != None:
#             return K.sum(x, axis=1) / K.sum(K.tf.to_float(mask), axis=1)
#         else:
#             return GlobalAveragePooling1D().call(x)


# class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
#     def call(self, x, mask=None):
#         if mask != None:
#             return K.max(x, axis=1)
#         else:
#             return GlobalAveragePooling1D().call(x)




class _Entailment(object):
    def __init__(self, nr_hidden, nr_out, dropout=0.0, L2=0.0):
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(nr_hidden*2,)))
        self.model.add(Dense(nr_hidden, name='entail1',
            init='he_normal', W_regularizer=l2(L2), input_shape=(nr_hidden*2,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(nr_hidden, name='entail2',
            init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(nr_out, name='entail_out', activation='softmax',
                        W_regularizer=l2(L2), init='zero'))

    def __call__(self, feats1, feats2):
        features = merge([feats1, feats2], mode='concat')
        return self.model(features)


class _GlobalSumPooling1D(Layer):
    '''Global sum pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(_GlobalSumPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.get_shape()[-1])
            # mask (batch, time, x_dim)
            mask = K.tf.transpose(mask, [0, 2, 1])
            x = x * mask
            return K.sum(x * K.clip(K.tf.to_float(mask), 0, 1), axis=1)
        else:
            return K.sum(x, axis=1)


def test_build_model():
    vectors = numpy.ndarray((100, 8), dtype='float32')
    shape = (10, 16, 2)
    settings = {'lr': 0.001, 'dropout': 0.2, 'gru_encode':True}
    model = build_model(vectors, shape, settings)


def test_fit_model():

    def _generate_X(nr_example, length, nr_vector):
        X1 = numpy.ndarray((nr_example, length), dtype='int32')
        X1 *= X1 < nr_vector
        X1 *= 0 <= X1
        X2 = numpy.ndarray((nr_example, length), dtype='int32')
        X2 *= X2 < nr_vector
        X2 *= 0 <= X2
        return [X1, X2]

    def _generate_Y(nr_example, nr_class):
        ys = numpy.zeros((nr_example, nr_class), dtype='int32')
        for i in range(nr_example):
            ys[i, i % nr_class] = 1
        return ys

    vectors = numpy.ndarray((100, 8), dtype='float32')
    shape = (10, 16, 2)
    settings = {'lr': 0.001, 'dropout': 0.2, 'gru_encode':True}
    model = build_model(vectors, shape, settings)

    train_X = _generate_X(20, shape[0], vectors.shape[0])
    train_Y = _generate_Y(20, shape[2])
    dev_X = _generate_X(15, shape[0], vectors.shape[0])
    dev_Y = _generate_Y(15, shape[2])

    model.fit(train_X, train_Y, validation_data=(dev_X, dev_Y), nb_epoch=5,
              batch_size=4)


__all__ = [build_model]