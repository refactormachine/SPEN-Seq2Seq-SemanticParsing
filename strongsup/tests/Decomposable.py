import tensorflow as tf
from keras.engine import Model
from keras.layers import Input, Embedding, LSTM, Dense, merge, Bidirectional, Dropout
from attention_decoder import AttentionDecoder

# TODO move all constants to the constants
from keras.models import Sequential

MAX_UTURRENCE_LEN = 10
UTTERANCE_WORD_EMBED_LEN = 10
MAX_PATH_LEN = 10
CASE_EMBED_LEN = 10


# TODO here will go the embedder also
class Decomposable:
    def __init__(self):
        # sizeof input beams is (batch/num_of_paths/num_of_cases
        # Placeholder inputs
        self._utterance_input = Input(shape=(MAX_UTURRENCE_LEN, UTTERANCE_WORD_EMBED_LEN,), dtype='float32')
        self._path_input = Input(shape=(MAX_PATH_LEN, CASE_EMBED_LEN,), dtype='float32')

        self._model_path = Sequential().add(self._path_input)
        self._model_path.add(Bidirectional(LSTM(CASE_EMBED_LEN,
                                                input_shape=(MAX_PATH_LEN, CASE_EMBED_LEN),
                                                return_sequences=True)))
        self._model_path.add(Dropout(0.1))
        self._model_path.add(AttentionDecoder(CASE_EMBED_LEN, UTTERANCE_WORD_EMBED_LEN),return_probabilities=True)
        # self._model_path.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        self._model_utterance = Sequential().add(self._utterance_input)
        self._model_utterance.add(Bidirectional(LSTM(UTTERANCE_WORD_EMBED_LEN,
                                                     input_shape=(MAX_UTURRENCE_LEN, UTTERANCE_WORD_EMBED_LEN),
                                                     return_sequences=True)))

        self._model_utterance.add(Dropout(0.1))
        self._model_utterance.add(AttentionDecoder(UTTERANCE_WORD_EMBED_LEN, CASE_EMBED_LEN,return_probabilities=True))
        # self._model_utterance.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])