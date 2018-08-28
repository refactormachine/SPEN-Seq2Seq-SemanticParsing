import csv
import os
import random
from collections import namedtuple

import numpy as np
import tensorflow as tf
from gtd.utils import flatten
from gtd.chrono import verboserate
from keras.utils.np_utils import to_categorical

from strongsup.case_weighter import get_case_weighter
from strongsup.decomposable import decomposable_model_generation
from strongsup.value import check_denotation
from strongsup.value_function import get_value_function, ValueFunctionExample


class NormalizationOptions(object):
    """Constants for normalization options"""
    LOCAL = 'local'
    GLOBAL = 'global'


# used by the Decoder to compute gradients
WeightedCase = namedtuple('WeightedCase', ['case', 'weight'])


class Decoder(object):
    """A decoder does two things:
    - Given a batch of examples, produce a Beam (list of ParsePaths) for each example.
        Internally it uses an ExplorationPolicy to produce beams, and a ParseModel
        to score the ParseCases.
    - Given a batch of Beams, update the model parameters by passing appropriate
        ParseCases to the TrainParseModel.
    """

    def __init__(self, parse_model, config, domain, glove_embeddings, predicates,
                 utter_len, max_stack_size, tb_logger,
                 predicate_embedder_type='one_hot', decomposable_weights_file=None):
        """Create a new decoder.

        Args:
            parse_model (TrainParseModel)
            config (Config): The decoder section of the config
            domain (Domain)
            glove_embeddings
            predicates
            utter_len
            max_stack_size
        """
        self._glove_embeddings = glove_embeddings
        self._parse_model = parse_model
        self._value_function = get_value_function(
            config.value_function, parse_model.parse_model) if parse_model else None
        self._case_weighter = get_case_weighter(
            config.case_weighter, parse_model.parse_model,
            self._value_function) if parse_model else None
        self._config = config
        self._caching = config.inputs_caching
        self._domain = domain
        self._path_checker = domain.path_checker
        self._utter_len = utter_len
        self._max_stack_size = max_stack_size
        self._tb_logger = tb_logger
        self._class_weight = {0: 1., 1: 5.}
        self._predicate_embedder_type = predicate_embedder_type
        self.correct_predictions = 0
        self.all_predictions = 0
        self._decomposable_data = None
        self._decomposable_weights_file = decomposable_weights_file

        # Normalization and update policy
        self._normalization = config.normalization
        if config.normalization == NormalizationOptions.GLOBAL:
            raise ValueError('Global normalization is no longer supported.')
        self._predicate2index = self._build_predicate_dictionary(predicates)

        # 100 is the glove embedding length per word
        max_len_utter = 100
        hidden_layers_num = 15
        settings = {'lr': 0.0001, 'dropout': 0.05}
        classifications = 2
        self._decomposable = decomposable_model_generation(
            (self._utter_len, max_len_utter), (self._max_stack_size, len(self.predicate_dictionary)),
            hidden_layers_num, classifications, settings)

        # if decomposable_weights_file and os.path.isfile(decomposable_weights_file):
        #     self._decomposable.load_weights(decomposable_weights_file)

        # Exploration policy
        # TODO: Resolve this circular import differently
        from strongsup.exploration_policy import get_exploration_policy
        self._test_exploration_policy = get_exploration_policy(
            self, config.test_exploration_policy,
            self._normalization, train=False)
        self._train_exploration_policy = get_exploration_policy(
            self, config.train_exploration_policy,
            self._normalization, train=True)

    @property
    def parse_model(self):
        return self._parse_model

    @property
    def caching(self):
        return self._caching

    @property
    def domain(self):
        return self._domain

    @property
    def step(self):
        return self._parse_model.step

    @property
    def predicate_dictionary(self):
        return self._predicate2index

    @property
    def decomposable_data(self):
        return self._decomposable_data

    @staticmethod
    def _build_predicate_dictionary(predicates):
        predicate_dict = {}
        for i, predicate in enumerate(predicates):
            predicate_dict[predicate.name] = i
        return predicate_dict

    def exploration_policy(self, train):
        """Returns the train or test exploration policy depending on
        train

        Args:
            train (bool)

        Returns:
            ExplorationPolicy
        """
        if train:
            return self._train_exploration_policy
        else:
            return self._test_exploration_policy

    def path_checker(self, path):
        """Return False if the ParsePath should be pruned away; True otherwise.

        Args:
            path (ParsePath)
        Returns:
            bool
        """
        return self._path_checker(path)

    def get_probs(self, beam):
        """Return a numpy array containing the probabilities of the paths
        in the given beam.

        The entries may not sum to 1 for local normalization since we have
        pruned away choices that are not executable.

        Args:
            beam (Beam)
        Returns:
            np.array of length len(beam) containing the probabilities.
        """
        if len(beam) == 0:
            return np.zeros(0)
        if self._normalization == NormalizationOptions.LOCAL:
            return np.exp(np.array([path.log_prob for path in beam]))
        else:
            stuff = np.array([path.score for path in beam])
            stuff = np.array(stuff - np.min(stuff))
            return stuff / np.sum(stuff)

    ################################
    # Prediction

    def predictions(self, examples, train, verbose=False):
        """Return the final beams for a batch of contexts.

        Args:
            examples
            train (bool): If you're training or evaluating
            verbose (bool)

        Returns:
            list[Beam]: a batch of Beams
        """
        exploration_policy = self.exploration_policy(train)
        beams = exploration_policy.get_beams(examples, verbose)
        return [beam.get_terminated() for beam in beams]

    def get_intermediate_beams(self, examples, train, verbose=False):
        exploration_policy = self.exploration_policy(train)
        return exploration_policy.get_intermediate_beams(examples, verbose)

    def decisions_to_one_hot(self, decisions):
        pred_dict = self.predicate_dictionary
        one_hot_decisions = np.empty(shape=(len(decisions), len(pred_dict)))

        for i, decision in enumerate(decisions):
            one_hot_decision = np.zeros(shape=len(pred_dict))
            one_hot_decision[pred_dict[decision]] = 1
            one_hot_decisions[i] = one_hot_decision
        return np.array(one_hot_decisions)

    def score_breakdown(self, paths):
        """Return the logits for all (parse case, choice, scorer) tuples.

        Args:
            paths (list[ParsePath])
        Returns:
            grouped_attentions:
                a list of length(paths). Each entry is an np.array of shape
                (>= len(utterance)) containing the attention scores
            grouped_subscores:
                a list of length len(paths). Each entry is an np.array of shape
                (>= number of cases, len(choices), number of scorers)
                containing the logits of each scorer on each choice.
                By default there are 3 scorers: basic, attention, and soft copy.
        """
        if len(paths) == 0:
            return [], []
        cumul = [0]  # Used to group the results back
        cases = []
        for path in paths:
            for case in path:
                cases.append(case)
            cumul.append(len(cases))
        # Get the scores from the model
        attentions, subscores = self._parse_model.score_breakdown(cases, ignore_previous_utterances=False,
                                                                  caching=False)
        # Group the scores by paths
        grouped_attentions, grouped_subscores = [], []
        for i in xrange(len(paths)):
            grouped_attentions.append(attentions[cumul[i]:cumul[i + 1]])
            grouped_subscores.append(subscores[cumul[i]:cumul[i + 1]])
        return grouped_attentions, grouped_subscores

    ################################
    # Training

    def train_step(self, examples):
        # sample a beam of logical forms for each example
        beams = self.predictions(examples, train=True)

        self._decomposable_data = self.train_decomposable_batches(beams, examples)

        all_cases = []  # a list of ParseCases to give to ParseModel
        all_case_weights = []  # the weights associated with the cases
        for example, paths in zip(examples, beams):
            case_weights = self._case_weighter(paths, example)
            case_weights = flatten(case_weights)
            cases = flatten(paths)
            assert len(case_weights) == sum(len(p) for p in paths)

            all_cases.extend(cases)
            all_case_weights.extend(case_weights)

        # for efficiency, prune cases with weight 0
        cases_to_reinforce = []
        weights_to_reinforce = []
        for case, weight in zip(all_cases, all_case_weights):
            if weight != 0:
                cases_to_reinforce.append(case)
                weights_to_reinforce.append(weight)

        # update value function
        vf_examples = []
        for example, paths in zip(examples, beams):
            vf_examples.extend(ValueFunctionExample.examples_from_paths(paths, example))
        self._value_function.train_step(vf_examples)

        # update parse model
        self._parse_model.train_step(
            cases_to_reinforce, weights_to_reinforce, caching=False)

    def decisions_embedder(self, decisions, path_cases=None):
        """
        predicate_embedder_type can either be 'one_hot' or 'stack_embedder'
        :param decisions: path._decisions.name from the beam path
        :param path_cases: path._cases from the beam path
        :return:
        """
        if self._predicate_embedder_type == 'stack_embedder':
            # define variables to fetch
            fetch = {
                'stack_embedder': self.parse_model._parse_model._stack_embedder.embeds
            }

            # fetch variables
            sess = tf.get_default_session()
            if sess is None:
                raise ValueError('No default TensorFlow Session registered.')

            feed = self.parse_model._parse_model._stack_embedder.inputs_to_feed_dict(path_cases)
            result = sess.run(fetch, feed_dict=feed)
            decisions_embedder = result['stack_embedder']  # stack_embedder_dim:96
            dim = 96
        else:  # assume it's 'one_hot'
            decisions_embedder = self.decisions_to_one_hot(decisions)
            dim = len(self.predicate_dictionary)

        decisions_embedder = np.concatenate((
            decisions_embedder,
            np.full((self._max_stack_size - len(decisions_embedder), dim), 0.)
        ))

        return decisions_embedder

    def train_decomposable_batches(self, beams, examples):
        y_hat_batch, decisions, utterances, beam_scores = [], [], [], []

        for example, beam in zip(examples, beams):
            if len(beam._paths) == 0:
                continue

            beam_batch_correct = False
            sentence_for_print = ''
            curr_decisions, curr_utterances, curr_y_hat_batch, curr_beam_scores = [], [], [], []

            for utter in beam._paths[0].context.utterances:
                for token in utter._tokens:
                    sentence_for_print += token + ' '

            for path in beam._paths:
                check_denote = int(check_denotation(example.answer, path.finalized_denotation))
                curr_y_hat_batch.append(check_denote)
                full_decision_for_print = ''

                if check_denote:
                    beam_batch_correct = True

                for decision in path.decisions:
                    full_decision_for_print += ' ' + decision._name

                curr_decisions.append(full_decision_for_print)
                curr_utterances.append(sentence_for_print)
                curr_beam_scores.append(path.score)

            # at least one correct path
            if not beam_batch_correct:
                continue

            # append to result vectors
            decisions.extend(curr_decisions)
            utterances.extend(curr_utterances)
            y_hat_batch.extend(curr_y_hat_batch)
            beam_scores.extend(curr_beam_scores)

        decomposable_data = [[utter, dec, y, score]
                             for utter, dec, y, score in zip(utterances, decisions, y_hat_batch, beam_scores)]

        return decomposable_data

    def train_decomposable_from_csv(self, csv_file):
        utterances, decisions, y_hats, beam_scores = [], [], [], []

        with open(csv_file, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            curr_utterances, curr_decisions, curr_y_hats, curr_beam_scores = [], [], [], []
            prev_utterance, prev_decision, prev_y_hat, prev_beam_score = None, None, None, None
            skip = False  # we only want the first and last example of each batch

            for utterance, decision, y_hat, beam_score in csv_reader:
                y_hat, beam_score = float(y_hat), float(beam_score)

                if not prev_utterance:  # first time initialization
                    prev_utterance = utterance
                    prev_decision = decision
                    prev_y_hat = y_hat
                    prev_beam_score = beam_score

                if prev_utterance != utterance:
                    # append the last example of the previous batch
                    curr_utterances.append(prev_utterance)
                    curr_decisions.append(prev_decision)
                    curr_y_hats.append(prev_y_hat)
                    curr_beam_scores.append(prev_beam_score)

                    # append to complete prev batch
                    utterances.extend(curr_utterances)
                    decisions.extend(curr_decisions)
                    y_hats.extend(curr_y_hats)
                    beam_scores.extend(curr_beam_scores)

                    curr_utterances, curr_decisions, curr_y_hats, curr_beam_scores = [], [], [], []
                    skip = False

                prev_utterance = utterance
                prev_decision = decision
                prev_y_hat = y_hat
                prev_beam_score = beam_score

                if not skip:  # append the first
                    curr_utterances.append(utterance)
                    curr_decisions.append(decision)
                    curr_y_hats.append(y_hat)
                    curr_beam_scores.append(beam_score)
                    skip = True

            # append the last batch after we finish reading csv
            curr_utterances.append(prev_utterance)
            curr_decisions.append(prev_decision)
            curr_y_hats.append(prev_y_hat)
            curr_beam_scores.append(prev_beam_score)

            utterances.extend(curr_utterances)
            decisions.extend(curr_decisions)
            y_hats.extend(curr_y_hats)
            beam_scores.extend(curr_beam_scores)

        # start training one batch per epoch
        num_batches = 32
        epochs = verboserate(xrange(1000000), desc='Training decomposable model')

        for epoch in epochs:
            # sample a batch
            batch_indices = random.sample(xrange(1, len(decisions)), num_batches)

            curr_utterances = np.array(np.array(utterances)[batch_indices])
            curr_decisions = np.array(np.array(decisions)[batch_indices])
            curr_y_hats = np.array(np.array(y_hats)[batch_indices])
            curr_beam_scores = np.array(np.array(beam_scores)[batch_indices])

            # randomize batch
            randomize = np.arange(len(curr_utterances))
            np.random.shuffle(randomize)
            curr_utterances = curr_utterances[randomize]
            curr_decisions = curr_decisions[randomize]
            curr_y_hats = curr_y_hats[randomize]
            curr_beam_scores = curr_beam_scores[randomize]

            self.train_decomposable_on_example(
                curr_utterances,
                curr_decisions,
                curr_y_hats,
                curr_beam_scores,
                epoch)

            if epoch % 100000 == 0:
                self._decomposable.save_weights(self._decomposable_weights_file.format(epoch))

    def train_decomposable_on_example(self, utters_batch, decisions_batch, y_hats_batch,
                                      beam_scores_batch, step):
        train_decisions_batch, train_utters_batch, y_hat_batch = \
            self.get_trainable_batches(utters_batch,
                                       decisions_batch,
                                       y_hats_batch,
                                       beam_scores_batch)

        # loss, accuracy = self._decomposable.train_on_batch(beam_batch, beam_score_batch)
        loss, accuracy = self._decomposable.train_on_batch(
            [train_utters_batch, train_decisions_batch],
            to_categorical(y_hat_batch))
        # prediction = self._decomposable.predict_on_batch([train_utters_batch, train_decisions_batch])

        reranked_predictions, learning_to_rank = \
            self.pairwise_approach(train_utters_batch, train_decisions_batch, y_hat_batch)

        self._tb_logger.log('decomposableLoss', loss, step)
        self._tb_logger.log('decomposableAccuracy', accuracy, step)
        self._tb_logger.log('decomposableRanker', learning_to_rank, step)

        if step % 1000 == 0:
            print '\nRanking index: {} at step {}'.format(learning_to_rank, step)
            print 'Loss: {} Accuracy before: {} '.format(loss, accuracy)

    def get_trainable_batches(self, utters_batch, decisions_batch, y_hats_batch, beam_scores_batch):
        train_utters_batch = []
        train_decisions_batch = []
        beam_score_batch = []
        y_hat_batch = []

        for decision, utter, y_hat, beam_score in zip(decisions_batch, utters_batch, y_hats_batch, beam_scores_batch):
            decision_tokens = decision.split()

            utter_embds = []
            for token in utter.split():
                utter_embds += [self._glove_embeddings[token]]

            utter_embds = np.array(utter_embds)
            utter_embds = np.concatenate((
                utter_embds,
                np.full((self._utter_len - len(utter_embds), 100), 0.)
            ))

            decisions_embedder = self.decisions_embedder(decision_tokens)

            train_utters_batch.append(utter_embds)
            train_decisions_batch.append(decisions_embedder)
            beam_score_batch.append(beam_score)
            y_hat_batch.append(y_hat)
        train_utters_batch = np.array(train_utters_batch)
        train_decisions_batch = np.array(train_decisions_batch)
        beam_score_batch = np.array(beam_score_batch)
        y_hat_batch = np.array(y_hat_batch)

        return train_decisions_batch, train_utters_batch, y_hat_batch

    def pairwise_approach(self, utterances, decisions, y_hats):
        """
        Rerank a batch and calculate index of how good the reranker is.
        The index implements the pairwise approach to ranking.
        The goal for the ranker is to minimize the number of inversions in ranking,
        i.e. cases where the pair of results are in the wrong order relative to the ground truth.
        :param utterances: ndarray of shape (BATCH_SIZE, 2) of utterances (e.g. BATCH_SIZE=32)
        :param decisions: ndarray of shape (BATCH_SIZE, 2) of decisions (e.g. BATCH_SIZE=32)
        :param y_hats: ndarray of shape (BATCH_SIZE, 2) of ground truth labels to predictions (e.g. BATCH_SIZE=32)
        :return: A list of (utterances, decisions) and accuracy.
        The list's order assures that the True predictions come first.
        """
        true_predictions = [(pred, dec) for (pred, dec, y) in zip(utterances, decisions, y_hats) if y]
        false_predictions = [(pred, dec) for (pred, dec, y) in zip(utterances, decisions, y_hats) if not y]
        total_pairs = 0
        px_gt_py = 0
        reranked_predictions, reranked_y_hats = [], []

        for true_pred_utter, true_pred_dec in true_predictions:
            for false_pred_utter, false_pred_dec in false_predictions:
                train_params = [np.array([false_pred_utter, true_pred_utter]),
                                np.array([false_pred_dec, true_pred_dec])]
                prediction = self._decomposable.predict_on_batch(train_params)
                total_pairs += 1

                # rank the two examples
                prob_false = prediction[0][1]  # prob that false example is true - should be low
                prob_true = prediction[1][1]  # prob that true example is true - should be high

                # don't care about the actual class of the prediction
                # only care about the relative order of the probabilities
                if prob_false < prob_true:  # the true must be 'truer' than the false
                    px_gt_py += 1  # prediction is correct
                    # rank according to prediction
                    reranked_predictions.insert(0, prediction)
                    reranked_y_hats.insert(0, 1)
                else:
                    # rank according to prediction
                    reranked_predictions.append(prediction)
                    reranked_y_hats.append(0)

        return [np.array(reranked_predictions), np.array(reranked_predictions)], float(px_gt_py) / total_pairs
