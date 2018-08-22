import csv
import os
import random
from collections import namedtuple

import numpy as np
import tensorflow as tf
from gtd.utils import flatten
from gtd.chrono import verboserate

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
        self._train_step_count = 0
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
        shape_utt = (self._utter_len, 100)
        shape_path = (self._max_stack_size, len(self.predicate_dictionary)) if \
            self._predicate_embedder_type == 'one_hot' else (self._max_stack_size, 96)
        settings = {'lr': 0.0001}
        self._decomposable = decomposable_model_generation(shape_utt, shape_path, settings)

        if decomposable_weights_file and os.path.isfile(decomposable_weights_file):
            print 'Using decomposable weights from file: {}'.format(decomposable_weights_file)
            self._decomposable.load_weights(decomposable_weights_file)

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
                # y_hat = [0, 0]
                # y_hat[check_denote] = 1
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

            BLOCK_SIZE = 20

            # pad to size BLOCK_SIZE
            while len(curr_decisions) < BLOCK_SIZE:
                curr_decisions.append(full_decision_for_print)
                curr_utterances.append(sentence_for_print)
                curr_y_hat_batch.append(0)
                curr_beam_scores.append(float('-inf'))

            # slice to size BLOCK_SIZE
            curr_decisions = curr_decisions[:BLOCK_SIZE]
            curr_utterances = curr_utterances[:BLOCK_SIZE]
            curr_y_hat_batch = curr_y_hat_batch[:BLOCK_SIZE]
            curr_beam_scores = curr_beam_scores[:BLOCK_SIZE]

            # append to result vectors
            decisions.extend(curr_decisions)
            utterances.extend(curr_utterances)
            y_hat_batch.extend(curr_y_hat_batch)
            beam_scores.extend(curr_beam_scores)

        decomposable_data = [[utter, dec, y, score]
                             for utter, dec, y, score in zip(utterances, decisions, y_hat_batch, beam_scores)]

        return decomposable_data

    def train_decomposable_from_csv(self, csv_file):
        self._train_step_count += 1

        with open(csv_file, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            utterances, decisions, y_hats, beam_scores = [], [], [], []
            curr_utterances, curr_decisions, curr_y_hats, curr_beam_scores = [], [], [], []
            lines_in_block = 0
            BATCH_SIZE = 1
            BLOCK_SIZE = 20

            for utterance, decision, y_hat, beam_score in csv_reader:
                curr_utterances.append(utterance)
                curr_decisions.append(decision)
                curr_y_hats.append(int(y_hat))
                curr_beam_scores.append(beam_score)
                lines_in_block += 1

                if lines_in_block == BLOCK_SIZE:
                    utterances.append(curr_utterances)
                    decisions.append(curr_decisions)
                    y_hats.append(curr_y_hats)
                    beam_scores.append(curr_beam_scores)

                    curr_utterances, curr_decisions, curr_y_hats, curr_beam_scores = [], [], [], []
                    lines_in_block = 0

            num_batches = len(decisions)
            num_of_accurate_predictions = 0
            iterations = verboserate(xrange(1000000), desc='Training decomposable model')

            for i in iterations:
                batch_indices = random.sample(xrange(1, num_batches), BATCH_SIZE)
                curr_utterances, curr_decisions, curr_y_hats, curr_beam_scores = [], [], [], []

                for j in batch_indices:
                    curr_utterances.extend(utterances[j])
                    curr_decisions.extend(decisions[j])
                    curr_y_hats.extend(y_hats[j])
                    curr_beam_scores.extend(beam_scores[j])
                    # TODO: TBD how to send these parameters
                    num_of_accurate_predictions += \
                        self.train_decomposable_on_example(curr_utterances, curr_decisions, curr_y_hats,
                                                           curr_beam_scores, i)

                if i % 1000 == 0:
                    self._decomposable.save_weights(self._decomposable_weights_file)
                    self._tb_logger.log('decomposableAccuracy', num_of_accurate_predictions, i)
                    num_of_accurate_predictions = 0

    def train_decomposable_on_example(self, utters, decisions, y_hats, beam_scores, step):
        if self._train_step_count < 0:
            return

        beam_batch = [[], []]
        y_hat_batch = []

        # for layer in self._decomposable.layers:
        #     weights = layer.get_weights()
        #     print weights

        for decision, utter, y_hat in zip(decisions, utters, y_hats):
            decision_tokens = decision.split()

            utter_embds = []
            for token in utter.split():
                utter_embds += [self._glove_embeddings[token]]

            utter_embds = np.array(utter_embds)
            utter_embds = np.concatenate((
                utter_embds,
                np.full((self._utter_len - len(utter_embds), 100), 0.)
            ))

            # y_hat_vec = [0, 0]
            # y_hat_vec[y_hat] = 1
            decisions_embedder = self.decisions_embedder(decision_tokens)

            beam_batch[0].append(utter_embds)
            beam_batch[1].append(decisions_embedder)
            # y_hat_batch.append(y_hat_vec)
            y_hat_batch.append(y_hat)
        beam_batch[0] = np.array(beam_batch[0])
        beam_batch[1] = np.array(beam_batch[1])
        y_hat_batch = np.array(y_hat_batch)

        random_order = np.random.permutation(len(y_hat_batch))
        y_hat_batch = y_hat_batch[random_order]
        beam_batch[0] = beam_batch[0][random_order]
        beam_batch[1] = beam_batch[1][random_order]

        loss, accuracy = self._decomposable.train_on_batch(beam_batch, y_hat_batch)
        # if i % 100 == 0:
        #     print 'loss: ' + str(loss) + ' accuracy: ' + str(accuracy)
        predicts = self._decomposable.predict_on_batch(beam_batch)

        is_accurate = 1 if y_hat_batch[np.argmax(predicts)] == 1 else 0
        self._tb_logger.log('decomposableLoss', loss, step)
        return is_accurate

