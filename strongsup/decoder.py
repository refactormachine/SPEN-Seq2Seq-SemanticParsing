import sys
from collections import namedtuple

import tensorflow as tf
import numpy as np

from gtd.utils import flatten
from strongsup.case_weighter import get_case_weighter
from strongsup.tests.keras_decomposable_attention import build_model
from strongsup.value_function import get_value_function, ValueFunctionExample
from strongsup.value import check_denotation


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
                 utter_len, max_stack_size, train_one_hot=True):
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
                config.value_function, parse_model.parse_model)
        self._case_weighter = get_case_weighter(
                config.case_weighter, parse_model.parse_model,
                self._value_function)
        self._config = config
        self._caching = config.inputs_caching
        self._domain = domain
        self._path_checker = domain.path_checker
        self._utter_len = utter_len
        self._max_stack_size = max_stack_size
        self._train_step_count = 0
        self._class_weight = {0: 1., 1: 5.}
        self._train_one_hot = train_one_hot

        # Normalization and update policy
        self._normalization = config.normalization
        if config.normalization == NormalizationOptions.GLOBAL:
            raise ValueError('Global normalization is no longer supported.')
        self._predicate2index = self._build_predicate_dictionary(predicates)

        # 100 is the glove embedding length per word
        shape_utt = (self._utter_len, 100, 2)
        shape_path = (self._max_stack_size, len(self.predicate_dictionary), 2) if self._train_one_hot else (self._max_stack_size, 96, 2)
        settings = {'lr': 0.001, 'dropout': 0.2, 'gru_encode': True}
        self._decomposable = build_model(shape_utt, shape_path, settings)

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
            one_hot_decision[pred_dict[decision.name]] = 1
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
        cumul = [0]         # Used to group the results back
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
            grouped_attentions.append(attentions[cumul[i]:cumul[i+1]])
            grouped_subscores.append(subscores[cumul[i]:cumul[i+1]])
        return grouped_attentions, grouped_subscores

    ################################
    # Training

    def train_step(self, examples):
        # sample a beam of logical forms for each example
        beams = self.predictions(examples, train=True)

        if self._train_one_hot:
            self.train_decomposable_batches_one_hot(beams, examples)
        else:
            self.train_decomposable_batches_on_stack(beams, examples)

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

    def train_decomposable_batches_one_hot(self, beams, examples):
        self._train_step_count += 1

        if self._train_step_count < 80:
            return

        for example, beam in zip(examples, beams):
            beam_batch = [[], []]
            if len(beam._paths) == 0:
                continue

            y_hat = np.zeros((len(beam._paths), 2), dtype='int32')

            utter_embds = []
            for utter in beam._paths[0].context.utterances:
                for token in utter._tokens:
                    utter_embds += [self._glove_embeddings[token]]
            utter_embds = np.array(utter_embds)

            utter_embds_np = np.concatenate((
                utter_embds,
                np.full((self._utter_len - len(utter_embds), 100), 0.)
            ))

            one_hot_vec_dim = len(self.predicate_dictionary)

            for idx, path in enumerate(beam._paths):
                check_denote = int(check_denotation(example.answer, path.finalized_denotation))
                y_hat[idx, check_denote] = 1
                decisions_one_hot = self.decisions_to_one_hot(path.decisions)

                decisions_one_hot = np.concatenate((
                    decisions_one_hot,
                    np.full((self._max_stack_size - len(decisions_one_hot), one_hot_vec_dim), 0.)
                ))

                beam_batch[0].append(utter_embds_np)
                beam_batch[1].append(decisions_one_hot)
            beam_batch[0] = np.array(beam_batch[0])
            beam_batch[1] = np.array(beam_batch[1])
            output = self._decomposable.train_on_batch(beam_batch, y_hat, class_weight=self._class_weight)

            if self._train_step_count % 100 == 0:
                print >> sys.stderr, '\rprediction: \r{}\routput: {} \033[F'.format(
                    self._decomposable.predict(beam_batch, batch_size=len(beam_batch[0]), verbose=0),
                    output
                )

    def train_decomposable_batches_on_stack(self, beams, examples):
        self._train_step_count += 1

        if self._train_step_count < 80:
            return

        for example, beam in zip(examples, beams):
            beam_batch = [[], []]
            if len(beam._paths) == 0:
                continue

            y_hat = np.zeros((len(beam._paths), 2), dtype='int32')

            utter_embds = []
            for utter in beam._paths[0].context.utterances:
                for token in utter._tokens:
                    utter_embds += [self._glove_embeddings[token]]
            utter_embds = np.array(utter_embds)

            utter_embds = np.concatenate((
                utter_embds,
                np.full((self._utter_len - len(utter_embds), 100), 0.)
            ))

            for idx, path in enumerate(beam._paths):
                check_denote = int(check_denotation(example.answer, path.finalized_denotation))
                y_hat[idx, check_denote] = 1

                # define variables to fetch
                fetch = {
                    'stack_embedder': self.parse_model._parse_model._stack_embedder.embeds,
                }

                # fetch variables
                sess = tf.get_default_session()
                if sess is None:
                    raise ValueError('No default TensorFlow Session registered.')
                feed = self.parse_model._parse_model._stack_embedder.inputs_to_feed_dict(path._cases)
                result = sess.run(fetch, feed_dict=feed)
                stack_embedder = result['stack_embedder']  # stack_embedder_dim:96

                stack_embedder = np.concatenate((
                    stack_embedder,
                    np.full((self._max_stack_size - len(stack_embedder), 96), 0.)
                ))

                beam_batch[0].append(utter_embds)
                beam_batch[1].append(stack_embedder)
            beam_batch[0] = np.array(beam_batch[0])
            beam_batch[1] = np.array(beam_batch[1])

            output = self._decomposable.train_on_batch(beam_batch, y_hat, class_weight=self._class_weight)

            if self._train_step_count % 10 == 0:
                print >> sys.stderr, '\rprediction: \r{}\routput: {} \033[F'.format(
                    self._decomposable.predict(beam_batch, batch_size=len(beam_batch[0]), verbose=0),
                    output
                )
