"""DyNA-PPO explorer."""
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.tree
import tensorflow as tf
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import tf_agents.policies.tf_policy

import flexs
from flexs import baselines
from flexs.baselines.explorers.environments.dyna_ppo import (
    DynaPPOEnvironment as DynaPPOEnv,
    DynaPPOEnvironmentStoppableEpisode as DynaPPOStoppableEnv
)
from flexs.baselines.explorers.environments.dyna_ppo import (
    DynaPPOEnvironmentMutative as DynaPPOEnvMut,
)
from flexs.utils import sequence_utils as s_utils


class DynaPPOEnsemble(flexs.Model):
    """
    Ensemble from DyNAPPO paper.

    Ensembles many models together but only uses those with an $r^2$ above
    a certain threshold (on validation data) at test-time.
    """

    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        r_squared_threshold: float = 0.2,
        models: Optional[List[flexs.Model]] = None,
        use_gaussian_process: bool = False,
    ):
        """Create the ensemble from `models`."""
        super().__init__(name="DynaPPOEnsemble")

        if models is None:
            models = [
                baselines.models.CNNEnsemble(seq_len, alphabet),
                baselines.models.SklearnRegressor(
                    sklearn.neighbors.KNeighborsRegressor,
                    alphabet,
                    "nearest_neighbors",
                    seq_len,
                    hparam_tune=True,
                    hparams_to_search={
                        'n_neighbors': [2, 5, 10, 15],
                    },
                    nfolds=5,
                ),
                baselines.models.BayesianRidge(
                    alphabet,
                    seq_len,
                    hparam_tune=True,
                    hparams_to_search={
                        'alpha_1': [1e-5, 1e-6, 1e-7],
                        'alpha_2': [1e-5, 1e-6, 1e-7],
                        'lambda_1': [1e-5, 1e-6, 1e-7],
                        'lambda_1': [1e-5, 1e-6, 1e-7],
                    },
                    nfolds=5,
                ),
                baselines.models.RandomForest(
                    alphabet,
                    seq_len,
                    hparam_tune=True,
                    hparams_to_search={
                        'max_depth': [8, None],
                        'max_features': [seq_len // 4, seq_len // 2, seq_len],
                        'n_estimators': [10, 100, 200],
                    },
                    nfolds=5,
                ),
                baselines.models.SklearnRegressor(
                    sklearn.tree.ExtraTreeRegressor,
                    alphabet,
                    "extra_trees",
                    seq_len,
                    hparam_tune=True,
                    hparams_to_search={
                        'max_depth': [8, None],
                        'max_features': [seq_len // 4, seq_len // 2, seq_len],
                    },
                    nfolds=5,
                ),
                baselines.models.SklearnRegressor(
                    sklearn.ensemble.GradientBoostingRegressor,
                    alphabet,
                    "gradient_boosting",
                    seq_len,
                    hparam_tune=True,
                    hparams_to_search={
                        'max_depth': [8, None],
                        'max_features': [seq_len // 4, seq_len // 2, seq_len],
                        'learning_rate': [1., 1e-1, 1e-2],
                    },
                    nfolds=5,
                ),
            ]

            if use_gaussian_process:
                models.append(
                    baselines.models.SklearnRegressor(
                        sklearn.gaussian_process.GaussianProcessRegressor,
                        alphabet,
                        "gaussian_process",
                        seq_len,
                        hparam_tune=True,
                        hparams_to_search={
                            'kernel': [RBF(), RationalQuadratic(), Matern()],
                        },
                        nfolds=5,
                    )
                )

        self.models = models
        self.r_squared_vals = np.ones(len(self.models))
        self.r_squared_threshold = r_squared_threshold

    def _train(self, sequences, labels):
        if len(sequences) < 10:
            return

        self.r_squared_vals = [
            model.train(sequences, labels)
            for model in self.models
        ]

    def _fitness_function(self, sequences):
        passing_models = [
            model
            for model, r_squared in zip(self.models, self.r_squared_vals)
            if r_squared >= self.r_squared_threshold
        ]

        if len(passing_models) == 0:
            val = np.argmax(self.r_squared_vals)
            return self.models[val].get_fitness(sequences)
            #return self.models[np.argmax(self.r_squared_vals)].get_fitness(sequences)

        return np.mean(
            [model.get_fitness(sequences) for model in passing_models], axis=0
        )


    def _fitness_function_uncert(self, sequences):
        passing_models = [
            model
            for model, r_squared in zip(self.models, self.r_squared_vals)
            if r_squared >= self.r_squared_threshold
        ]

        if len(passing_models) == 0:
            val = np.argmax(self.r_squared_vals)
            return self.models[val].get_fitness(sequences), np.zeros(len(sequences))
            #return self.models[np.argmax(self.r_squared_vals)].get_fitness(sequences)

        preds = np.array([model.get_fitness(sequences) for model in passing_models])

        return preds.mean(axis=0), preds.std(axis=0)

class DummySeqLenRewardEnsemble(flexs.Model):
    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        r_squared_threshold: float = 0.5,
        models: Optional[List[flexs.Model]] = None,
    ):
        """Create the ensemble from `models`."""
        super().__init__(name="DummySeqLenRewardEnsemble")

    def _train(self, sequences, labels):
        return

    def _fitness_function(self, sequences):
        return np.array([len(seq) for seq in sequences], dtype=np.float32)

    def _fitness_function_uncert(self, sequences):
        return (
            np.array([len(seq) for seq in sequences], dtype=np.float32),
            np.zeros(len(sequences), dtype=np.float32)
        )

class DynaPPO(flexs.Explorer):
    """
    Explorer which implements DynaPPO.

    This RL-based sequence design algorithm works as follows:
        for r in rounds:
            train_policy(experimental_data_rewards[r])
            for m in model_based_rounds:
                train_policy(model_fitness_rewards[m])

    An episode for the agent begins with an empty sequence, and at
    each timestep, one new residue is generated and added to the sequence
    until the desired length of the sequence is reached. The reward
    is zero at all timesteps until the last one, when the reward is
    `reward = lambda * sequence_density + sequence_fitness` where
    sequence density is the density of nearby sequences already proposed.

    As described above, this explorer generates sequences *constructively*.

    Paper: https://openreview.net/pdf?id=HklxbgBKvr
    """

    def __init__(
        self,
        landscape: flexs.Landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
        model: Optional[flexs.Model] = None,
        num_experiment_rounds: int = 10,
        num_model_rounds: int = 1,
        env_batch_size: int = 4,
        min_proposal_seq_len: int = 7,
        lr=1e-4,
        agent_train_epochs=10,
        penalty_scale = 0.1,
        distance_radius = 2,
        use_dummy_model=False,
        use_gaussian_process=False,
        use_stoppable_env=True,
    ):
        """
        Args:
            num_experiment_rounds: Number of experiment-based rounds to run. This is by
                default set to 10, the same number of sequence proposal of rounds run.
            num_model_rounds: Number of model-based rounds to run.
            env_batch_size: Number of epsisodes to batch together and run in parallel.

        """
        tf.config.run_functions_eagerly(False)

        name = f"DynaPPO_Agent_{num_experiment_rounds}_{num_model_rounds}"

        if model is None:
            if use_dummy_model:
                model = DummySeqLenRewardEnsemble(
                    len(starting_sequence),
                    alphabet,
                )

            else:
                model = DynaPPOEnsemble(
                    60,
                    #len(starting_sequence),
                    alphabet,
                    use_gaussian_process=use_gaussian_process,
                )
            # Some models in the ensemble need to be trained on dummy dataset before
            # they can predict
            #model.train(
            #    s_utils.generate_random_sequences(len(starting_sequence), 10, alphabet),
            #    [0] * 10,
            #)

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds
        self.env_batch_size = env_batch_size
        self.min_proposal_seq_len = min_proposal_seq_len

        env_type = DynaPPOStoppableEnv if use_stoppable_env else DynaPPOEnv
        env = env_type(
            self.alphabet,
            len(starting_sequence),
            model,
            landscape,
            env_batch_size,
            penalty_scale=penalty_scale,
            distance_radius=distance_radius
        )
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(), fc_layer_params=[128]
        )

        print(self.tf_env.action_spec())
        self.agent = ppo_agent.PPOAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=agent_train_epochs,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

        self.inner_rounds_iter = 0
        self.should_terminate_round = False
        self.highest_uncert = 0.0
        self.uncert_thresh = 0.5

        self.dataset_seqs = set(landscape.get_full_dataset()[0])
        print('heyo')

    def add_last_seq_in_trajectory(self, experience, new_seqs):
        """Add the last sequence in an episode's trajectory.

        Given a trajectory object, checks if the object is the last in the trajectory.
        Since the environment ends the episode when the score is non-increasing, it
        adds the associated maximum-valued sequence to the batch.

        If the episode is ending, it changes the "current sequence" of the environment
        to the next one in `last_batch`, so that when the environment resets, mutants
        are generated from that new sequence.
        """
        for is_bound, obs, reward in zip(experience.is_boundary(), experience.observation, experience.reward):
            if is_bound:
                seq = s_utils.one_hot_to_string(obs.numpy(), self.alphabet)
                new_seqs[seq] = reward.numpy()

                if self.tf_env.fitness_model_is_gt:
                    continue

                uncert = self.tf_env.get_cached_uncertainty(seq)
                if self.inner_rounds_iter == 1 and uncert >= self.highest_uncert:
                    self.highest_uncert = uncert
                elif self.inner_rounds_iter > 1 and uncert >= (1 + self.uncert_thresh) * self.highest_uncert:
                    self.should_terminate_round = True

    def _is_seq_long_enough(self, seq):
        return len(seq) >= self.min_proposal_seq_len

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.env_batch_size,
            max_length=replay_buffer_capacity,
        )

        sequences = {}
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=sequences),
            ],
            num_episodes=1,
        )

        # Experiment-based training round. Each sequence we generate here must be
        # evaluated by the ground truth landscape model. So each sequence we evaluate
        # reduces our sequence proposal budget by one.
        # We amortize this experiment-based training cost to be 1/2 of the sequence
        # budget at round one and linearly interpolate to a cost of 0 by the last round.

        experiment_based_training_budget = self.sequences_batch_size
        self.tf_env.set_fitness_model_to_gt(True)
        previous_landscape_cost = self.tf_env.landscape.cost
        while (
            self.tf_env.landscape.cost - previous_landscape_cost
            < experiment_based_training_budget
        ):
            collect_driver.run()

        #tf_agents.policies.tf_policy.num_iters += 1

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()
        sequences.clear()

        # Model-based training rounds
        self.should_terminate_round = False
        self.inner_rounds_iter = 1
        self.tf_env.set_fitness_model_to_gt(False)
        previous_model_cost = self.model.cost
        for _ in range(self.num_model_rounds):
            if self.model.cost - previous_model_cost >= self.model_queries_per_batch:
                break

            previous_round_model_cost = self.model.cost
            while self.model.cost - previous_round_model_cost < int(
                self.model_queries_per_batch / self.num_model_rounds
            ):
                collect_driver.run()
                if self.should_terminate_round:
                    break

            trajectories = replay_buffer.gather_all()
            rewards = trajectories.reward.numpy()[0]
            mask = trajectories.is_last().numpy()[0]

            masked_reward = rewards[mask]
            mean_reward = masked_reward.mean()

            self.agent.train(experience=trajectories)
            replay_buffer.clear()
            self.inner_rounds_iter += 1


        measured_seqs = self.dataset_seqs.union(set(measured_sequences_data["sequence"]))
        is_usable_seq = (
            lambda x: x not in measured_seqs and self._is_seq_long_enough(x)
        )

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        to_propose = {
            seq: fitness
            for seq, fitness in sequences.items()
            if is_usable_seq(seq)
        }

        while len(to_propose) < self.sequences_batch_size:
            previous_round_model_cost = self.model.cost
            while self.model.cost - previous_round_model_cost < int(
                self.model_queries_per_batch / self.num_model_rounds
            ):
                collect_driver.run()

            to_propose = {
                seq: fitness
                for seq, fitness in sequences.items()
                if is_usable_seq(seq)
            }

        new_seqs = np.array(list(to_propose.keys()))
        preds = np.array(list(to_propose.values()))
        sorted_order = np.argsort(preds)[::-1][: self.sequences_batch_size]

        return new_seqs[sorted_order], preds[sorted_order]


class DynaPPOMutative(flexs.Explorer):
    """
    Explorer which implements DynaPPO.

    Note that unlike the other DynaPPO explorer, this one is mutative rather than
    constructive. Specifically, instead of starting from an empty sequence
    and generating residues one-by-one, this explorer starts from a complete
    sequence (fitness thresholds to start with good sequences) and mutates it
    until the mutant's fitness has started to decrease. Then it ends the episode.

    This has proven to be a stronger algorithm than the original DyNAPPO.

    Paper: https://openreview.net/pdf?id=HklxbgBKvr
    """

    def __init__(
        self,
        landscape: flexs.Landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
        model: Optional[flexs.Model] = None,
        num_experiment_rounds: int = 10,
        num_model_rounds: int = 1,
    ):
        """
        Args:
            num_experiment_rounds: Number of experiment-based rounds to run. This is by
                default set to 10, the same number of sequence proposal of rounds run.
            num_model_rounds: Number of model-based rounds to run.

        """
        tf.config.run_functions_eagerly(False)

        name = f"DynaPPO_Agent_{num_experiment_rounds}_{num_model_rounds}"

        if model is None:
            model = DynaPPOEnsemble(
                len(starting_sequence),
                alphabet,
            )
            model.train(
                s_utils.generate_random_sequences(len(starting_sequence), 10, alphabet),
                [0] * 10,
            )

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds

        env = DynaPPOEnvMut(
            alphabet=self.alphabet,
            starting_seq=starting_sequence,
            model=model,
            landscape=landscape,
            max_num_steps=model_queries_per_batch,
        )
        validate_py_environment(env, episodes=1)
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        encoder_layer = tf.keras.layers.Lambda(lambda obs: obs["sequence"])
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )

        self.agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=10,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

    def add_last_seq_in_trajectory(self, experience, new_seqs):
        """Add the last sequence in an episode's trajectory.

        Given a trajectory object, checks if the object is the last in the trajectory.
        Since the environment ends the episode when the score is non-increasing, it
        adds the associated maximum-valued sequence to the batch.

        If the episode is ending, it changes the "current sequence" of the environment
        to the next one in `last_batch`, so that when the environment resets, mutants
        are generated from that new sequence.
        """
        if experience.is_boundary():
            seq = s_utils.one_hot_to_string(
                experience.observation["sequence"].numpy()[0], self.alphabet
            )
            new_seqs[seq] = experience.observation["fitness"].numpy().squeeze()

            top_fitness = max(new_seqs.values())
            top_sequences = [
                seq for seq, fitness in new_seqs.items() if fitness >= 0.9 * top_fitness
            ]
            if len(top_sequences) > 0:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(top_sequences)
            else:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(
                    [seq for seq, _ in new_seqs.items()]
                )

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        num_parallel_environments = 1
        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        sequences = {}
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=sequences),
            ],
            num_episodes=1,
        )

        # Experiment-based training round. Each sequence we generate here must be
        # evaluated by the ground truth landscape model. So each sequence we evaluate
        # reduces our sequence proposal budget by one.
        # We amortize this experiment-based training cost to be 1/2 of the sequence
        # budget at round one and linearly interpolate to a cost of 0 by the last round.
        current_round = measured_sequences_data["round"].max()
        experiment_based_training_budget = int(
            (self.rounds - current_round + 1)
            / self.rounds
            * self.sequences_batch_size
            / 2
        )
        self.tf_env.envs[0].set_fitness_model_to_gt(True)
        previous_landscape_cost = self.tf_env.envs[0].landscape.cost
        while (
            self.tf_env.envs[0].landscape.cost - previous_landscape_cost
            < experiment_based_training_budget
        ):
            collect_driver.run()

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()
        sequences.clear()

        # Model-based training rounds
        self.tf_env.envs[0].set_fitness_model_to_gt(False)
        previous_model_cost = self.model.cost
        for _ in range(self.num_model_rounds):
            if self.model.cost - previous_model_cost >= self.model_queries_per_batch:
                break

            previous_round_model_cost = self.model.cost
            while self.model.cost - previous_round_model_cost < int(
                self.model_queries_per_batch / self.num_model_rounds
            ):
                collect_driver.run()

            trajectories = replay_buffer.gather_all()
            self.agent.train(experience=trajectories)
            replay_buffer.clear()

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        sequences = {
            seq: fitness
            for seq, fitness in sequences.items()
            if seq not in set(measured_sequences_data["sequence"])
        }
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[
            : -(self.sequences_batch_size - experiment_based_training_budget) : -1
        ]

        return new_seqs[sorted_order], preds[sorted_order]
