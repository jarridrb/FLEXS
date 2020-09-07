import numpy as np

import flexs
from flexs.utils import sequence_utils as s_utils


class Random(flexs.Explorer):
    def __init__(
        self,
        model,
        landscape,
        rounds,
        mu,
        initial_sequence_data,
        experiment_budget,
        query_budget,
        alphabet,
        log_file=None,
        seed=None,
    ):
        name = f"Random_mu={mu}"

        super().__init__(
            model,
            landscape,
            name,
            rounds,
            experiment_budget,
            query_budget,
            initial_sequence_data,
            log_file,
        )
        self.mu = mu
        self.rng = np.random.default_rng(seed)
        self.alphabet = alphabet
        self.name = f"Random_mu{self.mu}"

    def propose_sequences(self, measured_sequences):
        """Propose `experiment_budget` samples."""

        old_sequences = measured_sequences["sequence"]
        old_sequence_set = set(old_sequences)
        new_seqs = set()

        while len(new_seqs) <= self.query_budget:
            seq = self.rng.choice(old_sequences)
            new_seq = s_utils.generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )

            if new_seq not in old_sequence_set:
                new_seqs.add(new_seq)

        new_seqs = np.array(list(new_seqs))
        preds = self.model.get_fitness(new_seqs)
        sorted_order = np.argsort(preds)[: -self.experiment_budget : -1]

        return new_seqs[sorted_order], preds[sorted_order]
