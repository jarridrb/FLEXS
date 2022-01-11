import flexs
import flexs.utils.sequence_utils as s_utils
import torch
from flexs import baselines

amp_landscape = flexs.landscapes.AMPLandscape(
    oracle_split='D2_target',
    oracle_type='MLP',
    oracle_features='AlBert',
    medoid_oracle_norm=1,
    proxy_data_split='D1',
    num_folds=5,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size=256
)

#starting_sequence = flexs.landscapes.AMPLandscape.starts[0]
starting_sequence = ''.join(['A' for _ in range(53)])

dynappo_explorer = baselines.explorers.DynaPPO(
    landscape=amp_landscape,
    env_batch_size=10,
    num_model_rounds=10,
    rounds=15,
    starting_sequence=starting_sequence,
    sequences_batch_size=100,
    model_queries_per_batch=1000,
    alphabet=s_utils.AAS,
)

dynappo_sequences, metadata = dynappo_explorer.run(amp_landscape)
dynappo_sequences.to_csv('dyna_ppo_seq.csv')
dynappo_sequences.to_csv('dyna_ppo_metadata.csv')
