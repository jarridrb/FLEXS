import flexs
import tensorflow as tf
import flexs.utils.sequence_utils as s_utils
from flexs import baselines

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

bert_gfp_landscape = flexs.landscapes.BertGFPBrightness()

starting_sequence = flexs.landscapes.BertGFPBrightness.starts['ed_10_wt']

dynappo_explorer = baselines.explorers.DynaPPO(
    landscape=bert_gfp_landscape,
    env_batch_size=10,
    num_model_rounds=10,
    rounds=10,
    starting_sequence=starting_sequence,
    sequences_batch_size=100,
    model_queries_per_batch=1000,
    alphabet=s_utils.AAS,
)

dynappo_sequences, metadata = dynappo_explorer.run(bert_gfp_landscape)
