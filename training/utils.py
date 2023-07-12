def batch_seed(training_hparams, ep):
    if training_hparams.data_order_seed is None:
        return None
    return training_hparams.data_order_seed + ep
