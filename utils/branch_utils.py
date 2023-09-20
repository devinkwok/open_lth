import models.registry


def load_dense_model(branch, state_step, level_root):
    # check current level first
    try:
        return models.registry.load(level_root, state_step, branch.lottery_desc.model_hparams, outputs=branch.lottery_desc.train_outputs)
    except:  # if checkpoint isn't there, check level_pretrain
        pretrain_root = branch.lottery_desc.run_path(branch.replicate, "pretrain")
        return models.registry.load(pretrain_root, state_step, branch.lottery_desc.model_hparams, outputs=branch.lottery_desc.train_outputs)


def get_output_layers(branch):
    # Get names of output layers using dummy models with different outputs
    dummy_1 = models.registry.get(branch.lottery_desc.model_hparams, outputs=branch.lottery_desc.train_outputs).state_dict()
    dummy_2 = models.registry.get(branch.lottery_desc.model_hparams, outputs=branch.lottery_desc.train_outputs + 1).state_dict()
    return [k for k in dummy_1.keys() if dummy_1[k].shape != dummy_2[k].shape]


def reinitialize_output_layers(branch, model, num_classes):
    output_layers = get_output_layers(branch)
    # Get a new initialization with num_classes and replace the output layers
    state_dict = model.state_dict()
    model = models.registry.get(branch.lottery_desc.model_hparams, outputs=num_classes)
    for k in output_layers:
        state_dict[k] = model.state_dict()[k]
    model.load_state_dict(state_dict)
    return model, output_layers
