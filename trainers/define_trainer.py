from .trainer_hifi_basic import TrainerHiFiBasic

all_trainers_dict = {
        'TrainerHiFiBasic': TrainerHiFiBasic,
        }

#TODO: modify if needed
def define_trainer(trainer, data_dict, model_dict, opt_dict, loss_function, cfg):
    """
    Define trainer class according to model configuration
    """
    assert trainer in all_trainers_dict.keys(), NotImplementedError("Not implemented trainer")

    return all_trainers_dict[trainer](data_dict, model_dict, opt_dict, loss_function, cfg)
    


