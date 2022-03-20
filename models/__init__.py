from models import reg_model


def create_model(opts):
    if opts.model_type == 'model_cnn':
        # model = cnn_model.CNNModel(opts)
        raise NotImplementedError

    elif opts.model_type == 'model_reg':
        model = reg_model.REGModel(opts)

    else:
        raise NotImplementedError

    return model
