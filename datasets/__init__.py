from . import cardiacspect_dataset_reg

def get_datasets(opts):
    if opts.dataset == 'CardiacSPECT_Reg':
        trainset = cardiacspect_dataset_reg.CardiacSPECT_Train_Reg(opts)
        validset = cardiacspect_dataset_reg.CardiacSPECT_Valid_Reg(opts)
        testset = cardiacspect_dataset_reg.CardiacSPECT_Test_Reg(opts)

    elif opts.dataset == 'XXX':
        a = 1
        # trainset = sv_dataset.SVTrain(opts.data_root)
        # valset = sv_dataset.SVTest(opts.data_root)

    return trainset, validset, testset
