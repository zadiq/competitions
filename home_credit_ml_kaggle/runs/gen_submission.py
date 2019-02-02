from credit.models import BoardModel, TrainConfig, TEST_MODE


if __name__ == '__main__':

    ckpt_model = 'weights.05-0.37-0.79.hdf5'
    tc = TrainConfig()
    tc.board_model['model_folder'] = '/media/zadiq/ZHD/datasets/home_credit/curated/v0/board_models/0_07-08_23-45-27'
    tc.board_model['use_multi_gpu'] = False
    model = BoardModel(TEST_MODE, tc, ckpt_model)
    model.save_predicted()
    print('Finished Predictions')
