import argparse
def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='Mfbp_Baseline_Model',
                        help='model name.')

    parser.add_argument('--input_dir', type=str, default='D:\data\VQA_data\VQAv2',
                        help='input directory for visual question answering.')


    parser.add_argument('--log_dir', type=str, default='./MUTAN_Baseline_logs_epoch_20',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./MUTAN_Baseline_model_epoch_20',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=20,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--num_ans', type=int, default=1000)

    parser.add_argument('--embed_size', type=int, default=2400,
                        help='embedding size of feature vector \
                              for both image and question.')

    # parser.add_argument('--word_embed_size', type=int, default=300,
    #                     help='embedding size of word \
    #                           used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=100,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='load saved model from which epoch')

    parser.add_argument('--saved_model', type=str, default='best_model.pt')
    parser.add_argument('--dim_v', type=int, default='2048')
    parser.add_argument('--dim_q', type=int, default='2400')
    parser.add_argument('--dim_hv', type=int, default='620')
    parser.add_argument('--dim_hq', type=int, default='310')
    parser.add_argument('--dim_mm', type=int, default='510')
    parser.add_argument('--R', type=int, default='10')#多头注意力
    parser.add_argument('--dropout_v', type=float, default='0.5')
    parser.add_argument('--dropout_q', type=float, default='0.5')
    parser.add_argument('--activation_v', type=str, default='tanh')
    parser.add_argument('--activation_q', type=str, default='tanh')
    parser.add_argument('--dropout_hv', type=int, default='0')
    parser.add_argument('--dropout_hq', type=int, default='0')
    parser.add_argument('--att_nb_glimpses',type=int,default='2',help='Number of attentional glimpses')
    parser.add_argument('--att_dim_hv',type=int,default='310')
    parser.add_argument('--att_dim_hq', type=int, default='310')
    parser.add_argument('--att_dim_mm', type=int, default='510')
    parser.add_argument('--att_R', type=int, default='5')
    parser.add_argument('--att_dropout_v',type=float,default='0.5')
    parser.add_argument('--att_dropout_q', type=float, default='0.5')
    parser.add_argument('--att_dropout_mm', type=float, default='0.5')
    parser.add_argument('--att_activation_v', type=str, default='tanh')
    parser.add_argument('--att_activation_q', type=str, default='tanh')
    parser.add_argument('--att_dropout_hv', type=int, default='0')
    parser.add_argument('--att_dropout_hq', type=int, default='0')
    parser.add_argument('--classif_activation', type=str, default='tanh')
    parser.add_argument('--classif_dropout', type=float, default='0.5')



    args = parser.parse_args()

    return args