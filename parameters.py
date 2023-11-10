import argparse

def parameter_reading():
    parser = argparse.ArgumentParser(description="Parameters to train the ICL-Equalizer")
    parser.add_argument('--embedding_dim', type=int, default=64, help='Input embedding dim')
    parser.add_argument('--embedding_dim_single', type=int, default=64,help='Input embedding dim of single layer attention')
    parser.add_argument('--num_head', type=int, default=4, help='Num heads of self attention')
    parser.add_argument('--num_layer', type=int, default=2, help='Transformer layers')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--prompt_seq_length', type=int, default=20, help='length of prompt sequence')
    parser.add_argument('--data_dim', type=int, default=4, help='Data dim of variable x')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Transformer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of Transformer')
    parser.add_argument('--training_steps', type=int, default=3000, help='Training Steps')
    parser.add_argument('--test_data_size', type=int, default=1000, help='test_data_size')
    parser.add_argument('--num_ant', type=int, default=2, help='the number of antenna')
    parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--SNR_dB_min', type=int, default=10, help='the minimum value of the SNR [dB]')
    parser.add_argument('--SNR_dB_max', type=int, default=10, help='the maximum value of the SNR [dB]')
    parser.add_argument('--bits', type=int, default=4, help='number of bits for quantization')
    parser.add_argument('--model_type',  default='GPT2', help='the model type')
    args = parser.parse_args()
    return args