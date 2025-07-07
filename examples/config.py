import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Distributed Deep Learning Training")
    parser.add_argument('-n', '--nodes', default=2, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Batch size to use (default is 64)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-cm', '--comm', default="gloo", type=str, help='Distributed learning communication backend')
    parser.add_argument('--model', default="vgg19", type=str, 
                        choices=["vgg19", "bert", "bart", "roberta", "gpt2"],
                        help="Model to train")

    # OptiReduce specific arguments    
    parser.add_argument("--algo", default="Ring", type=str)
    parser.add_argument("--tr_timeout", default=0, type=int)
    parser.add_argument("--tr_threads_offset", default=0, type=int)
    parser.add_argument("--tr_config", default="dpdk.cfg", type=str)
    parser.add_argument('-hd', '--hadamard', default=False, type=bool, help='Use hadamard transform? (default is False)')
    parser.add_argument("--dev", required=True, help='NIC device to use')

    # UHC specific arguments
    parser.add_argument("--uhc", action="store_true", help="Use UHC for compression")
    parser.add_argument("--quantization_levels", type=int, default=15)
    parser.add_argument("--partial_rotation_times", type=int, default=0)
    parser.add_argument("--is_chunk_based_quant", type=str, default="True")
    
    return parser