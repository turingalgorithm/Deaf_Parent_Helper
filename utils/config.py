import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Audio model')
    parser.add_argument('--save_dir', type=str, default="./save/")
    parser.add_argument('--gamma', type=float, default=0.1, help='Value of learning rate decay')
    parser.add_argument('--save_location', type=str, default='./')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument("--sampling_rate", default=5120, type=int)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--load_model', type=bool,default=False)
    parser.add_argument('--class_number', type=int, default=4)

    parser.add_argument('--n_spks', type=int, default=5)
    parser.add_argument('--indim', type=int, default=390)
    parser.add_argument('--outdim', type=int, default=5)
    # 16, 0.00001 할때 91 나옴 결과.
    parser.add_argument('--summary_location', type=str, default='./summary')
    parser.add_argument('--batch_size', type=int, default=12
                        )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='How num using process')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Number of epochs to wait before reducing lr')

    return parser.parse_args()
