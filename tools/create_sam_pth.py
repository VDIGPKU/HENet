import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--path', help='the path to source pth file')
    parser.add_argument('--output', help='the path to where results are saved')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    if args.path is not None:
        pth_dict = torch.load(args.path)
        key_list = list(pth_dict)
        for key in key_list:
            if key[:13] != 'image_encoder': # 丢弃
                del pth_dict[key]
            else: # 删除'image_encoder.' 注意是14个！
                pth_dict[key[14:]] = pth_dict[key]
                del pth_dict[key]
    else:
        raise ValueError('need --path to provide a path to source pth file')

    if args.output is not None:
        torch.save(pth_dict, args.output)
    else:
        raise ValueError('need --output to provide a path to where results are saved')


if __name__ == '__main__':
    main()