import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='ACDNet Sound Classification');

    # General settings
    parser.add_argument('--netType', default='ACDNet',  required=False);
    parser.add_argument('--data', default='{}/datasets/'.format(os.getcwd()),  required=False);
    parser.add_argument('--dataset', required=False, default='esc11', choices=['esc10','esc10_large','esc11', 'esc11_large', 'esc50', 'frog']);
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning');
    parser.add_argument('--strongAugment', default=True,  action='store_true', help='Add scale and gain augmentation');

    opt = parser.parse_args();

    #Leqarning settings
    opt.batchSize = 64;
    opt.weightDecay = 5e-4;
    opt.momentum = 0.9;
    opt.nEpochs = 250;
    opt.LR = 0.1;
    opt.schedule = [0.3, 0.6, 0.9];
    opt.warmup = 10;
    opt.kdLossType = 1;
    opt.start_at = 1
    opt.stop_after = 2
    opt.teacher_path = "/home/ezeq/ACDnet/acdNet_full_f2_esc11_large.pt"

    #Basic Net Settings
    opt.nClasses = 11;
    opt.nFolds = 5;
    opt.splits = [i for i in range(1, opt.nFolds + 1)];
    opt.sr = 20000;
    opt.inputLength = 30225;

    #Test data
    opt.nCrops = 10; #10

    return opt


def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.netType));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.LR));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| Splits: {}'.format(opt.splits));
    print('+------------------------------+');
