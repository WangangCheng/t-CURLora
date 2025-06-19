import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
from models.UNETR import UNETR
#from models.segmamba import SegMamba
from data.BraTS import BraTS
from predict import validate_softmax




os.environ['CUDA_VISIBLE_DEVICES']  = '0'
parser = argparse.ArgumentParser()

parser.add_argument('--user', default='Wangangcheng', type=str)

parser.add_argument('--root', default='/home/setdata/ubuntu2204/cwg/t-CURLora/Datasets/EADC-LPBA40', type=str)

parser.add_argument('--valid_dir', default='/home/setdata/ubuntu2204/cwg/t-CURLora/Datasets/EADC-LPBA40', type=str)

parser.add_argument('--valid_file', default='test.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='UNETR', type=str)

parser.add_argument('--test_date', default='2025-06-05', type=str)

parser.add_argument('--test_file', default='model_epoch_999.pth', type=str)

parser.add_argument('--use_TTA', default=False, type=bool)

parser.add_argument('--post_process', default=False, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='UNETR', type=str)

parser.add_argument('--num_cls', default=2, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=2, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #model = SegMamba(in_chans=1,out_chans=2)
    model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="conv",  # 使用proj_type参数
    norm_name="instance",
    res_block=True,
    dropout_rate=0.1
)


    #model = SwinUNETR(img_size = 128,in_channels = 1, out_channels = 2,feature_size = 48)
    #model = AttentionUNet_4pools()
    #model = DSNet()
    #model = UNet(1,[32,48,64,96,128],2,net_mode='3d')
    #model = UNet_4pools()
    #_,model = TransBTS(dataset = 'brats',_conv_repr=True,_pe_type="learned")
    #model = Effcient_AttenUnet(dataset='brats')

    model = model.cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment+args.test_date, args.test_file)
    print(load_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel= False,
                         savepath=submission,
                         verbose=True,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=False,
                         postprocess=False
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


