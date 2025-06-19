import argparse
import os
import random
import logging
import numpy as np
import time
from torch.fft import fft,ifft
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.UNETR import UNETR
import torch.distributed as dist
from models import criterionsWT
from models.criterions import*
from data.BraTS import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']  = '0'

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='Wangangcheng', type=str)
parser.add_argument('--experiment', default='UNETR', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',
                    default='UNETR,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./Datasets/EADC', type=str)
parser.add_argument('--train_dir', default='./Datasets/EADC', type=str)
parser.add_argument('--val_dir', default='./t-CURLora/Datasets/EADC', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--val_file', default='valid.txt', type=str)
parser.add_argument('--dataset', default='hippo', type=str)
parser.add_argument('--model_name', default='UNETR', type=str)

# Training Information
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=2e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='softmax_dice2', type=str)
parser.add_argument('--num_cls', default=1, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=1000, type=int)
parser.add_argument('--val_epoch', default=100, type=int)
parser.add_argument('--save_freq', default=500, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)
args = parser.parse_args()

class Finetune_UNETR:
    def __init__(self, resume_path, in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, proj_type, norm_name, res_block, dropout_rate, r):
        self.resume_path = resume_path
        self.r = r
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate
        )
        self.load_model()
        self.U_parameters = []

    def load_model(self):
        checkpoint = torch.load(self.resume_path, map_location=lambda storage, loc: storage, weights_only=False)

        #checkpoint = torch.load(self.resume_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def compute_new_weights(self):
        weights_linear1 = []
        weights_linear2 = []
        weights_qkv = []
        
        for name, param in self.model.named_parameters():
            if "linear1" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                reshaped_param = param.data.view(1, 3072, 768)
                weights_linear1.append(reshaped_param)
        
        if weights_linear1:
            weight_tensor_linear1 = torch.cat(weights_linear1, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
        
        for name, param in self.model.named_parameters():
            if "linear2" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                reshaped_param = param.data.view(1, 768, 3072)
                weights_linear2.append(reshaped_param)
        
        if weights_linear2:
            weight_tensor_linear2 = torch.cat(weights_linear2, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
        
        for name, param in self.model.named_parameters():
            if "qkv" in name and len(param.shape) == 2 and param.numel() == 2304 * 768:
                reshaped_param = param.data.view(3, 768, 768)  
                weights_qkv.append(reshaped_param)
            elif "attn.out_proj.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 768:
                reshaped_param = param.data.view(1, 768, 768)
                weights_qkv.append(reshaped_param)
        
        if weights_qkv:
            weight_tensor_qkv = torch.cat(weights_qkv, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
        
        # Perform Fourier Transform
        D1 = fft(weight_tensor_linear1, dim=0) 
        D2 = fft(weight_tensor_qkv, dim=0)
        D3 = fft(weight_tensor_linear2, dim=0)
        
        def cur_decomposition(tensor, r):
            num_slices = tensor.shape[0]
            # Step 1: Calculate column norms across all slices
            col_norms_sum = torch.zeros(tensor.shape[2])
            for k in range(num_slices):
                col_norms_sum += torch.norm(tensor[k], p=2, dim=0)
            
            # Step 2: Select top-r columns based on the summed norms
            col_indices = torch.topk(col_norms_sum, r, largest=True).indices
            
            # Step 3: Select columns based on col_indices to create a new tensor
            selected_tensors = []
            for k in range(num_slices):
                slice_matrix = tensor[k]
                selected_matrix = slice_matrix[:, col_indices]
                selected_tensors.append(selected_matrix)
            selected_tensor = torch.stack(selected_tensors, dim=0)
            
            # Step 4: Calculate row norms across all slices for reduced tensor
            row_norms_sum = torch.zeros(selected_tensor.shape[1])
            for k in range(num_slices):
                row_norms_sum += torch.norm(selected_tensor[k], p=2, dim=1)
            
            # Step 5: Select top-r rows based on the summed norms
            row_indices = torch.topk(row_norms_sum, r, largest=True).indices
            
            # Step 6: Perform CUR decomposition using the selected columns and rows
            I_indices = col_indices
            J_indices = row_indices
            C = []
            R = []
            U = []
            U_f = []
            
            for k in range(num_slices):
                slice_matrix = tensor[k]
                
                # Select columns based on I_indices
                C.append(slice_matrix[:, I_indices])
                
                # Select rows based on J_indices
                R.append(slice_matrix[J_indices, :])
                
                # Create U matrix based on selected rows and columns
                #U_matrix = slice_matrix[J_indices,I_indices]
                U_matrix = slice_matrix[J_indices][:,I_indices]
                
                # Compute the pseudo-inverse of U
                U_matrix = torch.linalg.pinv(U_matrix)
                U.append(U_matrix)
                
                U_frozen = slice_matrix[J_indices][:,I_indices]
                U_frozen = torch.linalg.pinv(U_frozen)
                U_f.append(U_frozen)
                # Register U as a learnable parameter
                U_param = nn.Parameter(U_matrix)
                self.U_parameters.append(U_param)
            
            C_tensor = torch.stack(C, dim=0)
            R_tensor = torch.stack(R, dim=0)
            U_tensor = torch.stack(U, dim=0)
            U_f_tensor = torch.stack(U_f, dim=0)
            
            return C_tensor, U_tensor, R_tensor, I_indices, J_indices, U_f_tensor
        
        # CUR decomposition for each tensor
        C1, U1, R1, I1, J1, U11 = cur_decomposition(D1, self.r)
        C2, U2, R2, I2, J2, U22 = cur_decomposition(D2, self.r)
        C3, U3, R3, I3, J3, U33 = cur_decomposition(D3, self.r)
        
        # Perform CUR multiplication slice by slice, followed by an inverse FFT transformation.
        def reconstruct_weight(C, U, R):
            num_slices = C.shape[0]
            reconstructed_slices = []
            for i in range(num_slices):
                
                C_slice = C[i,:, :]  # [num_rows, r]
                U_slice = U[i,:, :]  # [r, r]
                R_slice = R[i,:, :]  # [r, num_cols]
                
                CU = torch.matmul(C_slice, U_slice)  # [num_rows, r]
                CUR = torch.matmul(CU, R_slice)  # [num_rows, num_cols]
                
                reconstructed_slices.append(CUR)
            
            
            reconstructed_tensor = torch.stack(reconstructed_slices, dim=0)
            return reconstructed_tensor
        
        
        W1_reconstructed = reconstruct_weight(C1, U1, R1)
        W1_frozen_reconstructed = reconstruct_weight(C1, U11, R1)
        W2_reconstructed = reconstruct_weight(C2, U2, R2)
        W2_frozen_reconstructed = reconstruct_weight(C2, U22, R2)
        W3_reconstructed = reconstruct_weight(C3, U3, R3)
        W3_frozen_reconstructed = reconstruct_weight(C3, U33, R3)
        
        W1 = ifft(W1_reconstructed, dim=0).real
        W1_frozen = ifft(W1_frozen_reconstructed, dim=0).real
        W2 = ifft(W2_reconstructed, dim=0).real
        W2_frozen = ifft(W2_frozen_reconstructed, dim=0).real
        W3 = ifft(W3_reconstructed, dim=0).real
        W3_frozen = ifft(W3_frozen_reconstructed, dim=0).real

        W11 = weight_tensor_linear1 - W1_frozen + W1
        W22 = weight_tensor_qkv - W2_frozen + W2
        W33 = weight_tensor_linear2 - W3_frozen + W3
        
        return W11, W22 , W33
    
    def update_weights(self, W_linear1, W_qkv, W_linear2):
        idx_linear1 = 0
        idx_linear2 = 0
        idx_qkv = 0
        for name, param in self.model.named_parameters():
            if "linear1" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                param.data = W_linear1[idx_linear1:idx_linear1+1].reshape(param.shape).float()
                idx_linear1 += 1
            elif "linear2" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                param.data = W_linear2[idx_linear2:idx_linear2+1].reshape(param.shape).float()
                idx_linear2 += 1
            elif "qkv" in name and len(param.shape) == 2 and param.numel() == 2304 * 768:
                param.data = W_qkv[idx_qkv:idx_qkv+3].reshape(param.shape).float()
                idx_qkv += 3
            elif "attn.out_proj.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 768:
                param.data = W_qkv[idx_qkv:idx_qkv+1].reshape(param.shape).float()
                idx_qkv += 1
    
    def freeze_parameters(self):
        for name, param in self.model.named_parameters():
            if not any(key in name for key in ["decoder"]):
                param.requires_grad = False
        for param in self.U_parameters:
            param.requires_grad = True

    def finetune(self):
        W_linear1, W_qkv, W_linear2 = self.compute_new_weights()
        if W_linear1 is not None:
            self.update_weights(W_linear1, W_qkv, W_linear2)
            self.freeze_parameters()

            
def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
    log_file = log_dir + '.txt' 
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Example usage
    finetuner = Finetune_UNETR(
        resume_path='./checkpoint/UNETR2024-05-23/model_epoch_last.pth',
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
        r=1
    )
    finetuner.finetune()
    model = finetuner.model
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    criterionWT = getattr(criterionsWT, args.criterion)

 
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter()



    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.val_dir, args.val_file)
    val_root = os.path.join(args.root, args.val_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    val_set = BraTS(val_list, val_root, args.mode)

    
    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for val = {}'.format(len(val_set)))


    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)


    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch): 
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        #train

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(x)
            loss,loss_00,loss_01 = criterionWT(output, target)
            reduce_loss = loss.item()
            reduce_loss_00 = loss_00.item()
            reduce_loss_01 = loss_01.item()
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} |0:{:.4f}|1:{:.4f} |'.format(epoch, i, reduce_loss,reduce_loss_00, reduce_loss_01))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
        torch.cuda.empty_cache()
        end_epoch = time.time()

        #val
        if epoch%args.val_epoch==0:
             logging.info('Samples for val = {}'.format(len(val_set)))
             with torch.no_grad():
                 for i, data in enumerate(val_loader):
                     #adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
                     x, target = data
                     x = x.cuda(non_blocking=True)
                     target = target.cuda(non_blocking=True)
                     output = model(x)
                     loss_01 = Dice(output[:, 1, ...], (target == 1).float())
                     #loss_02 = Dice(output[:, 2, ...], (target == 2).float())
                     #loss_03 = Dice(output[:, 3, ...], (target == 3).float())


                     logging.info('Epoch: {}_Iter:{}  Dice: 1:{:.4f}||'
                         .format(epoch, i,  1-loss_01))
        end_epoch = time.time()  
        

        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                or (epoch + 1) % int(args.end_epoch - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss', reduce_loss, epoch)
            #writer.add_scalar('loss_0', reduce_loss_0, epoch)
            #writer.add_scalar('loss_1', reduce_loss_1, epoch)
            writer.add_scalar('loss_00', reduce_loss_00, epoch)
            writer.add_scalar('loss_01', reduce_loss_01, epoch)
            #writer.add_scalar('loss_02', reduce_loss_02, epoch)
            #writer.add_scalar('loss_03', reduce_loss_03, epoch)
   


        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))


    writer.close()

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')




def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
