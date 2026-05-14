import argparse
import random
import os
from pathlib import Path

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

try:
    import setproctitle
except ImportError:
    setproctitle = None


BASELINE_ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value):
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((BASELINE_ROOT / path).resolve())


def _prepare_protocol_output_paths(args):
    if not args.output_dir:
        return args
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints = str(output_dir / "checkpoints")
    args.results = str(output_dir / "results")
    args.test_results = str(output_dir / "test_results")
    args.log_path = str(output_dir / "result_long_term_forecast.txt")
    for path_value in [args.checkpoints, args.results, args.test_results]:
        Path(path_value).mkdir(parents=True, exist_ok=True)
    return args


def _resolve_runtime_paths(args):
    args.root_path = _resolve_path(args.root_path)
    for field in [
        "q_mat_file",
        "Q_MAT_file",
        "q_out_mat_file",
        "Q_OUT_MAT_file",
        "r_mat_file",
        "R_MAT_file",
        "rk_mat_file",
        "Rk_MAT_file",
    ]:
        setattr(args, field, _resolve_path(getattr(args, field)))
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--fix_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--seed', type=int, default=None, help='protocol alias for fix_seed')

    # save
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./exp_results/', help='location of results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--log_path', type=str, default='./result_long_term_forecast.txt', help='log path')
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)
    parser.add_argument('--run_id', type=str, default='', help='protocol run id')
    parser.add_argument('--output_dir', type=str, default='', help='protocol output directory')
    parser.add_argument('--checkpoint_path', type=str, default='', help='explicit checkpoint path for evaluation')
    parser.add_argument('--metric_policy', type=str, default='forecasting_v1', help='protocol metric policy')
    parser.add_argument('--selection_policy', type=str, default='best_val_mse', help='protocol selection policy')
    parser.add_argument('--skip_predictions', action='store_true', default=False, help='skip protocol prediction npz')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--add_noise', action='store_true', help='add noise')
    parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')
    parser.add_argument('--test_corruption_type', type=str, default='none',
                        choices=['none', 'spike', 'segment'], help='test-time input corruption type')
    parser.add_argument('--test_corruption_rate', type=float, default=0.05, help='test-time corruption rate')
    parser.add_argument('--test_corruption_amp', type=float, default=3.0, help='test-time corruption amplitude')
    parser.add_argument('--test_corruption_seed', type=int, default=2023, help='test-time corruption seed')
    parser.add_argument('--test_corruption_segment_len', type=int, default=4, help='test-time segment length')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--reconstruction_type', type=str, default="imputation", help='type of reconstruction')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # FreDF
    parser.add_argument('--rec_lambda', type=float, default=1, help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=0, help='weight of auxilary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')

    # IterNorm
    parser.add_argument('--iter_norm', type=int, default=1, help='use iter norm')
    parser.add_argument('--num_groups', type=int, default=8, help='number of groups')
    parser.add_argument('--T', type=int, default=5, help='iteration times')
    parser.add_argument('--momentum', type=float, default=1, help='momentum')
    parser.add_argument('--eps', type=float, default=1e-5, help='epsilon')
    parser.add_argument('--affine', type=int, default=0, help='affine')
    parser.add_argument('--revin', type=int, default=1, help='use revin')
    parser.add_argument('--add_module', type=str, default='embed', help='new module add to Linear')
    parser.add_argument('--individual', type=int, default=0, help='individual')
    parser.add_argument('--save_cov', type=int, default=0, help='save covariance matrix')
    parser.add_argument('--norm_type', type=str, default='mlp', help='norm type, options: [mlp, ind]')

    # LinearEnc
    parser.add_argument('--concat', type=int, default=0, help='concat')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--Q_mode', type=str, default='ItrNorm', help='Q_mode, options: [ItrNorm, lw]')

    # orthoformer
    parser.add_argument('--q_mat_file', type=str, default=None, help='q_mat_file npy file')
    parser.add_argument('--c_mat_file', type=str, default=None, help='c_mat_file npy file')
    parser.add_argument('--q_channel_file', type=str, default=None, help='q_channel_file npy file')
    parser.add_argument('--Q_MAT_file', type=str, default=None, help='Q_MAT_file npy file')
    parser.add_argument('--q_out_mat_file', type=str, default=None, help='q_out_mat_file npy file')
    parser.add_argument('--c_out_mat_file', type=str, default=None, help='c_out_mat_file npy file')
    parser.add_argument('--Q_OUT_MAT_file', type=str, default=None, help='Q_OUT_MAT_file npy file')
    parser.add_argument('--Q_chan_indep', type=int, default=0, help='Q_channel_independence')
    parser.add_argument('--Q_loss', type=int, default=0, help='use Q_mat in loss function')
    parser.add_argument('--FFT_loss', type=int, default=0, help='use FFT_loss in loss function')
    parser.add_argument('--Q_loss_alpha', type=float, default=0.5, help='Q_loss_alpha')
    parser.add_argument('--dim_reduce_ratio', type=float, default=1.0, help='dim_reduce_ratio')

    parser.add_argument('--CKA_flag', type=int, default=0, help='CKA_flag')
    parser.add_argument('--embed_size', type=int, default=8, help='embed_size')
    parser.add_argument('--loss_mode', type=str, default='L1', help='loss_mode',
                        choices=['L1', 'L2', 'L1L2', 'MAPE', 'MASE', 'SMAPE'])
    
    # PCCA
    parser.add_argument('--k_dim', type=int, default=96, help='k_dim')
    parser.add_argument('--enable_chan_align', type=int, default=0, help='enable_chan_align')
    parser.add_argument('--chan_align_type', type=str, default='layernorm', help='chan_align_type',
                        choices=['layernorm', 'linear'])
    parser.add_argument('--gate_module', type=str, default='CAR', help='gate_module',
                        choices=['CAR', 'GFM', 'PLA', 'None'])
    parser.add_argument('--pla_film_hidden', type=int, default=None, help='pla_film_hidden')
    parser.add_argument('--car_reduction', type=int, default=4, help='car_reduction')
    parser.add_argument('--pla_eps', type=float, default=0.2, help='pla_eps')
    parser.add_argument('--pla_dropout', type=float, default=0.0, help='pla_dropout')


    #RRR
    parser.add_argument('--r_mat_file', type=str, default=None, help='r_mat_file npy file')
    parser.add_argument('--R_MAT_file', type=str, default=None, help='R_MAT_file npy file')
    parser.add_argument('--freeze_R', type=int, default=0, help='freeze_R')
    parser.add_argument('--r_lora_rank', type=int, default=0, help='r_lora_rank')
    parser.add_argument('--r_lora_scale', type=float, default=1.0, help='r_lora_scale')
    parser.add_argument('--adapter_context', type=str, default='stats2', help='adapter_context',
                        choices=['stats2', 'stats8', 'conv'])
    parser.add_argument('--adapter_C', type=int, default=8, help='adapter_C')
    parser.add_argument('--q_adapter_hidden', type=int, default=64, help='q_adapter_hidden')
    parser.add_argument('--use_head_film', type=bool, default=False, help='use_head_film')
    parser.add_argument('--mix_hid_d', type=int, default=0, help='mix_hid_d')
    parser.add_argument('--mix_hid_r', type=int, default=0, help='mix_hid_r')
    parser.add_argument('--r_rank', type=int, default=0, help='r_rank')
    parser.add_argument('--k_top', type=int, default=16, help='k_top')
    parser.add_argument('--alpha_init', type=float, default=0.5, help='alpha_init')
    parser.add_argument('--use_norm_D', type=int, default=1, help='use_norm_D')
    parser.add_argument('--init_fc_avg', type=int, default=1, help='init_fc_avg')
    parser.add_argument('--rk_mat_file', type=str, default=None, help='rk_mat_file npy file')
    parser.add_argument('--Rk_MAT_file', type=str, default=None, help='Rk_MAT_file npy file')
    parser.add_argument('--mask_threshold', type=float, default=0.25, help='threshold for Mahalanobis mask')
    parser.add_argument('--mask_sharpness_k', type=float, default=100.0, help='sharpness k for Mahalanobis mask')
    

    # Patching
    parser.add_argument('--temp_stride', type=int, default=8, help='temp_stride for temporal patching')
    parser.add_argument('--temp_patch_len', type=int, default=16, help='temp_patch_len for patching')
    parser.add_argument('--temp_patch_len2', type=int, default=16, help='temp_patch_len2')
    parser.add_argument('--temp_stride2', type=int, default=8, help='temp_stride2')
    parser.add_argument('--Patch_CI', type=int, default=1, help='use channel independence or not')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    if args.seed is not None:
        args.fix_seed = args.seed
    if args.model == 'PDT':
        # PDT is the new-architecture name for the old R2Linear implementation.
        args.model = 'PDT'
    args = _prepare_protocol_output_paths(args)
    args = _resolve_runtime_paths(args)
    args.checkpoint_path = _resolve_path(args.checkpoint_path)

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name != 'long_term_forecast':
        raise ValueError('New-architecture PDT entry only supports long_term_forecast.')
    Exp = Exp_Long_Term_Forecast

    if setproctitle is not None:
        setproctitle.setproctitle(args.task_name)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_rr{}_frR{}_k{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.r_rank,
                args.freeze_R,
                args.k_top,
                args.des,
                ii
            )

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_rr{}_frR{}_k{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.r_rank,
            args.freeze_R,
            args.k_top,
            args.des,
            ii
        )

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    print("实际生效的 CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""), "\n")
