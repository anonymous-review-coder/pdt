import os
import warnings
import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Linear_EncDec import Encoder_ori, LinearEncoder, Mahalanobis_mask
# from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention

import sys


"""
ROLinear:
Q_in before + R + no_freeze_R + no delta2
ROLinear (RRR + OLinear) with Top-k Linear Head:
      x --RevIN--> Q_in投影到 r 维 --tokenEmb--> [B,N,r,D]
        ├─ Key 分支：取前 k 个方向 -> Linear(k*D -> r)  (简单线性预测)
        └─ Ortho 分支：orthotrans 处理复杂特征 -> D-池化到 r
      α 融合: z = σ(α)*z_key + (1-σ(α))*z_ortho  -> (可选 LayerNorm(r))
      经 R (r->H) 风格 fc(H*D->H) -> RevIN 反变换
"""

def _load_npy(path, root_path=None, device='cpu'):
    p = path if os.path.isfile(path) else os.path.join(root_path or '', path)
    assert os.path.isfile(p), f'File not found: {path}'
    arr = np.load(p)
    return torch.from_numpy(arr).to(torch.float32).to(device)




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff



        self.k_top = configs.k_top
        # assert 1 <= self.k_top <= self.r

        self.Q_chan_indep = configs.Q_chan_indep

        q_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        # print(q_mat_dir)
        if not os.path.isfile(q_path):
            q_path = os.path.join(configs.root_path, q_path)
        # print(q_mat_dir)
        assert os.path.isfile(q_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        Q_in = _load_npy(q_path, device=device)  # [L,r] 或 [N,L,r]
        if self.Q_chan_indep:
            assert Q_in.ndim == 3 and Q_in.shape[0] == self.enc_in and Q_in.shape[1] == self.seq_len
            self.r = Q_in.shape[2]
        else:
            assert Q_in.ndim == 2 and Q_in.shape[0] == self.seq_len
            self.r = Q_in.shape[1]
        self.register_buffer('Q_in', Q_in)

        # R
        r_path = configs.R_MAT_file if self.Q_chan_indep else configs.r_mat_file
        if not os.path.isfile(r_path):
            r_path = os.path.join(configs.root_path, r_path)
        R = _load_npy(r_path, device=device)      # [r,H] 或 [N,r,H]
        if self.Q_chan_indep:
            assert R.ndim == 3 and R.shape[0] == self.enc_in and R.shape[1] == self.r and R.shape[2] == self.pred_len
        else:
            assert R.ndim == 2 and R.shape[0] == self.r and R.shape[1] == self.pred_len
        
        # R_k
        rk_path = configs.Rk_MAT_file if self.Q_chan_indep else configs.rk_mat_file
        if not os.path.isfile(rk_path):
            rk_path = os.path.join(configs.root_path, rk_path)
        Rk = _load_npy(rk_path, device=device)      # [k,H] 或 [N,k,H]
        if self.Q_chan_indep:
            assert Rk.ndim == 3 and Rk.shape[0] == self.enc_in and Rk.shape[1] == self.k_top and Rk.shape[2] == self.pred_len
        else:
            assert Rk.ndim == 2 and Rk.shape[0] == self.k_top and Rk.shape[1] == self.pred_len
        self.Rk_param = nn.Parameter(Rk.clone())        # [k,H]

        
        self.freeze_R = configs.freeze_R
        if self.freeze_R:
            self.register_buffer('R_fix', R)
        else:
            self.R_param = nn.Parameter(R.clone())
        
        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        # self.mask_generator = Mahalanobis_mask(self.r)
        self.mask_generator = Mahalanobis_mask(self.k_top, threshold=configs.mask_threshold, sharpness_k=configs.mask_sharpness_k)

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        self._last_channel_mask = None

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation, token_num=self.enc_in,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        # self.ortho_trans = nn.Sequential(
        #     nn.Linear(self.r * self.embed_size, self.d_model),
        #     self.encoder,
        #     nn.Linear(self.d_model, self.r * self.embed_size)
        # )
        self.Linear_head = nn.Linear(self.r * self.embed_size, self.d_model)
        self.Linear_tail = nn.Linear(self.d_model, self.r * self.embed_size)

        # learnable delta
        # self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.r))
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.r)) 
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))
        self.delta3 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

        # ------- Key 分支：对每个 D slice 共享的 k→r 线性映射 -------

        
        # 映射矩阵形状 [r, k]（注意 F.linear 的 weight 语义）
        # self.key_map = nn.Sequential(nn.Linear(self.k_top*self.embed_size, self.d_model), nn.Linear(self.d_model, self.r*self.embed_size))
        self.key_map = nn.Linear(self.k_top*self.embed_size, self.d_model)
        # nn.init.xavier_uniform_(self.key_map.weight, gain=0.5)

        self.proj_dmodel = nn.Linear(self.d_model, self.r*self.embed_size)

        # alpha_init = float(getattr(configs, 'alpha_init', 0.5))
        # logit = math.log(alpha_init/(1-alpha_init))
        # self.alpha_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        self.alpha = configs.alpha_init

        # self.norm_D = nn.LayerNorm(self.pred_len)
        # self.norm_D = nn.LayerNorm(self.d_model)
        self.norm_D = nn.LayerNorm(self.embed_size)

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x.transpose(-1, -2)       # [B,N,T]

        if self.Q_chan_indep:
            z_k = torch.einsum('bnt,ntr->bnr', x_ori, self.Q_in) + self.delta1 # [B,N,r]
        else:
            z_k = torch.einsum('bnt,tr->bnr', x_ori, self.Q_in) + self.delta1    # [B,N,r]
        x_ori = z_k.transpose(-1, -2)  # B r N

        channel_mask = self.mask_generator(x_ori.transpose(-1,-2)[:,:,:self.k_top])  # [B, r, N]
        self._last_channel_mask = channel_mask.detach().cpu()
        
        # [B, T, N]
        # embedding x: [B, N, r, D]
        x = self.tokenEmb(x_ori, self.embeddings)

        # x = self.Fre_Trans(x)
        B, N, r, D = x.shape
        assert r == self.r
        # [B, N, D, r]
        x_trans = x.transpose(-1, -2)

        # ########## transformer ####
        # x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.r)    # [B,N,D,r]
        x_trans = self.Linear_head(x_trans.flatten(-2))
        x_trans = self.encoder(x=x_trans, attn_mask=channel_mask)
        x_trans = self.Linear_tail(x_trans).reshape(B, N, D, self.r)    # [B,N,D,r]



        # ------ Key 分支：Top-k -> [B,N,r,D] （对每个 D slice 共享 k→r 线性）------
        x_k = x[:, :, :self.k_top, :]
        x_k_dk = x_k.permute(0,1,3,2)                               # [B,N,D,k]
        # x_k_f = x_k_dk.flatten(-2)                                  # [B,N,D*k]
        # # z_key_rd = self.key_map(x_k_f).reshape(B, N, D, self.r)                             # [B,N,D,r]
        # z_key_rd = self.key_map(x_k_f)    # [B,N,dmodel]
        if self.Q_chan_indep:
            z_key_rd = torch.einsum('bnkd,nkh->bndh', x_k_dk.transpose(-1, -2), self.Rk_param) + self.delta3
        else:
            z_key_rd = torch.einsum('bnkd,kh->bndh', x_k_dk.transpose(-1, -2), self.Rk_param) + self.delta3

        # ------ α 融合（保留 D）------
        # alpha = torch.sigmoid(self.alpha_logit) 
        # alpha = self.alpha
        # z_fused = alpha * z_key_rd + (1.0 - alpha) * x_trans    # [B,N,D,r]   [B,N,dmodel]

        # z_fused = self.norm_D(z_fused.permute(0,1,3,2)).permute(0,1,3,2)
        # z_fused = self.norm_D(z_fused)          # [B,N,dmodel]

        # z_fused = self.proj_dmodel(z_fused).reshape(B, N, D, self.r)      # [B,N,D,r]

        # 暂时先用Identity
        if self.freeze_R:
            if self.Q_chan_indep:
                x = torch.einsum('bnrd,nrh->bndh', x_trans.transpose(-1, -2), self.R_fix)
            else:
                x = torch.einsum('bnrd,rh->bndh', x_trans.transpose(-1, -2), self.R_fix) + self.delta2
                # x = x_trans + self.delta2
                # added on 25/1/30
                # x = F.gelu(x)
        else:
            if self.Q_chan_indep:
                x = torch.einsum('bnrd,nrh->bndh', x_trans.transpose(-1, -2), self.R_param)
            else:
                x = torch.einsum('bnrd,rh->bndh', x_trans.transpose(-1, -2), self.R_param)
                # x = x_trans + self.delta2
                # added on 25/1/30
                # x = F.gelu(x)
        
        alpha = self.alpha
        # alpha = torch.sigmoid(self.alpha_logit) 
        x = alpha * z_key_rd + (1.0 - alpha) * x    # [B,N,D,h]
        # x = self.norm_D(x)

        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        x = self.norm_D(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out

    def save_channel_mask(self, output_dir):
        if self._last_channel_mask is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        mask = self._last_channel_mask.numpy()
        np.save(os.path.join(output_dir, 'channel_mask.npy'), mask)

        values = mask.astype(np.float64)
        stats = {
            'shape': list(values.shape),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'density_gt_0_5': float((values > 0.5).mean()),
        }
        with open(os.path.join(output_dir, 'mask_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, sort_keys=True)
