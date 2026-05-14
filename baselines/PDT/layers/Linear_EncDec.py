import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List
from torch.nn.functional import gumbel_softmax

from utils.CKA import CudaCKA

class STEThreshold(torch.autograd.Function):
    """
    使用直通估计器（STE）实现的确定性阈值函数。
    - 前向传播: 应用一个理想的阶跃函数 (p > threshold)。
    - 反向传播: 将梯度视为1，使其“直通”。
    """
    
    @staticmethod
    def forward(ctx, p, threshold=0.5):
        """
        在前向传播中，我们应用一个硬阈值，得到严格的0/1输出。
        """
        return (p > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们将上游传来的梯度原封不动地传回给p的梯度。
        threshold的梯度为None，因为它不参与学习。
        """
        grad_input = grad_output.clone()
        return grad_input, None

class SigmoidThreshold(nn.Module):
    def __init__(self, threshold=0.5, sharpness_k=10.0):
        """
        使用缩放Sigmoid实现的可微、确定性阈值函数。

        Args:
            threshold (float): 阈值 t。控制激活点的位置。
                               较低的 t -> 更稠密的掩码 (更多的1)。
                               较高的 t -> 更稀疏的掩码 (更少的1)。
            sharpness_k (float): 锐度参数 k。控制函数的陡峭程度。
                                 较高的 k -> 更接近0/1的“硬”掩码。
                                 较低的 k -> 包含更多中间值的“软”掩码。
        """
        super().__init__()
        # 将 t 和 k 定义为非训练参数的缓冲区，便于模块状态管理
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('sharpness_k', torch.tensor(sharpness_k))

    def forward(self, p):
        """
        Args:
            p (torch.Tensor): 输入的概率张量。
        
        Returns:
            torch.Tensor: 经过阈值处理后的张量，值域在(0, 1)之间。
        """
        return torch.sigmoid(self.sharpness_k * (p - self.threshold))

# 输入是原始时序数据 (B, N, L)，输出是掩码 (B, 1, N, N)
class Mahalanobis_mask(nn.Module):
    def __init__(self, seq_len, threshold=0.25, sharpness_k=200.0):
        super(Mahalanobis_mask, self).__init__()

        # self.A = nn.Parameter(torch.randn(seq_len, seq_len), requires_grad=True)
        self.feature_weights = nn.Parameter(torch.ones(seq_len), requires_grad=True)
        self.thresholder = SigmoidThreshold(threshold=threshold, sharpness_k=sharpness_k)

    def calculate_prob_distance(self, X):
        # X: [B, C, L] -> B: batch_size, C: n_vars, L: seq_len
        # XF = torch.abs(torch.fft.rfft(X, dim=-1))
        XF = X  # 直接使用RRR数据
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)

        # B x C x C x D
        diff = X1 - X2
        
        # 构造半正定矩阵 Q = A^T * A
        # Q = self.A.t() @ self.A
        w = F.softplus(self.feature_weights)
        
        # 计算马氏距离
        # temp: B x C x C x D
        # temp = torch.einsum("dk,bxck->bxcd", Q, diff)
        # dist: B x C
        # dist = torch.einsum("bxcd,bxcd->bxc", temp, diff)
        dist = torch.sum(w * (diff ** 2), dim=-1) # 输出维度 [B, C, C]

        # 距离转为相似度
        exp_dist = 1 / (dist + 1e-10)
        
        # 对角线置零
        identity_matrices = 1 - torch.eye(exp_dist.shape[-1], device=exp_dist.device)
        mask = identity_matrices.unsqueeze(0) # unsqueeze for broadcasting
        exp_dist = exp_dist * mask
        
        # 归一化
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / (exp_max + 1e-10)

        # 加上对角线，并乘以一个折扣因子
        identity_matrices = torch.eye(p.shape[-1], device=p.device).unsqueeze(0)
        p = (p * mask + identity_matrices) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        # 使用 Gumbel-Softmax 进行伯努利采样
        log_prob = torch.log(distribution_matrix / (1 - distribution_matrix + 1e-10))
        # gumbel_softmax 需要 (..., num_classes)
        # 我们模拟二分类 [prob_false, prob_true]
        log_probs_paired = torch.stack([torch.log(1 - distribution_matrix + 1e-10), torch.log(distribution_matrix + 1e-10)], dim=-1)
        
        resample_matrix = gumbel_softmax(log_probs_paired, hard=True)

        # 取样为1的概率
        return resample_matrix[..., 1]

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        # bernoulli中两个通道有关系的概率
        # sample = self.bernoulli_gumbel_rsample(p)
        sample = self.thresholder(p)
        
        # 添加 head 维度以匹配 attention mask 格式
        mask = sample.unsqueeze(1)
        return mask     # [B, 1, N, N]

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Encoder_ori(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, one_output=False, CKA_flag=False):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag
        if self.CKA_flag:
            print('CKA is enabled...')

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, nvars, D]
        attns = []
        X0 = None  # to make Pycharm happy
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0:
                    X0 = x

                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    CudaCKA1 = CudaCKA(device=x.device)
                    cka_value = CudaCKA1.linear_CKA(X0.flatten(0, 1)[:1000], x.flatten(0, 1)[:1000])
                    print(f'CKA: \t{cka_value:.3f}')

        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns


class LinearEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None, **kwargs):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        # self.bias = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def _compute_attention_base(self):
        weight = F.softplus(self.weight_mat)

        return weight

    def get_attention_matrix(self):
        A = self._compute_attention_base()
        return F.normalize(A, p=1, dim=-1).detach()

    def forward(self, x, attn_mask=None, **kwargs):
        # x.shape: b, n, d_model
        B, N, D = x.shape
        values = self.v_proj(x)

        # --- MODIFICATION START ---
        # 步骤 1: 计算基础权重矩阵
        A_base = self._compute_attention_base()  # [1, L, L]

        if attn_mask is not None:
            # attn_mask 期望的维度是 [B, L, L] 或可广播的 [B, 1, L, L]
            # Mahalanobis_mask 输出是 [B, 1, L, L]，需要 squeeze
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.squeeze(1)  # -> [B, L, L]
            
            # 使用掩码对基础权重进行稀疏化
            A_masked = A_base * attn_mask  # [1, L, L] * [B, L, L] -> [B, L, L]
        else:
            # 如果没有提供掩码，则退化为原始行为
            A_masked = A_base

        A = F.normalize(A_masked, p=1, dim=-1)
        A = self.dropout(A)

        new_x = A @ values  # + self.bias

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)
        # x = x

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None
    
