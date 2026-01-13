"""
Modified DMD - Attentive & Curriculum Dropout Ready
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加路径以正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

try:
    from ..mome_router import MoMERouter, MoMELosses, AttentionPooling
except ImportError:
    print("Warning: Cannot import mome_router.")

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads

        # Dataset specifics
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1

        # ==================== MoME (Attentive & Robust) ====================
        self.mome_enabled = getattr(args, 'use_mome', True)
        if self.mome_enabled:
            # 3模态输入
            router_input_dim = self.d_l * 2 + self.d_v * 2 + self.d_a * 2

            # 注意力池化层 (替换 mean pooling)
            self.pooler_l = AttentionPooling(self.d_l * 2)
            self.pooler_v = AttentionPooling(self.d_v * 2)
            self.pooler_a = AttentionPooling(self.d_a * 2)

            self.mome_router = MoMERouter(
                input_dim=router_input_dim,
                hidden_dims=[512, 256],
                num_modalities=3,
                temperature=1.0,
                dropout=0.2
            )

            self.mome_losses = MoMELosses(alpha=getattr(args, 'mome_alpha', 0.1))

            # 辅助预测头
            self.visual_pred_head = nn.Sequential(
                nn.Linear(self.d_v * 2, self.d_v), nn.ReLU(), nn.Linear(self.d_v, output_dim)
            )
            self.language_pred_head = nn.Sequential(
                nn.Linear(self.d_l * 2, self.d_l), nn.ReLU(), nn.Linear(self.d_l, output_dim)
            )
            self.audio_pred_head = nn.Sequential(
                nn.Linear(self.d_a * 2, self.d_a), nn.ReLU(), nn.Linear(self.d_a, output_dim)
            )

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # Encoders
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # Decoders
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # Alignment
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # Attention
        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        # Final Projection
        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # Crossmodal
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # GD Layers
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # Ensemble Projection
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def compute_mome_weights(self, feats_list):
        # 使用 Attention Pooling 而不是 Mean Pooling
        # feats_list: [v, a, l]
        processed_feats = []

        # Visual
        v_feat = feats_list[0]
        processed_feats.append(self.pooler_v(v_feat))

        # Audio
        a_feat = feats_list[1]
        processed_feats.append(self.pooler_a(a_feat))

        # Language
        l_feat = feats_list[2]
        processed_feats.append(self.pooler_l(l_feat))

        weights = self.mome_router(processed_feats)
        return weights

    def compute_auxiliary_predictions(self, feats_list, weights):
        # 同样使用 Attention Pooling 确保一致性
        v_feat = self.pooler_v(feats_list[0])
        a_feat = self.pooler_a(feats_list[1])
        l_feat = self.pooler_l(feats_list[2])

        v_pred = self.visual_pred_head(v_feat)
        a_pred = self.audio_pred_head(a_feat)
        l_pred = self.language_pred_head(l_feat)

        w_v, w_a, w_l = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        weighted_uni_pred = w_v * v_pred + w_a * a_pred + w_l * l_pred

        return {
            'visual_pred': v_pred,
            'audio_pred': a_pred,
            'language_pred': l_pred,
            'weighted_uni_pred': weighted_uni_pred
        }

    def forward(self, text, audio, video, is_distill=False, mome_active=False, modality_dropout_prob=0.0):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)
        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]
        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))
        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)
        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)
        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)
        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)
        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)
        proj_s_l = self.proj_cosine_l(s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple: c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple: c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple: c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
        c_proj = self.proj2_c(F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout, training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple: h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]
        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple: h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]
        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple: h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]
        hs_proj_l_high = self.proj2_l_high(F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)
        hs_proj_v_high = self.proj2_v_high(F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)
        hs_proj_a_high = self.proj2_a_high(F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        # ==================== MoME Gating Logic ====================
        mome_weights = None
        mome_aux_preds = None

        if self.mome_enabled and mome_active:
            # === Curriculum Modality Dropout ===
            # 仅在训练阶段应用
            if self.training and modality_dropout_prob > 0:
                # 随机生成概率判断是否丢弃
                # 假设文本永远保留，只丢弃 Vision 或 Audio
                rand_val = torch.rand(1).item()
                if rand_val < modality_dropout_prob:
                    # Drop Visual
                    last_h_v = torch.zeros_like(last_h_v)
                elif rand_val < 2 * modality_dropout_prob:
                    # Drop Audio
                    last_h_a = torch.zeros_like(last_h_a)

            # 2. 计算权重
            feats_list = [last_h_v, last_h_a, last_h_l]
            mome_weights = self.compute_mome_weights(feats_list)

            # 3. 动态门控
            w_v, w_a, w_l = mome_weights[:, 0:1], mome_weights[:, 1:2], mome_weights[:, 2:3]

            # 使用更平滑的门控，避免剧烈波动
            # weight * feat (scale) + feat (residual)
            last_h_v = last_h_v * (1 + w_v)
            last_h_a = last_h_a * (1 + w_a)
            last_h_l = last_h_l * (1 + w_l)

            # 4. 辅助预测
            if is_distill:
                mome_aux_preds = self.compute_auxiliary_predictions(feats_list, mome_weights)

        # 后续DMD流程
        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        res = {
            'logits_l_homo': logits_l_low, 'logits_v_homo': logits_v_low, 'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low, 'repr_v_homo': repr_v_low, 'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l, 'origin_v': proj_x_v, 'origin_a': proj_x_a,
            's_l': s_l, 's_v': s_v, 's_a': s_a,
            'proj_s_l': proj_s_l, 'proj_s_v': proj_s_v, 'proj_s_a': proj_s_a,
            'c_l': c_l, 'c_v': c_v, 'c_a': c_a,
            's_l_r': s_l_r, 's_v_r': s_v_r, 's_a_r': s_a_r,
            'recon_l': recon_l, 'recon_v': recon_v, 'recon_a': recon_a,
            'c_l_sim': c_l_sim, 'c_v_sim': c_v_sim, 'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high, 'logits_v_hetero': logits_v_high, 'logits_a_hetero': logits_a_high,
            'repr_l_hetero': hs_proj_l_high, 'repr_v_hetero': hs_proj_v_high, 'repr_a_hetero': hs_proj_a_high,
            'last_h_l': h_ls[-1], 'last_h_v': h_vs[-1], 'last_h_a': h_as[-1],
            'logits_c': logits_c, 'output_logit': output
        }

        if self.mome_enabled and mome_active and mome_weights is not None:
            res['mome_weights'] = mome_weights
            if mome_aux_preds is not None:
                res.update(mome_aux_preds)

        return res