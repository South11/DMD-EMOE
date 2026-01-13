import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss

logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model[0].parameters()) + list(model[1].parameters()) + list(model[2].parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = [model[0], model[1], model[2]]
        model = net

        # ==================== Schedule Config ====================
        start_temp, end_temp = 1.0, 0.5
        total_epochs = self.args.epochs if hasattr(self.args, 'epochs') else 50
        mome_warmup_epochs = 5

        # Curriculum Dropout 设置
        dropout_start_epoch = 10
        max_dropout_prob = 0.15

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            # 1. Warmup & Temperature
            is_mome_active = (epochs > mome_warmup_epochs)
            if is_mome_active:
                eff_epoch = epochs - mome_warmup_epochs
                eff_total = total_epochs - mome_warmup_epochs
                current_temp = start_temp - (start_temp - end_temp) * (min(eff_epoch, eff_total) / eff_total)
                if hasattr(model[0], 'mome_router'):
                    model[0].mome_router.temperature.data.fill_(current_temp)

            # 2. Curriculum Dropout Rate Calculation
            current_dropout_prob = 0.0
            if is_mome_active and epochs > dropout_start_epoch:
                # 线性增加概率
                dropout_progress = (epochs - dropout_start_epoch) / (total_epochs - dropout_start_epoch)
                current_dropout_prob = min(max_dropout_prob, dropout_progress * max_dropout_prob)

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    # 传入当前 dropout 概率
                    output = model[0](text, audio, vision, is_distill=True,
                                      mome_active=is_mome_active,
                                      modality_dropout_prob=current_dropout_prob)

                    # ... (Distillation Code - Same) ...
                    logits_homo.extend([output['logits_l_homo'], output['logits_v_homo'], output['logits_a_homo']])
                    reprs_homo.extend([output['repr_l_homo'], output['repr_v_homo'], output['repr_a_homo']])
                    logits_hetero.extend(
                        [output['logits_l_hetero'], output['logits_v_hetero'], output['logits_a_hetero']])
                    reprs_hetero.extend([output['repr_l_hetero'], output['repr_v_hetero'], output['repr_a_hetero']])
                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)
                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)
                    edges_homo, _ = model[1](logits_homo, reprs_homo)
                    edges_hetero, _ = model[2](logits_hetero, reprs_hetero)

                    # ==================== Loss Calculation ====================
                    mome_loss = 0.0
                    if is_mome_active and 'mome_weights' in output:
                        loss_task = self.criterion(output['output_logit'], labels)

                        loss_aux = 0.0
                        if 'visual_pred' in output:
                            l_v = self.criterion(output['visual_pred'], labels)
                            l_a = self.criterion(output['audio_pred'], labels)
                            l_t = self.criterion(output['language_pred'], labels)
                            loss_aux = (l_v + l_a + l_t) / 3.0

                        loss_distill = 0.0
                        if 'weighted_uni_pred' in output:
                            loss_distill = self.mse_loss(output['weighted_uni_pred'], output['output_logit'].detach())

                        loss_balance = 0.0
                        if 'mome_weights' in output:
                            err_v = torch.abs(output['visual_pred'] - labels)
                            err_a = torch.abs(output['audio_pred'] - labels)
                            err_l = torch.abs(output['language_pred'] - labels)
                            modality_errors = torch.cat([err_v, err_a, err_l], dim=1)
                            if hasattr(model[0], 'mome_losses'):
                                loss_balance = model[0].mome_losses.balance_loss(output['mome_weights'],
                                                                                 modality_errors.detach())

                        mome_loss = 0.1 * loss_aux + 0.2 * loss_distill + 0.1 * loss_balance
                    else:
                        loss_task_all = self.criterion(output['output_logit'], labels)
                        loss_task = loss_task_all + \
                                    self.criterion(output['logits_l_homo'], labels) + \
                                    self.criterion(output['logits_v_homo'], labels) + \
                                    self.criterion(output['logits_a_homo'], labels) + \
                                    self.criterion(output['logits_l_hetero'], labels) + \
                                    self.criterion(output['logits_v_hetero'], labels) + \
                                    self.criterion(output['logits_a_hetero'], labels) + \
                                    self.criterion(output['logits_c'], labels)

                    # ... (Standard DMD Losses - Same) ...
                    loss_recon = self.MSE(output['recon_l'], output['origin_l']) + self.MSE(output['recon_v'], output[
                        'origin_v']) + self.MSE(output['recon_a'], output['origin_a'])
                    loss_s_sr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r']) + self.MSE(
                        output['s_v'].permute(1, 2, 0), output['s_v_r']) + self.MSE(output['s_a'].permute(1, 2, 0),
                                                                                    output['s_a_r'])

                    def safe_cosine_loss(input1, input2, target_val=-1):
                        if input1.dim() == 3:
                            batch_size, seq_len, feat_dim = input1.shape
                            return self.cosine(input1.reshape(batch_size * seq_len, feat_dim),
                                               input2.reshape(batch_size * seq_len, feat_dim),
                                               torch.tensor([target_val] * (batch_size * seq_len)).to(
                                                   input1.device)).mean(0)
                        else:
                            return self.cosine(input1, input2, torch.tensor([target_val]).to(input1.device)).mean(0)

                    loss_ort = safe_cosine_loss(output['s_l'], output['c_l']) + safe_cosine_loss(output['s_v'], output[
                        'c_v']) + safe_cosine_loss(output['s_a'], output['c_a'])

                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.extend([c_l[i].view(1, -1), c_v[i].view(1, -1), c_a[i].view(1, -1)])
                        ids.extend([labels[i].view(1, -1)] * 3)
                    loss_sim = self.sim_loss(torch.cat(ids, dim=0), torch.cat(feats, dim=0))

                    loss_reg_homo, loss_logit_homo, loss_repr_homo = model[1].distillation_loss(logits_homo, reprs_homo,
                                                                                                edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = model[2].distillation_loss(logits_hetero,
                                                                                                      reprs_hetero,
                                                                                                      edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    combined_loss = loss_task + graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1) * 0.1 + \
                                    mome_loss

                    combined_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    train_loss += combined_loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            # Logger 增加 dropout 信息
            drop_info = f", Drop={current_dropout_prob:.2f}" if current_dropout_prob > 0 else ""
            status = "Warmup" if not is_mome_active else f"MoME(T={current_temp:.2f}{drop_info})"

            logger.info(
                f">> Epoch: {epochs} [{status}] "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')
            if (min_or_max == 'min' and cur_valid <= best_valid - 1e-6) or (
                    min_or_max == 'max' and cur_valid >= best_valid + 1e-6):
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model[0].state_dict(), './pt/dmd.pth')
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                    # 测试时 mome_active=True, dropout=0
                    output = model(text, audio, vision, is_distill=True, mome_active=True, modality_dropout_prob=0.0)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results