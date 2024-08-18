import os

import torch
import lightning.pytorch as pl

from monai.transforms import AsDiscrete

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from calc_circum import calc_circum
from conditional_dice import ConditionalDice
from fix_model_state_dict import fix_model_state_dict
from get_optimizer import get_optimizer
from loss import Loss


class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModule, self).__init__()
        self.cfg = cfg
        
        if cfg.General.mode == "train":
            self.lossfuns = Loss(cfg.Loss)
            self.lossfuns_valid = Loss(cfg.Loss)
        elif cfg.General.mode == "validate":
            self.lossfuns_valid = Loss(cfg.Loss)
        self.txt_logger = cfg.txt_logger

        self.training_step_outputs = []        
        self.validation_step_outputs = []

        # model
        if cfg.Model.arch == 'unet_multitask':
            from unet_multitask import UnetMultitask
            self.model = UnetMultitask(**cfg.Model.params)              
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')
        
        if cfg.Model.pretrained is not None:
            # Load pretrained model weights
            print(f'Loading: {cfg.Model.pretrained}')
            checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = fix_model_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
        
        # post processing
        self.post_pred = AsDiscrete(argmax=True, dim=1)

        # metrics
        seg_metrics_fun = MetricCollection([
            ConditionalDice(num_classes=self.cfg.Data.dataset.num_classes,
                            ignore_index=0)
        ])        
        self.seg_metrics_list = ['ConditionalDice']
        cls_metrics_fun = MetricCollection([
            BinaryAccuracy(multidim_average='global'),
            BinaryF1Score(multidim_average='global'),
            BinaryPrecision(multidim_average='global'),
            BinaryRecall(multidim_average='global'),
        ])
        self.cls_metrics_list = ['BinaryAccuracy', 'BinaryF1Score', 'BinaryPrecision', 'BinaryRecall']

        self.train_seg_metrics_fun = seg_metrics_fun.clone(prefix='train_')
        self.valid_seg_metrics_fun = seg_metrics_fun.clone(prefix='valid_')
        self.train_cls_metrics_fun = cls_metrics_fun.clone(prefix='train_')
        self.valid_cls_metrics_fun = cls_metrics_fun.clone(prefix='valid_')
        
        # flag to check the validation is performed or not at the end of epoch
        self.did_validation = False

    def forward(self, x):
        y = self.model(x)
        return y

    def on_train_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()
        self.did_validation = False

    def training_step(self, batch, batch_idx):        
        image = batch["image"]
        mask = batch["mask"]
        clas = batch["clas"]
        
        if self.cfg.Model.arch == 'unet_multitask':
            # forward
            pred_mask, pred_clas = self.forward(image)
            # loss
            pred = {}
            target = {}
            for lossfun in self.cfg.Loss.lossfuns:
                name = self.cfg.Loss[lossfun].name
                if lossfun == "ConditionalDiceLoss" or lossfun == "HausdorffDTLoss":
                    pred[name] = pred_mask
                    target[name] = mask
                elif lossfun == "CrossEntropyLoss" or lossfun == "FocalLoss":
                    pred[name] = pred_clas
                    target[name] = clas                
            total_loss, losses = self.lossfuns(pred, target)
        
            # metrics
            pred_mask = self.post_pred(pred_mask)
            
            for metric in self.seg_metrics_list:
                self.train_seg_metrics_fun[metric].update(preds=pred_mask.int(),
                                                          target=mask.int())
            for metric in self.cls_metrics_list:
                self.train_cls_metrics_fun[metric].update(preds=torch.sigmoid(pred_clas[:, 1]),
                                                          target=clas)
            output = {"loss": total_loss,
                      "losses": losses}
        
        self.training_step_outputs.append(output)

        return total_loss

    def on_train_epoch_end(self):
        # print the results
        outputs_gather = self.all_gather(self.training_step_outputs)

        # mmetrics (each metric is synced and reduced across each process)
        train_seg_metrics = self.train_seg_metrics_fun.compute()
        self.train_seg_metrics_fun.reset()
        train_cls_metrics = self.train_cls_metrics_fun.compute()
        self.train_cls_metrics_fun.reset()
        
        if self.trainer.is_global_zero:
            epoch = int(self.current_epoch)
            self.txt_logger.info(f'Epoch: {epoch}')
           
            train_loss = torch.stack([o['loss']
                                      for o in outputs_gather]).mean().detach()
            
            train_losses = dict()
            lossfuns = self.cfg.Loss.lossfuns
            for lossfun in lossfuns:
                name = self.cfg.Loss[lossfun].name
                train_losses[name] = torch.stack([o['losses'][name]
                                                  for o in outputs_gather]).mean().detach()
            
            # log
            d = dict()
            d['epoch'] = epoch
            d['train_loss'] = train_loss
            for key in train_losses.keys():
                d[f'train_{key}'] = train_losses[key]
            d.update(train_seg_metrics)
            d.update(train_cls_metrics)

            print('\n Mean:')
            s = f'  Train:\n'
            s += f'    loss: {train_loss.item():.3f}'
            for key in train_losses.keys():
                s += f'  {key}: {train_losses[key]:.3f}'
            s += '\n'
            s += '  '
            for metric in self.seg_metrics_list:
                s += f'  {metric.replace("Binary", "")}: {train_seg_metrics[f"train_{metric}"].cpu().numpy():.3f}'
            for metric in self.cls_metrics_list:
                s += f', {metric.replace("Binary", "")}: {train_cls_metrics[f"train_{metric}"].cpu().numpy():.3f}'
            print(s)
            self.txt_logger.info(s)
            
            if self.did_validation:
                s = '  Valid:\n'
                s += f'    loss: {self.valid_loss:.3f}'
                for key in self.valid_losses.keys():
                    s += f'  {key}: {self.valid_losses[key]:.3f}'
                s += '\n'
                s += '  '
                for metric in self.seg_metrics_list:
                    s += f'  {metric.replace("Binary", "")}: {self.valid_seg_metrics[f"valid_{metric}"].cpu().numpy():.3f}'
                for metric in self.cls_metrics_list:
                    s += f'  {metric.replace("Binary", "")}: {self.valid_cls_metrics[f"valid_{metric}"].cpu().numpy():.3f}'
                print(s)
                self.txt_logger.info(s)

            self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # free up the memory
        self.training_step_outputs.clear()
        
    def on_validation_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()        
     
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        mask = batch["mask"]
        clas = batch["clas"]
        circum = batch["circum"]
        opt_circum = batch["opt_circum"]
        filepath = batch["filepath"]

        if self.cfg.Model.arch == 'unet_multitask':
            # forward
            pred_mask, pred_clas = self.forward(image)

            # print
            if self.cfg.General.mode == "validate":
                mm_pixel = 0.28 / self.cfg.Transform.resize_ratio
                pred_clas_sm = torch.softmax(pred_clas, dim=1).cpu()
                pred_mask_am = torch.argmax(pred_mask, dim=1).cpu().numpy()
                for b in range(pred_clas.shape[0]):
                    if pred_clas_sm[b][1] >= 0.5:
                        pred_circ, pred_flatten = calc_circum(pred_mask_am[b], mm_pixel)
                    else:
                        pred_circ = 0
                        pred_flatten = 0
                    
                    s = f"path: {os.path.basename(filepath[b])}, "
                    s += f"pred: {torch.argmax(pred_clas[b])}, true: {clas[b]}, "
                    s += f"pred_circ: {pred_circ:.4f}, true_circ: {circum[b]:.4f}, "
                    s += f"opt_circ: {opt_circum[b]:.4f}, pred_flatten: {pred_flatten:.4f}"
                    print(s)

            # loss
            pred = {}
            target = {}
            for lossfun in self.cfg.Loss.lossfuns:
                name = self.cfg.Loss[lossfun].name
                if lossfun == "ConditionalDiceLoss" or lossfun == "HausdorffDTLoss":
                    pred[name] = pred_mask
                    target[name] = mask
                elif lossfun == "CrossEntropyLoss" or lossfun == "FocalLoss":
                    pred[name] = pred_clas
                    target[name] = clas                
            total_loss, losses = self.lossfuns_valid(pred, target)

            # metrics
            pred_mask = self.post_pred(pred_mask)
            for metric in self.seg_metrics_list:
                self.valid_seg_metrics_fun[metric].update(preds=pred_mask.int(),
                                                          target=mask.int())
            for metric in self.cls_metrics_list:
                self.valid_cls_metrics_fun[metric].update(preds=torch.sigmoid(pred_clas[:, 1]),
                                                          target=clas)
            output = {"loss": total_loss,
                      "losses": losses}
        
        self.validation_step_outputs.append(output)
        
        return
   
    def on_validation_epoch_end(self):
        # all gather
        outputs_gather = self.all_gather(self.validation_step_outputs)

        # metrics (in torchmetrics, gathered automatically)
        self.valid_seg_metrics = self.valid_seg_metrics_fun.compute()
        self.valid_seg_metrics_fun.reset()        
        self.valid_cls_metrics = self.valid_cls_metrics_fun.compute()
        self.valid_cls_metrics_fun.reset()        
        
        epoch = int(self.current_epoch)
        
        # loss
        valid_loss = torch.stack([o['loss']
                                  for o in outputs_gather]).mean().item()
        self.valid_loss = valid_loss

        self.valid_losses = dict()
        lossfuns = self.cfg.Loss.lossfuns
        for lossfun in lossfuns:
            name = self.cfg.Loss[lossfun].name
            self.valid_losses[name] = torch.stack([o['losses'][name]
                                                   for o in outputs_gather]).mean().detach()

        # log
        d = dict()
        d['epoch'] = epoch
        d['valid_loss'] = valid_loss
        for key in self.valid_losses.keys():
            d[f'valid_{key}'] = self.valid_losses[key]
        
        for metric in self.seg_metrics_list:
            d[f'valid_{metric}'] = self.valid_seg_metrics[f'valid_{metric}']
        for metric in self.cls_metrics_list:
            d[f'valid_{metric}'] = self.valid_cls_metrics[f'valid_{metric}']
        self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # free up the memory
        self.validation_step_outputs.clear()

        # setup flag
        self.did_validation = True

    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        elif isinstance(obj, str):
            return obj
        else:
            print('obj (error):', obj)
            raise TypeError("Invalid type for move_to")

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        def is_encoder(n): return 'encoder' in n

        params = list(self.model.named_parameters())
        
        base_lr = conf_optim.optimizer.params.lr
        encoder_lr = base_lr * conf_optim.encoder_lr_ratio

        grouped_parameters = [
            {"params": [p for n, p in params if is_encoder(n)], 'lr': encoder_lr},
            {"params": [p for n, p in params if not is_encoder(n)], 'lr': base_lr},
        ]

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            """
            optimizer = optimizer_cls(self.parameters(),
                                      **conf_optim.optimizer.params)
            """
            optimizer = optimizer_cls(grouped_parameters,
                                      **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(grouped_parameters)

        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            
            return [optimizer], [scheduler]
       
    def get_progress_bar_dict(self):
        items = dict()

        return items
