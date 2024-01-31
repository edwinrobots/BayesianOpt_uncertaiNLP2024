from transformers.utils.dummy_pt_objects import DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST
from swag.swag import SWAG
from swag import utils
import torch
import os
from dataset_collection import SEPairwiseDataset, SESingleDataset, create_dataset
import transformers as tf
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import trange


class BayesainBert(torch.nn.Module):
    def __init__(self,
                 base_model,
                 subspace,
                 max_num_models,
                 device,
                 lr_init,
                 momentum,
                 weight_decay,
                 swag_start,
                 swag_lr,
                 swag_c_epochs,
                 epochs, 
                 batch_size,
                 pretrained_model) -> None:
        super().__init__()
        self.swag = SWAG(base_model,
                         subspace_type=subspace,
                         subspace_kwargs={'max_rank': max_num_models})
        self.base_model = base_model
        self.swag.to(device)
        self.swag_start = swag_start
        self.lr_init = lr_init
        self.swag_lr = swag_lr
        self.loss_fn = nn.MarginRankingLoss(margin=0.1)
        self.swag_c_epochs = swag_c_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.pretrained_model = pretrained_model

        self.optimizer = torch.optim.SGD(self.base_model.parameters(),
                                         lr=self.lr_init,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        # self.optimizer = AdamW(self.base_model.parameters(), lr=self.lr_init, correct_bias=False, weight_decay=weight_decay)
        self.set_swag = False

    def train(self, data_loader,  valid_loader, save_dir, stop_epochs, save_name, checkpoint, schedule=True):
        val_acc, save_epoch, start_epoch = 0, 0, 0
        if checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, self.epochs):
            print(f'---- epoch: {epoch}/{self.epochs} ----')
            if schedule:
                lr = utils.schedule(epoch,
                                    swag_start=self.swag_start,
                                    swag_lr=self.swag_lr,
                                    lr_init=self.lr_init)
                utils.adjust_learning_rate(self.optimizer, lr)
            utils.train_epoch(data_loader,
                              self.base_model,
                              self._loss,
                              self.optimizer,
                              verbose = True,
                              regularizer=None)

            if (epoch + 1) >= self.swag_start and (
                    epoch + 1 - self.swag_start) % self.swag_c_epochs == 0:
                #sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
                # sgd_res = utils.predict(loaders["test"], model)
                # sgd_preds = sgd_res["predictions"]
                # sgd_targets = sgd_res["targets"]
                # # print("updating sgd_ens")
                # if sgd_ens_preds is None:
                #     sgd_ens_preds = sgd_preds.copy()
                # else:
                #     #TODO: rewrite in a numerically stable way
                #     sgd_ens_preds +=  (sgd_preds - sgd_ens_preds)/ (n_ensembled + 1)
                # sgd_ens_nll = utils.nll(sgd_ens_preds, sgd_targets)
                # sgd_ens_acc = (np.argmax(sgd_ens_preds, axis=1) == sgd_targets).mean()

                self.swag.collect_model(self.base_model)
                # _, var = self.swag._get_mean_and_variance()
                # print(f'wegiths var at epoch {epoch}', var)

                # if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
                self.swag.set_swa()
                swag_metric = utils.eval(valid_loader, self.base_model, self._loss)
                print('validation', swag_metric)
                if swag_metric['accuracy'] > val_acc:
                    print('finding better model')
                    save_epoch = epoch
                    val_acc = swag_metric['accuracy']

                if epoch - save_epoch > stop_epochs:
                    print(
                        f'stopping without any improvement after {stop_epochs}')
                    break

        utils.save_checkpoint(save_dir, epoch=epoch, name=save_name, state_dict=self.state_dict(
                ), optimizer=self.optimizer.state_dict(), scheduler=None)
        print(f'save {epoch} model to {os.path.join(save_dir, save_name)}')


    def _loss(self, model, target, **kwargs):
        # standard cross-entropy loss function
        output = model(**kwargs)
        output = torch.cat(output, axis=0)
        # loss = F.cross_entropy(output, target)
        loss = self.loss_fn(output[0,:], output[1,:], target)
        return loss, output, {}

    def update(self, data, stop_epochs):
        dataset = SEPairwiseDataset(log_data=data, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        min_loss = float('inf')
        for epoch in range(self.epochs):
            metric = utils.train_epoch(data_loader,
                              self.base_model,
                              self._loss,
                              self.optimizer,
                              verbose=True,
                              regularizer=None,
                              scheduler=None)
            # if epoch // 2
            
            if metric['loss'] < min_loss:
                end_epoch = epoch
                min_loss = metric['loss']
            if (metric['accuracy'] == 100.0) or (epoch - end_epoch) > stop_epochs:
                print('no improvement stopping...')
                print(f'----update epoch: {epoch+1}/{self.epochs}, {metric} ----')
                break
            print(f'----update epoch: {epoch+1}/{self.epochs}, {metric} ----')
        self.swag.collect_model(self.base_model)

    def predict(self, test_data, question, swag_samples, eval=None):
        dataset = SESingleDataset(log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=512, shuffle=False)
        all_scores = np.zeros((len(dataset), swag_samples))
        # torch.manual_seed(2021)
        for i in trange(swag_samples):
            self.swag.sample()
            # print('params', params)
            # print(len(params))
            scores = utils.predict(data_loader, self.swag.base_model)
            all_scores[:, i] = scores
        # print('score', all_scores)

        # if not set_swag:
        #     self.swag.set_swa()
        # utilities = utils.predict(data_loader, self.swag)
        self.mean, self.var = np.mean(
            all_scores, axis=1), np.var(all_scores, axis=1)
        return self.mean, self.var

    def get_utilities(self, test_data, question, sample_nums=20):
        # mean, _ = self.predict(test_data, question, sample_nums)
        # return mean
        dataset = SESingleDataset(log_data=test_data, question=question, pretrained_model=self.pretrained_model)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # if not self.set_swag:
        self.swag.set_swa()
        #     self.set_swag = True
        return utils.predict(data_loader, self.swag.base_model)
