
from irregulars_neureka_codebase.Dataloader.tuh_dataloader import TUH_Dataloader
from tqdm import tqdm
from easydict import EasyDict
import einops
from irregulars_neureka_codebase.pytorch_models.neureka_models import NeurekaNet
import torch
from irregulars_neureka_codebase.utils.bce_labelsmooth import BinaryCrossEntropyWithLabelSmoothingAndWeights
from collections import defaultdict
import os
from colorama import Fore


config = EasyDict()
config.dataset = EasyDict()
config.dataset.window_size = 4096
config.dataset.stride = 4096
config.dataset.data_path = "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5"
config.training_params = EasyDict()
config.training_params.len_sample = 4096 * 30
config.training_params.fs = 200
config.training_params.batch_size = 32
config.training_params.test_batch_size = 32
config.training_params.pin_memory = False
config.training_params.num_workers = 6
config.training_params.seed = 0
config.training_params.total_num_epochs = 5
config.optimizer = EasyDict()
config.optimizer.lr = 0.001
config.optimizer.weight_decay = 0.0001
config.optimizer.beta1 = 0.9
config.optimizer.beta2 = 0.999
config.scheduler = EasyDict()
config.scheduler.T_0 = 10
config.scheduler.T_mult = 1
config.model = EasyDict()
config.model.window_size = 4096
config.model.n_channels = 18
config.model.n_filters = 8
config.model.pre_dir_raw = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_raw.pth'
config.model.pre_dir_wiener = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_wiener.pth'
config.model.pre_dir_iclabel = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_iclabel.pth'
config.model.pre_dir_lstm = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_lstm.pth'

config.model.save_preds = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/prediction_test_raw.h5'
config.model.save_dir = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/pytorch_neureka.pth'

def save_model(model_save, is_best_model, file_name, unwrapped_model, optimizer, scheduler, logs, config, dataloaders, post_test_results=None, verbose=True):
    save_dict = {}
    savior = {}
    savior["optimizer_state_dict"] = optimizer.state_dict()
    savior["scheduler_state_dict"] = scheduler.state_dict()
    savior["logs"] = logs
    savior["configs"] = config
    if hasattr(dataloaders.train_loader, "generator"):
        savior["training_dataloder_generator_state"] = dataloaders.train_loader.generator.get_state()

    if not model_save:
        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu")
            if "best_model_state_dict" in prev_checkpoint:
                savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
            if "model_state_dict" in prev_checkpoint:
                savior["model_state_dict"] = prev_checkpoint["model_state_dict"]
    else:
        savior["model_state_dict"] = unwrapped_model.state_dict()

    if is_best_model:
        savior["best_model_state_dict"] = unwrapped_model.state_dict()
    else:
        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu")
            if "best_model_state_dict" in prev_checkpoint:
                savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
        else:
            savior["best_model_state_dict"] = unwrapped_model.state_dict()
    if post_test_results:
        savior["post_test_results"] = post_test_results
    if hasattr(dataloaders, "metrics"):
        savior["metrics"] = dataloaders.metrics

    save_dict.update(savior)

    try:
        # accelerator.save(save_dict, file_name)
        torch.save(save_dict, file_name)
        if verbose:
            print(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
    except:
        raise Exception("Problem in model saving")


dataloaders = TUH_Dataloader(config)

pytorch_net = NeurekaNet(config)
pytorch_net.cuda()

optimizer = torch.optim.Adam(pytorch_net.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta1, config.optimizer.beta2))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.T_0, T_mult=config.scheduler.T_mult)

loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()

logs = {"best": {"loss": 1e20}}
for epoch in range(config.training_params.total_num_epochs):
    logs[epoch] = {"train": defaultdict(list), "val": defaultdict(list)}
    #Training loop
    pytorch_net.train()
    for i, batch in tqdm(enumerate(dataloaders.train_loader), total=len(dataloaders.train_loader)):
        optimizer.zero_grad()
        data = batch['data']['raw'].cuda().float()
        label = batch['label'].cuda()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)

        preds = pytorch_net(data)
        losses = {}
        for key_pred, pred in preds.items():

            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze()

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss

        total_loss = losses[0] + 0.2*(torch.tensor([v for k, v in losses.items() if k!=0]).sum())

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}
        logs[epoch]["train"]["losses"].append(total_loss.item())

        threshold = 0.5
        binary_preds = (preds[0] > threshold)
        binary_preds = torch.nn.functional.one_hot(binary_preds.long().cuda().flatten(), num_classes=2).float()

        if i == 10:
            break

    #Validation loop
    pytorch_net.eval()
    for i, batch in tqdm(enumerate(dataloaders.valid_loader), total=len(dataloaders.valid_loader)):
        data = batch['data']['raw'].cuda().float()
        label = batch['label'].cuda()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)

        preds = pytorch_net(data)
        losses = {}
        for key_pred, pred in preds.items():

            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze()

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss


        total_loss = losses[0] + 0.2*(torch.tensor([v for k, v in losses.items() if k!=0]).sum())

        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}
        logs[epoch]["val"]["losses"].append(total_loss.item())
        if i == 10:
            break

    this_epoch_val_loss = torch.tensor(logs[epoch]["val"]["losses"]).mean()

    is_best_model = False
    if this_epoch_val_loss < logs["best"]["loss"]:
        logs["best"]["loss"] = this_epoch_val_loss
        is_best_model = True
    model_save = True if epoch == 0 else False
    save_model(model_save, is_best_model, config.model.save_dir, pytorch_net, optimizer, scheduler, logs, config, dataloaders)


