
from Dataloader.tuh_dataloader import TUH_Dataloader
from tqdm import tqdm
from easydict import EasyDict
import einops
from pytorch_models.neureka_models import *
import torch
from utils.bce_labelsmooth import BinaryCrossEntropyWithLabelSmoothingAndWeights
from collections import defaultdict
import os
from colorama import Fore

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
        # torch.save(save_dict, file_name)
        if verbose:
            print(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
    except:
        raise Exception("Problem in model saving")

def load_dir(file_name, model, optimizer, scheduler, dataloaders):
    checkpoint = torch.load(file_name, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logs = checkpoint["logs"]
    config = checkpoint["configs"]
    if hasattr(dataloaders.train_loader, "generator"):
        dataloaders.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
    if "metrics" in checkpoint:
        dataloaders.metrics = checkpoint["metrics"]
    return model, optimizer, scheduler, logs, config, dataloaders

def load_encoder(enc_args, config):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = globals()[enc_args[num_enc]["model_class"]]
            args = enc_args[num_enc]["args"]
            if "encoders" in enc_args[num_enc]:
                enc_enc = load_encoder(enc_args[num_enc]["encoders"], config)
                enc = enc_class(encs = enc_enc, args = args)
            else:
                enc = enc_class(args = args, encs=[])
            pretrained_encoder_args =  enc_args[num_enc].get("pretrainedEncoder", {"use":False})
            if pretrained_encoder_args["use"]:
                # print("Loading encoder from {}".format(enc_args[num_enc]["pretrainedEncoder"]["dir"]))
                file_path = pretrained_encoder_args.get("dir","")
                if "save_base_dir" in config.model:
                    file_path = os.path.join(config.model.save_base_dir, file_path)
                checkpoint = torch.load(file_path)
                if "encoder_state_dict" in checkpoint:
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

                elif "best_model_state_dict" in checkpoint:
                    logging.info("Loading enc best model state dict from {}".format(file_path))
                    print(enc_args[num_enc]["model"])
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["best_model_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

            encs.append(enc)
        return encs

def load_model(config):
    enc = load_encoder(enc_args=config.model.get("encoders", []), config=config)
    model_class = globals()[config.model.model_class]
    model = model_class(encs=enc, args=config.model.args)
    model.cuda()
    return model

def load_optimizer_and_scheduler(config, model):

    if config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta1, config.optimizer.beta2))
    else:
        raise Exception("Optimizer type not supported")

    if config.scheduler.type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.T_0, T_mult=config.scheduler.T_mult)
    else:
        raise Exception("Scheduler type not supported")

    return optimizer, scheduler

def train(config):

    dataloaders = TUH_Dataloader(config)

    pytorch_net = load_model(config)
    optimizer, scheduler = load_optimizer_and_scheduler(config, pytorch_net)

    loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()

    if os.path.exists(config.model.save_dir) and config.model.get("load_ongoing", True):
        model, optimizer, scheduler, logs, config, dataloaders = load_dir(config.model.save_dir, pytorch_net, optimizer, scheduler, dataloaders)
    else:
        logs = {"best": {"loss": 1e20}}

    for epoch in range(config.early_stopping.max_epoch):
        if epoch not in logs:
            logs[epoch] = {"train": defaultdict(list), "val": defaultdict(list)}
        #Training loop
        pytorch_net.train()
        pbar = tqdm(enumerate(dataloaders.train_loader), total=len(dataloaders.train_loader), desc="Training", leave=None,)
        for current_step, batch in pbar :
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

            message = "Train Epoch {0:d} step {1:d} with ".format(epoch, current_step)
            for i, v in losses.items(): message += "{} : {:.6f} ".format(i, v)
            pbar.set_description(message)
            pbar.refresh()
            # print(message)

            threshold = 0.5
            binary_preds = (preds[0] > threshold)
            binary_preds = torch.nn.functional.one_hot(binary_preds.long().cuda().flatten(), num_classes=2).float()

            if current_step == 10:
                break

        #Validation loop
        pytorch_net.eval()
        pbar = tqdm(enumerate(dataloaders.valid_loader), total=len(dataloaders.valid_loader))
        for current_step, batch in pbar:
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

            message = "Val Epoch {0:d} step {1:d} with ".format(epoch, current_step)
            for i, v in losses.items(): message += "{} : {:.6f} ".format(i, v)
            pbar.set_description(message)
            pbar.refresh()

            logs[epoch]["val"]["losses"].append(total_loss.item())
            if current_step == 10:
                break

        this_epoch_val_loss = torch.tensor(logs[epoch]["val"]["losses"]).mean()

        is_best_model = False
        if this_epoch_val_loss < logs["best"]["loss"]:
            logs["best"]["loss"] = this_epoch_val_loss
            is_best_model = True
        model_save = True if epoch == 0 else False
        save_model(model_save, is_best_model, config.model.save_dir, pytorch_net, optimizer, scheduler, logs, config, dataloaders)


