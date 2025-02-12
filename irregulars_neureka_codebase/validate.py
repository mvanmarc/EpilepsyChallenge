
from Dataloader.tuh_dataloader import TUH_Dataloader
from Dataloader.seizit2_dataloader import SeizIT2_Dataloader
from Dataloader.tuhseizit2_dataloader import TUHSeizIT2_Dataloader
from tqdm import tqdm
from easydict import EasyDict
import einops
from pytorch_models.neureka_models import *
import torch
from utils.bce_labelsmooth import BinaryCrossEntropyWithLabelSmoothingAndWeights
from collections import defaultdict
import os
from colorama import Fore
from utils.deterministic_pytorch import deterministic
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
import sys
import pathlib
sys.path.insert(0, './')
from SzcoreEvaluation import MetricsStore

def save_model(model_save, is_best_model, unwrapped_model, optimizer, scheduler, logs, config, dataloaders, post_test_results=None, verbose=True):
    save_dict = {}
    savior = {}
    file_name = config.model.save_dir
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

def load_dir(file_name, model, optimizer, scheduler, dataloaders):
    checkpoint = torch.load(file_name, map_location="cpu")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logs = checkpoint["logs"]
    config = checkpoint["configs"]
    if hasattr(dataloaders.train_loader, "generator"):
        dataloaders.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
    if "metrics" in checkpoint:
        dataloaders.metrics = checkpoint["metrics"]
    print(Fore.WHITE + "Model has loaded successfully from {}".format(file_name))
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

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def validate(model, dataloaders, logs, epoch, loss, config, set_name="val"):

    metricsStore = MetricsStore(config)
    
    # Validation loop
    model.eval()

    total_preds = defaultdict(list)
    pbar = tqdm(enumerate(dataloaders), total=len(dataloaders))
    for current_step, batch in pbar:
        data = batch['data']['raw'].cuda().float()
                label = batch['label'].cuda() # 128 x 4096
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1) # 128 x 1 x 4096 x 18
        preds = model(data) # dictionary with keys [0,1,2,3,4,5] and values [128x4096,128x1024,128x256,128x64,128x16,128x4]
        losses = {}

        metricsStore.evaluate_multiple_predictions(label, preds[0]>.5, batch['patient'])

        for key_pred, pred in preds.items():
            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze() # 128 x 4096
            total_preds[key_pred].append(pred.detach().cpu().numpy())
            total_preds["{}_label".format(key_pred)].append(this_label.detach().cpu().numpy())

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss

        total_loss = losses[0] + 0.2 * (torch.tensor([v for k, v in losses.items() if k != 0]).sum())

        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}

        if epoch in logs:
            logs[epoch][set_name]["losses"].append(total_loss.item())
            if set_name == "test":
                logs["best"]["test_loss"] = logs[epoch][set_name]["losses"]

        aggr_loss = {key: torch.tensor(val).mean().item() for key, val in losses.items()}
        message = "{0:} Epoch {1:d} step {2:d} with ".format(set_name, epoch, current_step)
        for i, v in aggr_loss.items(): message += "{} : {:.6f} ".format(i, v)
        pbar.set_description(message)
        pbar.refresh()

        # if current_step == 1000:
        #     break

    metrics = defaultdict(dict)

    outDir = pathlib.Path("./irregulars_neureka_codebase/predictions/"+set_name)
    metricsStore.store_scores(outDir)
    metricsStore.store_metrics(outDir, outDir)

    for total_size in [1000, 400, 200, 100]: #fs=200 so 5, 2, 1, 0.5 seconds
        this_label = np.concatenate(total_preds["{}_label".format(0)])
        this_pred = np.concatenate(total_preds[0], axis=0)
        this_pred = (this_pred > 0.5)
        #TODO: Remove this necessity to transform to torch Tensor and back in numpy
        this_label = torch.nn.functional.interpolate(torch.from_numpy(this_label).unsqueeze(dim=1), size=(total_size), mode='nearest').flatten().numpy()
        this_pred = torch.nn.functional.interpolate(torch.from_numpy(this_pred).unsqueeze(dim=1).float(), size=(total_size), mode='nearest').flatten().numpy().astype(int)

        #print unique label count in percentage with 2 decimal points
        unique, counts = np.unique(this_label, return_counts=True)
        if len(unique) == 1:
            print("Label percentage 0: {:.2f}% 1: {:.2f}%".format(1, 0))
        else:
            print("Label percentage 0: {:.2f}% 1: {:.2f}%".format(counts[0]/len(this_label), counts[1]/len(this_label)))

        metrics[total_size]["f1"] = f1_score(this_label, this_pred)
        metrics[total_size]["auc"] = roc_auc_score(this_label, this_pred) if len(np.unique(this_label)) > 1 else 0
        metrics[total_size]["confusion_matrix"] = confusion_matrix(this_label, this_pred)
        metrics[total_size]["accuracy"] = accuracy_score(this_label, this_pred)
        metrics[total_size]["precision"] = precision_score(this_label, this_pred)
        metrics[total_size]["recall"] = recall_score(this_label, this_pred)
        metrics[total_size]["specificity"] = specificity_score(this_label, this_pred) if len(np.unique(this_label)) > 1 else 0
        metrics[total_size]["false_alarm_rate"] = 1 - metrics[total_size]["specificity"] if len(np.unique(this_label)) > 1 else 0

    message = "{0:} Epoch {1:d} step {2:d} with \n".format(set_name, epoch, current_step)
    for i, v in metrics.items():
        message += "Window size: {0:.1f}sec \n".format(i/200)
        for key, val in v.items():
            #do sth for the confusion in one line
            if key == "confusion_matrix":
                message += "{} : [".format(key)
                for row in val:
                    message += " ".join([str(i) for i in row]) + ","
                message += "] ".format(key)

            else:
                message += "{} : {:.3f} ".format(key, val)
        message += "\n"
    print(message)

    return logs

def is_best(logs, epoch):
    this_epoch_val_loss = torch.tensor(logs[epoch]["val"]["losses"]).mean()
    is_best_model = False
    if this_epoch_val_loss < logs["best"]["loss"]:
        logs["best"]["loss"] = this_epoch_val_loss
        is_best_model = True
    return is_best_model

def train_loop(epoch, model, optimizer, scheduler, loss, dataloader, logs, config):
    
    metricsStoreTrain = MetricsStore(config)
    
    # Training loop
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=None, )
    for current_step, batch in pbar:
        optimizer.zero_grad()
        data = batch['data']['raw'].cuda().float()
        label = batch['label'].cuda()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)

        preds = model(data)
        losses = {}

        metricsStoreTrain.evaluate_multiple_predictions(label, preds[0]>.5, batch['patient'])

        for key_pred, pred in preds.items():
            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze()

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss

        total_loss = losses[0] + 0.2 * (torch.tensor([v for k, v in losses.items() if k != 0]).sum())

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}
        logs[epoch]["train"]["losses"].append(total_loss.item())


        aggr_loss = {key: torch.tensor(val).mean().item() for key, val in losses.items()}
        message = "Train Epoch {0:d} step {1:d} with ".format(epoch, current_step)
        for i, v in aggr_loss.items(): message += "{} : {:.6f} ".format(i, v)
        pbar.set_description(message)
        pbar.refresh()

        # threshold = 0.5
        # binary_preds = (preds[0] > threshold)
        # binary_preds = torch.nn.functional.one_hot(binary_preds.long().cuda().flatten(), num_classes=2).float()

        # if current_step == 10:
        #     break

    outDir = pathlib.Path("./irregulars_neureka_codebase/predictions/train/epoch_"+epoch)
    try:
        os.mkdir(outDir)
    except Exception as exc:
        pass
    metricsStoreTrain.store_scores(outDir)
    metricsStoreTrain.store_metrics(outDir, outDir)

    return logs

def load_best_model(model, config):
    if os.path.exists(config.model.save_dir):
        checkpoint = torch.load(config.model.save_dir, map_location="cpu")
        model.load_state_dict(checkpoint["best_model_state_dict"])
    return model


def validate(config):
    deterministic(config.training_params.seed)
    dataloader_class = globals()[config.dataset.dataloader_class]
    dataloaders = dataloader_class(config)

    model = load_model(config)
    optimizer, scheduler = load_optimizer_and_scheduler(config, model)

    loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()

    if os.path.exists(config.model.save_dir) and config.model.get("load_ongoing", True):
        model, optimizer, scheduler, logs, config, dataloaders = load_dir(config.model.save_dir, model, optimizer, scheduler, dataloaders)
    else:
        logs = {"best": {"loss": 1e20}}

    try:
        _ = validate(model, dataloaders.valid_loader, logs, -1, loss, config, "val")

        for epoch in range(config.early_stopping.max_epoch):
            if epoch not in logs:
                logs[epoch] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

            logs = train_loop(epoch, model, optimizer, scheduler, loss, dataloaders.train_loader, logs, config)

            logs = validate(model, dataloaders.valid_loader, logs, epoch, loss, config, "val")

            save_model(True if epoch == 0 else False,
                       is_best(logs, epoch),
                       model, optimizer, scheduler, logs, config, dataloaders)
    #catch a control c and save the model
    except KeyboardInterrupt:
        save_model(False, False, model, optimizer, scheduler, logs, config, dataloaders)
        raise KeyboardInterrupt

    model = load_best_model(model, config)
    logs = validate(model, dataloaders.test_loader, logs, epoch, loss, config, "test")
    save_model(True if epoch == 0 else False,
               is_best(logs, epoch),
               model, optimizer, scheduler, logs, config, dataloaders)

