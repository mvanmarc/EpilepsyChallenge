
import csv
import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
import json
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import h5py
from library import nedc

class TUHDataset(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode = mode

        data_path = config.get("data_path", "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5")
        self.h5_file = h5py.File(data_path, 'r')

        #or load it from cumulated dict
        with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/tuh_patient_dict.pkl", "rb") as f:
            self.patient_dict = pickle.load(f)
        #

        # count = {"non_seizure": 0, "seizure": 0}
        # self.patient_dict = {}
        # percentages = []
        # for dataset in tqdm(self.h5_file.keys()):
        #     self.patient_dict[dataset] = {}
        #     for patient in self.h5_file[dataset].keys():
        #         self.patient_dict[dataset][patient] = {}
        #         for session in self.h5_file[dataset][patient].keys():
        #             # for rec in self.h5_file[dataset][patient][session].keys():
        #             duration = self.h5_file[dataset][patient][session]["duration"][()]
        #             events = self.h5_file[dataset][patient][session]["events"][()]
        #             if len(events) > 0:
        #                 count["seizure"] += 1
        #             else:
        #                 count["non_seizure"] += 1
        #
        #             labels_per_window, perc = self.estimate_labels_per_window(events, duration)
        #             percentages.append(perc)
        #             self.patient_dict[dataset][patient][session] = {"duration": duration, "events": events, "labels_per_window": labels_per_window}
        # print(np.mean(percentages))
        # #remove zero from the percentages
        # percentages = [x for x in percentages if x != 0]
        # print("Average seizure percentage in seconds: ", np.mean(percentages))
        #
        # with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/tuh_patient_dict.pkl", "wb") as f:
        #     pickle.dump(self.patient_dict, f)

        self._subselect_patients(mode=mode)

        self._get_cumulative_lens()

    def estimate_labels_per_window(self, events, duration):
        labels_per_window = np.zeros(int(duration)*self.config.dataset.fs//self.config.dataset.window_size)
        total_sz_duration = 0
        for event in events:
            start = event[0]*self.config.dataset.fs/self.config.dataset.window_size
            end = event[1]*self.config.dataset.fs/self.config.dataset.window_size
            start = int(np.floor(start))
            end = int(np.ceil(end))
            labels_per_window[start:end] = 1
            total_sz_duration += event[1]-event[0]

        #estimate percentage of seizure windows lengths
        # print("Percentage of seizure windows lengths: {0:.2f}".format(total_sz_duration / (int(duration))))

        return labels_per_window, total_sz_duration/(int(duration))

    def __len__(self):
        return int(list(self.cumulated_dict.keys())[-1])

    def _discard_non_seizure(self):
        new_patient_dict = {}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                for session in self.patient_dict[dataset][patient].keys():
                    if len(self.patient_dict[dataset][patient][session]["events"]) > 0:
                        if dataset not in new_patient_dict.keys():
                            new_patient_dict[dataset] = {}
                        if patient not in new_patient_dict[dataset].keys():
                            new_patient_dict[dataset][patient] = {}
                        new_patient_dict[dataset][patient][session] = self.patient_dict[dataset][patient][session]
        print("We discard non-seizure recordings")
        self.patient_dict = new_patient_dict

    def _discard_non_seizure_windows(self):
        #estimate current ratio of seizure to non-seizure windows
        count = {"non_seizure": 0, "seizure": 0}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                for session in self.patient_dict[dataset][patient].keys():
                    count["non_seizure"] += self.patient_dict[dataset][patient][session]["labels_per_window"].shape[0] - self.patient_dict[dataset][patient][session]["labels_per_window"].sum()
                    count["seizure"] += self.patient_dict[dataset][patient][session]["labels_per_window"].sum()
        current_ratio = count["seizure"]/count["non_seizure"]
        if current_ratio > self.config.dataset.ratio_sz_nsz:
            print("Current ratio is {} and desired ratio is {} so we discard nothing".format(current_ratio, self.config.dataset.ratio_sz_nsz))
            return
        #how many non seizure should we discard?
        to_discard = int((self.config.dataset.ratio_sz_nsz-current_ratio)*count["non_seizure"])
        discard_ratio = to_discard/count["non_seizure"]
        print("We discard randomly further {} non-seizure windows".format(to_discard))

        #discard non-seizure windows
        new_patient_dict = {}
        for dataset in self.patient_dict.keys():
            for patient in self.patient_dict[dataset].keys():
                for session in self.patient_dict[dataset][patient].keys():
                    if len(self.patient_dict[dataset][patient][session]["events"]) > 0:
                        if dataset not in new_patient_dict.keys():
                            new_patient_dict[dataset] = {}
                        if patient not in new_patient_dict[dataset].keys():
                            new_patient_dict[dataset][patient] = {}
                        new_patient_dict[dataset][patient][session] = self.patient_dict[dataset][patient][session]
                        discard_idx = np.zeros(len(new_patient_dict[dataset][patient][session]["labels_per_window"]))
                        for i in range(len(new_patient_dict[dataset][patient][session]["labels_per_window"])):
                            #check if previous or after are not seizures
                            if i > 0 and i < len(new_patient_dict[dataset][patient][session]["labels_per_window"])-1:
                                if new_patient_dict[dataset][patient][session]["labels_per_window"][i-1] == 0 and new_patient_dict[dataset][patient][session]["labels_per_window"][i+1] == 0:
                                    if random.random() < discard_ratio:
                                        discard_idx[i] = 1
                        new_patient_dict[dataset][patient][session]["discard_idx"] = discard_idx
                        new_duration = len(new_patient_dict[dataset][patient][session]["labels_per_window"]) - discard_idx.sum()

                        new_patient_dict[dataset][patient][session]["duration"] = (new_duration * self.config.dataset.window_size)/self.config.dataset.fs
        self.patient_dict = new_patient_dict

    def _subselect_patients(self, mode):

        patient_names = list(self.patient_dict.keys())
        train_patients, test_patients = train_test_split(patient_names, test_size=0.2, random_state=42)
        train_patients, val_patients = train_test_split(train_patients, test_size=0.2, random_state=42)

        if mode == "train":
            for patient in val_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "val":
            for patient in train_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "test":
            for patient in train_patients+val_patients:
                self.patient_dict.pop(patient)

        self._discard_non_seizure()
        if mode == "train":
            self._discard_non_seizure_windows()


    def check_for_discard(self, patient, session, recording):
        if len(self.patient_dict[patient][session][recording]["events"]) == 0:
            return True
        else:
            return False

    def _get_cumulative_lens(self):

        self.cumulated_dict = {}
        cum_idx = 0 #number of windows per patient
        for patient in self.patient_dict.keys():
            for session in self.patient_dict[patient].keys():
                for recording in self.patient_dict[patient][session].keys():
                    self.cumulated_dict[cum_idx] = {"patient": patient, "session": session, "recording": recording, "len": self.patient_dict[patient][session][recording]["duration"]*self.config.dataset.fs}
                    cum_idx += self.patient_dict[patient][session][recording]["duration"]*self.config.dataset.fs//self.config.dataset.window_size


    def _choose_patient_session_recording_len(self, idx):
        #find the cumulative idx that is right smaller than the idx
        for i, cum_idx in enumerate(list(self.cumulated_dict.keys())):
            if cum_idx > idx:
                break
        cum_idx = list(self.cumulated_dict.keys())[i-1]

        patient = self.cumulated_dict[cum_idx]["patient"]
        session = self.cumulated_dict[cum_idx]["session"]
        recording = self.cumulated_dict[cum_idx]["recording"]

        window_idx = idx-cum_idx
        #find the id of the window_idx in the discarding list that is zero
        if "discard_idx" in self.patient_dict[patient][session][recording].keys():
            okay_windows = 0
            for disc_id, v in enumerate(self.patient_dict[patient][session][recording]["discard_idx"]):
                if v == 0:
                    if okay_windows == window_idx:
                        break
                    okay_windows += 1
            window_idx = disc_id

        len_from = window_idx*self.config.dataset.window_size
        len_to = (window_idx+1)*self.config.dataset.window_size
        # if len_to > self.cumulated_dict[cum_idx]["len"]:
        #     len_to = self.cumulated_dict[cum_idx]["len"]
        return {"patient": patient, "session": session, "recording": recording, "len_from": int(len_from), "len_to": int(len_to)}

    def _windowize(self, data, window_size, stride):
        data = np.array(data)
        data = data[:len(data) // window_size * window_size]
        data = data.reshape(-1, window_size)
        return data

    def _get_signals(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        ch_names = [x.decode() for x in self.h5_file[patient][session][recording]['channel_names'][()]]
        sig = []
        for ch in ch_names:
            sig.append(self.h5_file[patient][session][recording][ch][len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def _get_label(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        events = self.h5_file[patient][session][recording]['events'][()]

        #find the label of the corresponding segment
        if len(events) == 0:
            return torch.zeros(len_to-len_from)
        else:
            window_start = len_from
            window_end = len_to
            duration = self.h5_file[patient][session][recording]['duration'][()]
            total_label = torch.zeros(int(duration*self.config.dataset.fs))

            for event in events:
                event[0] = event[0]*self.config.dataset.fs
                event[1] = event[1]*self.config.dataset.fs
                total_label[int(event[0]):int(event[1])] = 1
            return total_label[window_start:window_end]

    def __getitem__(self, idx):

        demographics = self._choose_patient_session_recording_len(idx)
        signal = self._get_signals(demographics)
        signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)

        label = self._get_label(demographics)

        return {"data":{"raw":signal},"label": label, "idx": idx, "patient":demographics["patient"]}


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # I have a list of dicts that I would like you to aggregate into a single dict of lists
    aggregated_batch = {}
    for key in batch[0].keys():
        aggregated_batch[key] = {}
        if type(batch[0][key]) is int:
            aggregated_batch[key] = torch.LongTensor([d[key] for d in batch])

    key = "data"
    subkey = 0 #Spectrogram
    aggregated_list = [d[key][subkey].unsqueeze(dim=0) for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        aggregated_batch[key][subkey] = torch.cat(aggregated_list, dim=0)

    subkey = 1 #Video
    aggregated_list = [d[key][subkey].unsqueeze(dim=0) for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        aggregated_batch[key][subkey] = torch.cat(aggregated_list, dim=0)

    subkey = 2 #Audio
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
        audio_attention_mask = torch.zeros((len(aggregated_list), max(length_list)))
        for data_idx, dur in enumerate(length_list):
            audio_attention_mask[data_idx, :dur] = 1
        aggregated_batch[key]["attention_mask_audio"] = audio_attention_mask

    subkey = 3 #Face
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        max_length = max(length_list)
        if max_length>150:
            # print("Here length was {}".format(length_list))
            max_length = 150
            aggregated_list = [i[:max_length] for i in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)

        face_attention_mask = torch.zeros((len(aggregated_list), max_length ))
        for data_idx, dur in enumerate(length_list):
            face_attention_mask[data_idx, :dur] = 1

    subkey = 4 #Face_image
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        max_length = max(length_list)
        if max_length>150:
            # print("Here length was {}".format(length_list))
            max_length = 150
            aggregated_list = [i[:max_length] for i in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)

        face_attention_mask = torch.zeros((len(aggregated_list), max_length ))
        for data_idx, dur in enumerate(length_list):
            face_attention_mask[data_idx, :dur] = 1




    # total_wav = []
    # total_vid = []
    # total_lab = []
    # total_dur = []
    # total_utt = []
    #
    # for cur_batch in batch:
    #     total_wav.append(torch.Tensor(cur_batch[0]))
    #     total_vid.append(torch.Tensor(cur_batch[1]))
    #     total_lab.append(cur_batch[2])
    #     total_dur.append(cur_batch[3])
    #
    #     total_utt.append(cur_batch[4])
    #     # print(total_utt)
    #
    # total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    # total_vid = nn.utils.rnn.pad_sequence(total_vid, batch_first=True)
    #
    # total_lab = torch.Tensor(total_lab)
    # max_dur = np.max(total_dur)
    # attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    # for data_idx, dur in enumerate(total_dur):
    #     attention_mask[data_idx, :dur] = 1
    ## compute mask

    return aggregated_batch

class TUH_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        train_dataset, valid_dataset, test_dataset = self._get_datasets()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0))-1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=num_cores,
                                                        shuffle=True,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        # generator=g,
                                                        # collate_fn=collate_fn_padd,
                                                        # worker_init_fn=seed_worker
                                                        )
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_cores,
                                                        # collate_fn=collate_fn_padd,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=num_cores,
                                                       # collate_fn=collate_fn_padd,
                                                       pin_memory=self.config.training_params.pin_memory)
    def _get_datasets(self):

        train_dataset = TUHDataset(config=self.config, mode="train")
        valid_dataset = TUHDataset(config=self.config, mode="val")
        test_dataset = TUHDataset(config=self.config, mode="test")

        return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    from easydict import EasyDict

    config = EasyDict()
    config.dataset = EasyDict()
    config.training_params = EasyDict()
    config.dataset.window_size = 4096
    config.dataset.fs = 200
    config.dataset.ratio_sz_nsz = 0.2
    config.dataset.stride = 4096
    config.dataset.data_path = "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5"
    config.training_params.len_sample = 4096*30
    config.training_params.fs = 200
    config.training_params.batch_size = 32
    config.training_params.test_batch_size = 32
    config.training_params.pin_memory = False
    config.training_params.num_workers = 6
    config.training_params.seed = 0


    dl = TUH_Dataloader(config)
    list_of_labels = []
    for i, batch in tqdm(enumerate(dl.valid_loader), total=len(dl.valid_loader)):
        # print(batch["data"]["raw"].shape)
        list_of_labels.append(batch["label"])
        if i == 1000:
            break
    print(torch.unique(torch.cat(list_of_labels).flatten(), return_counts=True))
    # for batch in dl.valid_loader:
    #     print(batch["data"]["raw"].shape)
    #     print(batch["label"].shape)
    #     break
    # for batch in dl.test_loader:
    #     print(batch["data"]["raw"].shape)
    #     print(batch["label"].shape)
    #     break