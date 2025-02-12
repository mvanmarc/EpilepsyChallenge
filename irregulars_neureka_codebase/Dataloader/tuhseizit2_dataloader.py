
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

class TUHSeizIT2Dataset(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode = mode

        self.h5_file_sz2 = h5py.File(config.dataset.data_path["sz2"], 'r')
        self.h5_file_tuh = h5py.File(config.dataset.data_path["tuh"], 'r')

        #or load it from cumulated dict
        with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/seizit2_patient_dict.pkl", "rb") as f:
            self.patient_dict_sz2 = pickle.load(f)

        with open("/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/Dataloader/tuh_patient_dict.pkl", "rb") as f:
            self.patient_dict_tuh = pickle.load(f)

        self.patient_dict = {**self.patient_dict_sz2, **self.patient_dict_tuh}
        #merge the two patient_dicts and on each key add the "sz2" or "tuh" as value
        self.patient_dataset = {patient:"sz2" if patient in self.patient_dict_sz2 else "tuh" for patient in self.patient_dict.keys()}

        self._subselect_patients(mode=mode)

        self._get_cumulative_lens()

    def __len__(self):
        return int(list(self.cumulated_dict.keys())[-1])

    def _subselect_patients(self, mode):
        #get all patient_names from self.patient_dict
        patient_names = list(self.patient_dict.keys()) #scramble
        random.shuffle(patient_names)
        #split the patient_names into train, val, test
        train_patients, test_patients = train_test_split(patient_names, test_size=0.2, random_state=42)
        train_patients, val_patients = train_test_split(train_patients, test_size=0.2, random_state=42)

        #change patient_dict to only contain the selected patients
        if mode == "train":
            for patient in val_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "val":
            for patient in train_patients+test_patients:
                self.patient_dict.pop(patient)
        elif mode == "test":
            for patient in train_patients+val_patients:
                self.patient_dict.pop(patient)


    def _get_cumulative_lens(self):

        self.cumulated_dict = {}
        cum_idx = 0
        for patient in self.patient_dict.keys():
            for session in self.patient_dict[patient].keys():
                if self.patient_dataset[patient] == "sz2":
                    patient_len = self.patient_dict[patient][session][0] * self.config.dataset.fs
                    self.cumulated_dict[cum_idx] = {"patient": patient, "session": session, "recording": 0, "len": patient_len}
                    cum_idx += patient_len//self.config.dataset.window_size
                elif self.patient_dataset[patient] == "tuh":
                    for recording in self.patient_dict[patient][session].keys():
                        self.cumulated_dict[cum_idx] = {"patient": patient, "session": session, "recording": recording, "len": self.patient_dict[patient][session][recording]}
                        cum_idx += self.patient_dict[patient][session][recording]//self.config.dataset.window_size

    def _choose_patient_session_recording_len(self, idx):
        #find the cumulative idx that is right smaller than the idx
        for i, cum_idx in enumerate(list(self.cumulated_dict.keys())):
            if cum_idx > idx:
                break
        cum_idx = list(self.cumulated_dict.keys())[i-1]

        patient = self.cumulated_dict[cum_idx]["patient"]
        session = self.cumulated_dict[cum_idx]["session"]
        recording = self.cumulated_dict[cum_idx]["recording"]

        len_from = (idx-cum_idx)*self.config.dataset.window_size
        len_to = (idx-cum_idx+1)*self.config.dataset.window_size
        if len_to > self.cumulated_dict[cum_idx]["len"]:
            len_to = self.cumulated_dict[cum_idx]["len"]
        return {"patient": patient, "session": session, "recording": recording, "len_from": int(len_from), "len_to": int(len_to)}

    def _windowize(self, data, window_size, stride):
        data = np.array(data)
        data = data[:len(data) // window_size * window_size]
        data = data.reshape(-1, window_size)
        return data

    def _get_signals_seizit2(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        ch_names = [x.decode() for x in self.h5_file_sz2['sz2'][patient][session]['channel_names'][()]]
        sig = []
        for ch in ch_names:
            sig.append(self.h5_file_sz2['sz2'][patient][session][ch][len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def _get_label_seizit2(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        events = self.h5_file_sz2['sz2'][patient][session]['events'][()]

        #find the label of the corresponding segment
        if len(events) == 0:
            return torch.zeros(len_to-len_from)
        else:
            window_start = len_from
            window_end = len_to
            duration = self.h5_file_sz2['sz2'][patient][session]['duration'][()]
            total_label = torch.zeros(int(duration*self.config.dataset.fs))

            for event in events:
                event[0] = event[0]*self.config.dataset.fs
                event[1] = event[1]*self.config.dataset.fs
                total_label[int(event[0]):int(event[1])] = 1
            return total_label[window_start:window_end]

    def _get_signals_tuh(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        ch_names = [x.decode() for x in self.h5_file_tuh[patient][session][recording]['channel_names'][()]]
        sig = []
        for ch in ch_names:
            sig.append(self.h5_file_tuh[patient][session][recording][ch][len_from:len_to])

        # Put signals in a montage
        (signals_ds, _) = nedc.rereference(sig, [x.upper() for x in ch_names])
        data = np.array(signals_ds).T

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data-mean)/(std+1e-8)

    def _get_label_tuh(self, demographics):
        patient = demographics["patient"]
        session = demographics["session"]
        recording = demographics["recording"]
        len_from, len_to = demographics["len_from"], demographics["len_to"]

        events = self.h5_file_tuh[patient][session][recording]['events'][()]

        #find the label of the corresponding segment
        if len(events) == 0:
            return torch.zeros(len_to-len_from)
        else:
            window_start = len_from
            window_end = len_to
            duration = self.h5_file_tuh[patient][session][recording]['duration'][()]
            total_label = torch.zeros(int(duration*self.config.dataset.fs))

            for event in events:
                event[0] = event[0]*self.config.dataset.fs
                event[1] = event[1]*self.config.dataset.fs
                total_label[int(event[0]):int(event[1])] = 1
            return total_label[window_start:window_end]

    def __getitem__(self, idx):

        demographics = self._choose_patient_session_recording_len(idx)
        if self.patient_dataset[demographics["patient"]] == "sz2":
            signal = self._get_signals_seizit2(demographics)
            signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)
            label = self._get_label_seizit2(demographics)
        elif self.patient_dataset[demographics["patient"]] == "tuh":
            signal = self._get_signals_tuh(demographics)
            signal = self._windowize(signal, self.config.dataset.window_size, self.config.dataset.stride)
            label = self._get_label_tuh(demographics)

        return {"data":{"raw":signal},"label": label, "idx": idx}

class TUHSeizIT2_Dataloader():

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

        train_dataset = TUHSeizIT2Dataset(config=self.config, mode="train")
        valid_dataset = TUHSeizIT2Dataset(config=self.config, mode="val")
        test_dataset = TUHSeizIT2Dataset(config=self.config, mode="test")

        return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    from easydict import EasyDict

    config = EasyDict()
    config.dataset = EasyDict()
    config.training_params = EasyDict()
    config.dataset.window_size = 4096
    config.dataset.stride = 4096
    config.dataset.fs = 200
    config.dataset.data_path = {"sz2": "/esat/biomeddata/mbhaguba/seizeit2.h5",
                                "tuh": "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5"}
    config.training_params.len_sample = 4096*30
    config.training_params.fs = 200
    config.training_params.batch_size = 32
    config.training_params.test_batch_size = 32
    config.training_params.pin_memory = False
    config.training_params.num_workers = 6
    config.training_params.seed = 0


    dl = TUHSeizIT2_Dataloader(config)
    list_of_labels = []
    for i, batch in tqdm(enumerate(dl.train_loader), total=len(dl.train_loader)):
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