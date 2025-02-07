
from irregulars_neureka_codebase.Dataloader.tuh_dataloader import TUH_Dataloader
from tqdm import tqdm
from easydict import EasyDict
import einops
from irregulars_neureka_codebase.pytorch_models.neureka_models import NeurekaNet

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
config.model = EasyDict()
config.model.window_size = 4096
config.model.n_channels = 18
config.model.n_filters = 8
config.model.pre_dir_raw = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_raw.pth'
config.model.pre_dir_wiener = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_wiener.pth'
config.model.pre_dir_iclabel = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_iclabel.pth'
config.model.pre_dir_lstm = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_lstm.pth'

config.model.save_preds = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/prediction_test_raw.h5'

dataloaders = TUH_Dataloader(config)

pytorch_net = NeurekaNet(config)
pytorch_net.cuda()
pytorch_net.eval()


agg_features, labels = [], []

for i, batch in tqdm(enumerate(dataloaders.valid_loader), total=len(dataloaders.valid_loader)):

    data = batch['data']['raw']
    label = batch['label']
    data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1).cuda().float()

    pred = pytorch_net(data)

    print(pred.shape)

    break
