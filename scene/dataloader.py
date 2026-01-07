from torch.utils.data import Dataset, DataLoader
from utils.camera_utils import cameraList_from_camInfos
from scene import Scene


class CameraDataset(Dataset):
    def __init__(self, scene):
        self.camera_list = scene.getTrainCameras().copy()

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        camera = self.camera_list[idx]

        return {'idx': idx,  # data index
                'image': camera.original_image}