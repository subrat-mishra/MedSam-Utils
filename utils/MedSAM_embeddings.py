import os

import numpy as np

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import gdown

MODEL_ID = "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_"
MedSAM_CKPT_PATH = 'medsam_vit_b.pth'


def download_medsam_model():
    url = f'https://drive.google.com/uc?id={MODEL_ID}'
    if not os.path.exists(MedSAM_CKPT_PATH):
        gdown.download(url, MedSAM_CKPT_PATH, quiet=False)
    print(f'Downloaded to {MedSAM_CKPT_PATH}')


def getMedSamModel():
    download_medsam_model()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    med_sam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    med_sam_model = med_sam_model.to(device)
    med_sam_model.eval()
    return med_sam_model


MED_SAM_MODEL = getMedSamModel()


def get_medsam_embeddings(img_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if img_path.endswith('npy'):
        img_np = np.load(img_path)
    else:
        img_np = io.imread(img_path)

    if len(img_np.shape) == 3:
        img_np = img_np.mean(axis=0)
    assert len(img_np.shape) <= 3, "Image must not contain more than 3-channel"
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)

    H, W, _ = img_3c.shape
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    # convert the shape to (3, H, W)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = MED_SAM_MODEL.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    return image_embedding
