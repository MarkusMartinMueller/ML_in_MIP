import pandas as pd
import os
import aneurysm_utils
from aneurysm_utils.data_collection import load_nifti
from aneurysm_utils.models import get_model
from addict import Dict
import torch
import time

if "workspace" in os.getcwd():
    ROOT = "/workspace" # local 
elif "/group/cake" in os.getcwd(): 
    ROOT = "/group/cake" # Jupyter Lab


def prepare_case(x, device=None):
    """Prepare batch for training: pass to a device with options."""
    if device is None:
        device = "cpu"
    return (
           np.expand_dims(x, axis=0).astype(np.float32).to(device)
        )
    """
    except ValueError:
        batch_tensor, pos, ptr, x, y = batch
        batch_tensor = convert_tensor(
            batch_tensor, device=device, non_blocking=non_blocking
        )[1]
        pos = convert_tensor(pos, device=device, non_blocking=non_blocking)[1]
        x = convert_tensor(x, device=device, non_blocking=non_blocking)[1]
        y = convert_tensor(y, device=device, non_blocking=non_blocking)[1]
        
        return {"batch": batch_tensor, "pos": pos, "x": x}, y
    """

env = aneurysm_utils.Environment(project="our-git-project", root_folder=ROOT)
env.cached_data["comet_key"] = "EGrR4luSis87yhHbs2rEaqAWs" 
env.print_info()

experiments_folder = env.experiments_folder

case_list = [
    "A101",
    "A144_M",
    "A104",
    "A146"
    "A145",
    "A107",
    "A065",
    "A150",
    "A139",
    "A149", 
    "A075",
    "A022",
    "A147",
    "A030",
    "A109", 
    "A144_L",
    "A131",
    "A141",
    "A036",
    "A148",
    "A151",
    "A099"]


cases = []
for case in case_list:
    case_dict = {}
    case_dict["Path Orig"] = "/data/test/" + case + "_orig.nii.gz"
    case_dict["Case"] = case
    cases.append(case_dict)

df = pd.DataFrame(cases)

base_model = "Attention_Unet"
model_name = "2021-07-12-15-27-17_mask-pytorch-attention-unet"
weights_name = "Attention_Unet_Attention_Unet_640.pt"

PATH = os.path.join(env.experiments_folder, f"{model_name}/{weights_name}")

params = {
    "use_cuda": True,
    "model_name": base_model,
}
params = Dict(params)
model, model_params = get_model(params)


device = torch.device('cuda')

model.load_state_dict(torch.load(PATH, map_location=device))

times = []
masks = []

for case in case_list:
    start_time = time.time()
    # Load image
    path_orig = "/data/test/" + case + "_orig.nii.gz"
    ori_struct = load_nifti(path_orig, resample_dim=params.resample_voxel_dim,resample_size=params.resample_size,order=params.order)
    print(ori_struct.shape)
    # Preprocess
    ori_struct = preprocessing.preprocess(env, [ori_struct], params)[0]

    # Predict Segmention Mask
    x = prepare_case(ori_struct, device)
    pred = model(x)

    # Post Process


    # Save time and results
    elapsed = time.time()- start_time
    times.append(elapsed)
    print(f"{case} finsihed in {elapsed} seconds")


# Generate JSON
