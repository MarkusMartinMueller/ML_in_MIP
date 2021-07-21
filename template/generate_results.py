import pandas as pd
import os
import aneurysm_utils
from aneurysm_utils.data_collection import load_nifti
import aneurysm_utils.preprocessing as preprocessing
from aneurysm_utils.models import get_model
from addict import Dict
import torch
import time
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import aneurysm_utils.postprocessing as postprocessing
from aneurysm_utils.evaluation import draw_mask_3d
from monai.transforms import Spacing


def generate_results():

    if "workspace" in os.getcwd():
        ROOT = "/workspace" # local 
    elif "/group/cake" in os.getcwd(): 
        ROOT = "/group/cake" # Jupyter Lab


    def prepare_case(x, device=None):
        """Prepare batch for training: pass to a device with options."""
        if device is None:
            device = "cpu"
        x = torch.tensor(np.expand_dims(x, axis=0).astype(np.float32), device=device)
        x = torch.unsqueeze(x, 0)
        return (
               x
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
    data_path = os.path.split(env.datasets_folder)[0]

    case_list = [
        "A101",
        "A144_M",
        "A104",
        "A146",
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
    model_name = "2021-07-12-17-17-25_mask-pytorch-attention-unet"
    weights_name = "Attention_Unet_Attention_Unet_1560.pt"

    PATH = os.path.join(env.experiments_folder, f"{model_name}/{weights_name}")
    result_path = os.path.join(experiments_folder, model_name)
    
    params = {
        "use_cuda": True,
        "model_name": base_model,
        "feature_scale":2,
        "resample_voxel_dim": (1.2, 1.2, 1.2),
        "patch_size": 64,
        "num_imgs": 1,
        "resample": True,
        "dbscan": True, 
        "eps": 2, 
        "min_samples": 5,
        "size_smallest": 40,
        "size_biggest": 20000,
        "remove_border_candidates": True
    }
    params = Dict(params)
    model, model_params = get_model(params)


    device =  torch.device('cuda') if params.use_cuda else  torch.device('cpu')

    model.load_state_dict(torch.load(PATH, map_location=device))

    times = []
    masks = []

    for case in case_list:
        print(f"\n---------------------- Processing {case} ----------")
        start_time = time.time()
        
        # Load image
        path_orig = "/data/test/" + case + "_orig.nii.gz"
        nifti = nib.load(path_orig)
        struct_arr = nifti.get_fdata().astype("<f4")
       
        img_shape = struct_arr.shape
        print(img_shape)
        if img_shape != (256, 256, 220):
            struct_arr = struct_arr[:256, :256, :220]
        
        if params.resample_voxel_dim is not None:
            struct_arr = np.expand_dims(struct_arr, axis=0)
            spacing = Spacing(pixdim=params.resample_voxel_dim )
            struct_arr = spacing(struct_arr, nifti.affine)[0]
            struct_arr = np.squeeze(struct_arr, axis=0)
            
         
        # Preprocess
        inputs = preprocessing.preprocess(env, [struct_arr], params)

        # Predict Segmention Mask
        preds = []
        for x in inputs:
            x = prepare_case(x, device)
            pred = model(x)
            pred = torch.squeeze(pred.max(1)[1], 0).cpu().detach().numpy()
            preds.append(pred)
        
        # Post Process
        #params["resample_voxel_dim"] = params["resample_voxel_dim"][0]
        processed = postprocessing.postprocess(env, preds, params) [0]
        
        

        # Save time and results
        elapsed = time.time()- start_time
        times.append(elapsed)
        print(f"{case} finsihed in {elapsed} seconds")
        
        preds_in_ori_shape = np.zeros(img_shape)
        print(processed.shape)
        preds_in_ori_shape[:256, :256, :220] = processed
        masks.append(preds_in_ori_shape)
        print(preds_in_ori_shape.shape)
        
        # TODO: print image
        draw_mask_3d(preds_in_ori_shape)
        plt.savefig(os.path.join(result_path, f"{case}.png"))
   


    # Generate JSON   #TODO change path
    postprocessing.create_task_one_json(
        masks,
        case_list,
        processing_times=times,
        path=os.path.join(result_path, "reference.json"),
        path_datasets=os.path.join(data_path, "test"),
    )
    
    postprocessing.create_nifits(
        masks,
        case_list,
        path_cada=result_path,
        path_datasets=os.path.join(data_path, "test")
    )
    
if __name__ == "__main__":
    generate_results()
