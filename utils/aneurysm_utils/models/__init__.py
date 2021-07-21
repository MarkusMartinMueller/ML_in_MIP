import torch
from addict import Dict
from torch import nn

from aneurysm_utils.models import (
    simple_cnn,
    unet_3d,
    pointnet,
    attention_unet,
    unet_3d_oktay
)


def get_model(params: Dict):
    # mode
    # TODO
    # sample_size_x
    # sample_size_y
    # sample_size_z
    # n_classes
    # model_name
    # model_depth
    # no_cuda
    # pretrain_path
    # resnet_shortcut
    # new_layer_names
    # train_pretrain

    # Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).
    if not params.mode or params.mode == "score":
        last_fc = True
    elif params.mode == "feature":
        last_fc = False
    else:
        raise ValueError("Unknown mode.")

    assert params.model_name in [
        "SimpleCNN3D",
        "SimpleCNN2D",
        "Unet3D",
        "Attention_Unet",
        "SegNet",
        "Unet3D_Oktay",

    ]

    
    if params.model_name == "SimpleCNN2D":
        model = simple_cnn.SimpleCNN2D()
    elif params.model_name == "SimpleCNN3D":
        model = simple_cnn.SimpleCNN3D()

    elif params.model_name == "SegNet":
        model = pointnet.SegNet(
            num_classes= params.num_classes,
            dropout = params.dropout,
            start_radius=params.start_radius,
            #sample_rate1= params.sample_rate1,
            #sample_rate2= params.sample_rate2
            
        )
    elif params.model_name == "Unet3D":
        model = unet_3d.Unet_3D(
            in_channels= 1,
            out_channels= params.num_classes, 
            filters = [64,128,256],
            )
    elif params.model_name == "Unet3D_Oktay":
        model = unet_3d_oktay.unet_3D(feature_scale=params.feature_scale, n_classes=2, is_deconv=True, in_channels=1, is_batchnorm=True)
    
    
    elif params.model_name == "Attention_Unet":
        model = attention_unet.unet_grid_attention_3D(feature_scale=4, 
                                                      n_classes=2, 
                                                      is_deconv=True, 
                                                      in_channels=1,
                                                      nonlocal_mode='concatenation',  
                                                      attention_dsample=(2,2,2), is_batchnorm=True       
        )
    

    print("Selected model: " + model.__class__.__name__)

    if params.use_cuda:
        if params.device:
            model = model.to(params.device)
        else:
            model = model.cuda()
            #model = nn.DataParallel(model, device_ids=None)
            #net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

 

    return model, model.parameters()
