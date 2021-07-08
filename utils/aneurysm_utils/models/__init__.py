import torch
from addict import Dict
from torch import nn

from aneurysm_utils.models import (
    cnn_3d,
    densenet,
    linear_3d,
    pre_resnet,
    resnet,
    simple_cnn,
    wide_resnet,
    unet,
    simple_cnn,
    unet_3d,
    pointnet,
    attention_unet
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
        "resnet",
        "preresnet",
        "wideresnet",
        "densenet",
        "simpleCNN",
        "CNN3DSoftmax",
        "CNN3DMoboehle",
        "CNN3DTutorial",
        "ClassificationModel3D",
        "LinearModel3D",
        "MonaiUnet",
        "SimpleCNN3D",
        "SimpleCNN2D",
        "Unet3D",
        "Attention_Unet",
        "SegNet",

    ]

    if params.model_name == "resnet":
        if not params.resnet_shortcut:
            params.resnet_shortcut = "B"
        assert params.resnet_shortcut in ["A", "B"]

        if not params.model_depth:
            params.model_depth = 10
        assert params.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if params.model_depth == 10:
            model = resnet.resnet10(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 18:
            model = resnet.resnet18(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 34:
            model = resnet.resnet34(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 50:
            model = resnet.resnet50(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 101:
            model = resnet.resnet101(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )

        elif params.model_depth == 152:
            model = resnet.resnet152(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 200:
            model = resnet.resnet200(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
    elif params.model_name == "wideresnet":
        if not params.resnet_shortcut:
            params.resnet_shortcut = "B"
        assert params.resnet_shortcut in ["A", "B"]

        if not params.model_depth:
            params.model_depth = 50

        assert params.model_depth in [50]

        if params.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
    elif params.model_name == "preresnet":
        if not params.resnet_shortcut:
            params.resnet_shortcut = "B"
        assert params.resnet_shortcut in ["A", "B"]

        if not params.model_depth:
            params.model_depth = 18
        assert params.model_depth in [18, 34, 50, 101, 152, 200]

        if params.model_depth == 18:
            model = pre_resnet.resnet18(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 34:
            model = pre_resnet.resnet34(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 50:
            model = pre_resnet.resnet50(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 101:
            model = pre_resnet.resnet101(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 152:
            model = pre_resnet.resnet152(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 200:
            model = pre_resnet.resnet200(
                num_classes=params.num_classes,
                shortcut_type=params.resnet_shortcut,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
    elif params.model_name == "densenet":
        if not params.model_depth:
            params.model_depth = 121
        assert params.model_depth in [121, 169, 201, 264]

        if params.model_depth == 121:
            model = densenet.densenet121(
                num_classes=params.num_classes,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 169:
            model = densenet.densenet169(
                num_classes=params.num_classes,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 201:
            model = densenet.densenet201(
                num_classes=params.num_classes,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
        elif params.model_depth == 264:
            model = densenet.densenet264(
                num_classes=params.num_classes,
                sample_size1=params.sample_size[0],
                sample_size2=params.sample_size[1],
                sample_duration=params.sample_size[2],
                last_fc=last_fc,
            )
    elif params.model_name == "simpleCNN":
        # TODO: support last_fc, num_classes, and sample_size
        model = simple_cnn.CNN3D()
    elif params.model_name == "CNN3DSoftmax":
        if not params.dropout:
            params.dropout = 0
        if not params.dropout2:
            params.dropout2 = 0
        
        # TODO: support last_fc
        model = cnn_3d.CNN3DSoftmax(
            dropout=params.dropout,
            dropout2=params.dropout2,
            input_shape=params.sample_size,
            num_classes=params.num_classes,
        )
    elif params.model_name == "CNN3DTutorial":
        # TODO: support last_fc
        model = cnn_3d.CNN3DTutorial(
            input_shape=params.sample_size, num_classes=params.num_classes
        )
    elif params.model_name == "SimpleCNN2D":
        model = simple_cnn.SimpleCNN2D()
    elif params.model_name == "SimpleCNN3D":
        model = simple_cnn.SimpleCNN3D()

    elif params.model_name == "CNN3DMoboehle":
        # TODO: support last_fc
        model = cnn_3d.CNN3DMoboehle(
            input_shape=params.sample_size, num_classes=params.num_classes,
        )
    elif params.model_name == "MonaiUnet":
        model = unet.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=params.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
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
    elif params.model_name == "Attention_Unet":
        model = attention_unet.unet_grid_attention_3D(feature_scale=4, 
                                                      n_classes=2, 
                                                      is_deconv=True, 
                                                      in_channels=1,
                                                      nonlocal_mode='concatenation',  
                                                      attention_dsample=(2,2,2), is_batchnorm=True       
        )
    elif params.model_name == "LinearModel3D":
        # TODO: support last_fc,
        model = linear_3d.LinearModel3D(
            input_shape=params.sample_size, num_classes=params.num_classes
        )
    elif params.model_name == "ClassificationModel3D":
        # TODO: support last_fc
        if not params.dropout:
            params.dropout = 0
        if not params.dropout2:
            params.dropout2 = 0
        print(params.sample_size)
        model = cnn_3d.ClassificationModel3D(
            input_shape=params.sample_size,
            num_classes=params.num_classes,
            dropout=params.dropout,
            dropout2=params.dropout2,
        )

    print("Selected model: " + model.__class__.__name__)

    if params.use_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if params.pretrain_path:
        if not params.new_layer_names:
            params.new_layer_names = ["fc"]

        print("loading pretrained model {}".format(params.pretrain_path))
        pretrain = torch.load(params.pretrain_path)
        pretrain_dict = {
            k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()
        }

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in params.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        if not params.train_pretrain:
            for p in model.parameters():
                # freeze the pretrained parameters that not in new_parameters
                if id(p) not in new_parameters_id:
                    p.requires_grad = False

        base_parameters = list(
            filter(lambda p: id(p) not in new_parameters_id, model.parameters())
        )

        parameters = {
            "base_parameters": base_parameters,
            "new_parameters": new_parameters,
        }

        return model, parameters

    return model, model.parameters()
