import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from addict import Dict
import os
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, ConfusionMatrix, Loss, Recall, IoU, DiceCoefficient
from torch.utils.data.dataloader import DataLoader
from pytorch3dunet.unet3d.losses import BCEDiceLoss
#from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete, Compose
from monai.handlers import MeanDice
from torch_geometric.data import DataLoader as DataLoaderGeometric
import aneurysm_utils
from aneurysm_utils.environment import Experiment
from aneurysm_utils.models import get_model
from aneurysm_utils.utils import pytorch_utils
from aneurysm_utils import evaluation
from aneurysm_utils.utils.ignite_utils import prepare_batch, DiceCELoss
from aneurysm_utils.utils.point_cloud_utils import extend_point_cloud
import time
import psutil

# -------------------------- Train model ---------------------------------
def train_pytorch_model(exp: Experiment, params, artifacts):

    # Get artifacts for the experiment run
    params = Dict(params)
    if not params.process:
        mri_imgs_train,labels_train,mri_imgs_val, labels_val=None
    else:
        mri_imgs_train, labels_train = artifacts["train_data"]
        mri_imgs_val, labels_val = artifacts["val_data"]
    mri_imgs_test, labels_test = None, None
    if "test_data" in artifacts:
        mri_imgs_test, labels_test = artifacts["test_data"]

    if not params.process:
        params.process = False
    # check params and set basic values
    if not params.learning_rate:
        params.learning_rate = 0.001

    if not params.optimizer_momentum:
        params.optimizer_momentum = 0.9

    if not params.weight_decay:
        params.weight_decay = 0

    if not params.epochs:
        params.epochs = 50

    if not params.optimizer:
        params.optimizer = "SGD"

    if not params.batch_size:
        params.batch_size = 10

    if not params.criterion:
        params.criterion = "CrossEntropyLoss"

    if not params.num_classes:
        try:
            params.num_classes = len(
                np.unique(np.concatenate(labels_train).astype(int))
            )
        except TypeError:
            params.num_classes = len(set(labels_train))
        print("Number of Classes", params.num_classes)

    if not params.num_threads:
        params.num_threads = None

    if params.shuffle_train_set is None:
        params.shuffle_train_set = True

    if params.use_cuda is not False and params.use_cuda is not True:
        # params.use_cuda can be empty dict or None
        params.use_cuda = torch.cuda.is_available()

    if not params.sample_size:
        # auto detect sample size from first training image
        params.sample_size = mri_imgs_train[0].shape

    if not params.dropout:
        params.dropout = 0.0

    if not params.dropout2:
        params.dropout2 = 0.0

    if params.seed:
        torch.manual_seed(params.seed)

    if params.training_size:
        mri_imgs_train = mri_imgs_train[: params.training_size]

    if params.val_size:
        mri_imgs_val = mri_imgs_val[: params.val_size]

    if params.prediction in ["mask", "vessel"]:
        params.segmentation = True
    else:
        params.segmentation = None

    # Get Model Architecture
    model, model_params = get_model(params)
    exp.artifacts["model"] = model

    # print('number of trainable parameters =', count_parameters(model))

    # Log all params to comet
    exp.comet_exp.log_parameters(params.to_dict())

    # Initialize data loaders
    if params.model_name != "SegNet":
        train_dataset = pytorch_utils.PytorchDataset(
            mri_imgs_train,
            labels_train,
            dtype=np.float64,
            segmentation=params.segmentation,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=params.shuffle_train_set,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )
        validation_dataset = pytorch_utils.PytorchDataset(
            mri_imgs_val, labels_val, dtype=np.float64, segmentation=params.segmentation
        )
        test_dataset = pytorch_utils.PytorchDataset(
            mri_imgs_test,
            labels_test,
            dtype=np.float64,
            segmentation=params.segmentation,
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=params.batch_size,  # TODO: use fixed batch size of 5
            shuffle=False,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # TODO: use fixed batch size of 5
            shuffle=False,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )
    else:
        datasets_folder = exp._env.project_folder
        train_dataset = pytorch_utils.PyTorchGeometricDataset(
            mri_images=mri_imgs_train,
            labels=labels_train,
            root=datasets_folder,
            split="train",
            force_processing=params.process
        )
        val_dataset = pytorch_utils.PyTorchGeometricDataset(
            mri_images=mri_imgs_val,
            labels=labels_val,
            root=datasets_folder,
            split="val",
            force_processing=params.process
        )
        test_dataset = pytorch_utils.PyTorchGeometricDataset(
            mri_images=mri_imgs_test,
            labels=labels_test,
            root=datasets_folder,
            split="test",
            force_processing=params.process
        )
        train_loader = DataLoaderGeometric(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=params.shuffle_train_set,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )
        val_loader = DataLoaderGeometric(
            val_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )
        test_loader = DataLoaderGeometric(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=params.num_threads if params.num_threads else 0,
            pin_memory=params.use_cuda,
        )

    exp.log.info("Train dataset loaded. Length: " + str(len(train_loader.dataset)))
    exp.log.info("Validation dataset loaded. Length: " + str(len(val_loader.dataset)))

    device = torch.device("cuda" if params.use_cuda else "cpu")
    if params.device:
        device = params.device

    if params.criterion == "CrossEntropyLoss":
        if params.criterion_weights:
            weights = params.criterion_weights
            if isinstance(weights, int) or isinstance(weights, float):
                criterion = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor([1.0, weights]).to(device)
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(weights).to(device)
                )
        else:
            criterion = nn.CrossEntropyLoss()

    elif params.criterion == "BCEWithLogitsLoss":
        if params.criterion_weights:
            criterion = nn.BCEWithLogitsLoss(
                weight=torch.FloatTensor(params.criterion_weights).to(device)
            )
        else:
            criterion = nn.BCEWithLogitsLoss()

    elif params.criterion == "BCELoss":
        if params.criterion_weights:
            criterion = nn.BCELoss(
                weight=torch.FloatTensor(params.criterion_weights).to(device)
            )
        else:
            criterion = nn.BCELoss()

    elif params.criterion == "MultiLabelMarginLoss":
        criterion = nn.MultiLabelMarginLoss()

    elif params.criterion == "BCEDiceLoss":
        # alpha = loss_config.get('alphs', 1.)
        # beta = loss_config.get('beta', 1.)
        criterion = BCEDiceLoss(1, 1)
    
    elif params.criterion == "DiceCELoss":
        if params.criterion_weights:
            weights = params.criterion_weights
            if isinstance(weights, int) or isinstance(weights, float):
                criterion = DiceCELoss(
                    softmax=True, 
                    ce_weight=torch.FloatTensor([1.0, weights]).to(device),
                    to_onehot_y=True,
                )
            else:
                criterion = DiceCELoss(
                    softmax=True, 
                    ce_weight=torch.FloatTensor(weights).to(device),
                    to_onehot_y=True
                )
    elif params.criterion == "DiceLoss":
        
        criterion = DiceLoss()
        

    else:
        raise ValueError("No criterion given")

    if params.use_cuda:
       
        criterion = criterion.to(device)

    if params.pretrain_path:
        model_params = [
            {
                "params": filter(
                    lambda p: p.requires_grad, model_params["base_parameters"]
                ),
                "lr": params.learning_rate,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad, model_params["new_parameters"]
                ),
                "lr": params.learning_rate * 10,
            },
        ]
    else:
        pass
        # model_params = [{'params': filter(lambda p: p.requires_grad, model_params), 'lr': params.learning_rate}]

    optimizer: torch.optim.Optimizer

    if params.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model_params,
            lr=params.learning_rate,
            momentum=params.optimizer_momentum,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model_params, lr=params.learning_rate, weight_decay=params.weight_decay
        )
    else:
        raise ValueError("Unknown optimizer: " + str(params.optimizer))
        print(params.segmentation)
    if params.segmentation:
        output_transform = lambda x: (
            x[0],
            torch.squeeze(x[1]),  # torch.squeeze(x[1], 1),
        )
    else:
        output_transform = None

    # trainer and evaluator
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "accuracy": Accuracy(),  # output_transform=output_transform, is_multilabel=True
            "loss": Loss(criterion),
            "recall": Recall(),
            "confusion_matrix": ConfusionMatrix(
                params.num_classes, output_transform=output_transform
            ),
        },
        device=device,
        prepare_batch=prepare_batch,
    )

    if not params.scheduler:
        # Dont use any scheduler
        pass
    elif params.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.5, patience=2
        )

        @evaluator.on(Events.COMPLETED)
        def reduce_lr(engine):
            # Execute after every validation
            # engine is evaluator
            # engine.metrics is a dict with metrics, e.g. {"loss": val_loss_value, "acc": val_acc_value}
            scheduler.step(evaluator.state.metrics["loss"])
            exp.log.info("Learning rate: " + str(optimizer.param_groups[0]["lr"]))

    else:
        raise ValueError("Unknown scheduler: " + str(params.scheduler))

    def score_function(engine):
        accuracy = evaluator.state.metrics["loss"]
        print(accuracy)
        return -accuracy

    if params.es_patience:
        # Only activate early stopping if patience is provided
        handler = EarlyStopping(
            patience=params.es_patience, score_function=score_function, trainer=trainer
        )
        evaluator.add_event_handler(Events.COMPLETED, handler)

    def empty_cuda_cache(engine):
        if params.use_cuda:
            torch.cuda.empty_cache()
            import gc

            gc.collect()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        with exp.comet_exp.train():
            exp.comet_exp.log_current_epoch(engine.state.epoch)
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            acc = metrics["accuracy"]
            avg_nll = metrics["loss"]
            spec = float(metrics["recall"].data[0])
            sen = float(metrics["recall"].data[1])
            bal_acc = (spec + sen) / 2
            exp.log.info(
                "Training Results - Epoch: {} Bal Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, bal_acc, avg_nll
                )
            )
            exp.comet_exp.log_metrics(
                {
                    "accuracy": acc,
                    "bal_acc": bal_acc,
                    "spec": spec,
                    "sen": sen,
                    "avg_loss": avg_nll,
                    "lr": optimizer.param_groups[0]["lr"]
                },
                epoch=engine.state.epoch,
            )
            empty_cuda_cache(engine)
            print('RAM memory % used:', psutil.virtual_memory()[2])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        with exp.comet_exp.validate():
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            acc = metrics["accuracy"]
            avg_nll = metrics["loss"]
            spec = float(metrics["recall"].data[0])
            sen = float(metrics["recall"].data[1])
            bal_acc = (spec + sen) / 2
            conf_matrix = metrics["confusion_matrix"]
            exp.log.info(
                "Validation Results - Epoch: {} Bal Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, bal_acc, avg_nll
                )
            )
            exp.comet_exp.log_metrics(
                {"accuracy": acc, "bal_acc": bal_acc, "spec": spec, "sen": sen},
                epoch=engine.state.epoch,
            )
            exp.comet_exp.log_confusion_matrix(
                matrix=conf_matrix.tolist(),
                step=engine.state.epoch,
                file_name="confusion-matrix-" + str(engine.state.epoch) + ".json",
            )
            empty_cuda_cache(engine)
            print('RAM memory % used:', psutil.virtual_memory()[2])

    # create model files
    if params.save_models:
        checkpointer = ModelCheckpoint(
            exp.output_path,
            params.model_name,
            save_interval=1,
            n_saved=5,
            create_dir=True,
            save_as_state_dict=True,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpointer, {params.model_name: model}
        )

    trainer.run(train_loader, max_epochs=params.epochs)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    if mri_imgs_test is not None:
        with exp.comet_exp.test():
            pred_classes, pred_scores = zip(
                *pytorch_utils.predict(model, test_loader, cuda=params.use_cuda, device=device)
            )
            print('RAM memory % used:', psutil.virtual_memory()[2])
            if params.model_name == "SegNet":
                pred_classes, pred_scores = extend_point_cloud(
                    pred_classes, pred_scores, test_dataset, labels_test
                )
            
            exp.comet_exp.log_metrics(
                evaluation.evaluate_model(
                    labels_test,
                    pred_classes,
                    params.segmentation,
                )
            )
            print('RAM memory % used:', psutil.virtual_memory()[2])

 
    exp.comet_exp.end()
    
