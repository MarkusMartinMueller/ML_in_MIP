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
from monai.transforms import AsDiscrete, Compose
from monai.handlers import MeanDice
from torch_geometric.data import DataLoader as DataLoaderGeometric
import aneurysm_utils
from aneurysm_utils.environment import Experiment
from aneurysm_utils.models import get_model
from aneurysm_utils.utils import pytorch_utils
from aneurysm_utils import evaluation
from aneurysm_utils.utils.ignite_utils import prepare_batch
from aneurysm_utils.utils.point_cloud_utils import extend_point_cloud

# -------------------------- Train model ---------------------------------

## -------------------------- Sklearn ------------------------------
def train_sklearn_model(exp, params, artifacts):

    mri_imgs_train, labels_train = artifacts["train_data"]
    mri_imgs_val, labels_val = artifacts["val_data"]
    mri_imgs_test, labels_test = None, None
    if "test_data" in artifacts:
        mri_imgs_test, labels_test = artifacts["test_data"]

    params = Dict(params)

    if not params.seed:
        params.seed = None

    if not params.n_jobs:
        params.n_jobs = None

    if not params.class_weight:
        params.class_weight = None

    if params.training_size:
        mri_imgs_train = mri_imgs_train[: params.training_size]

    if params.val_size:
        mri_imgs_val = mri_imgs_val[: params.val_size]

    assert params.model_name in [
        "SVC",
        "LinearSVC",
        "HyperoptEstimator",
        "KNeighborsClassifier",
        "NuSVC",
        "LogisticRegression",
        "GridSearchSVC",
        "SGDClassifier",
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "PassiveAggressiveClassifier",
        "GaussianProcessClassifier",
        "ComplementNB",
        "DBSCAN",
    ]

    if params.model_name == "SVC":
        if not params.gamma:
            params.gamma = "auto"

        if not params.C:
            params.C = 1.0

        if not params.kernel:
            params.kernel = "rbf"

        from sklearn.svm import SVC

        model = SVC(
            gamma=params.gamma,
            C=params.C,
            kernel=params.kernel,
            random_state=params.seed,
            verbose=True,
            class_weight=params.class_weight,
        )
    if params.model_name == "GridSearchSVC":
        if not params.gamma:
            params.gamma = "auto"

        from sklearn.svm import SVC

        svc = SVC(gamma=params.gamma, class_weight=params.class_weight, verbose=2)

        from sklearn.model_selection import GridSearchCV

        parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
        model = GridSearchCV(svc, parameters, n_jobs=params.n_jobs, verbose=2)
    elif params.model_name == "LinearSVC":

        if not params.C:
            params.C = 1.0

        from sklearn.svm import LinearSVC

        model = LinearSVC(
            C=params.C, random_state=params.seed, class_weight=params.class_weight
        )
    elif params.model_name == "GaussianProcessClassifier":
        from sklearn.gaussian_process import GaussianProcessClassifier

        model = GaussianProcessClassifier(
            random_state=params.seed, n_jobs=params.n_jobs
        )
    elif params.model_name == "DBSCAN":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=0.3, min_samples=100)

    elif params.model_name == "ComplementNB":
        from sklearn.naive_bayes import ComplementNB

        model = ComplementNB()
    elif params.model_name == "SGDClassifier":

        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier(
            random_state=params.seed,
            verbose=2,
            class_weight=params.class_weight,
            n_jobs=params.n_jobs,
        )
    if params.model_name == "NuSVC":
        if not params.gamma:
            params.gamma = "auto"

        from sklearn.svm import NuSVC

        model = NuSVC(
            gamma=params.gamma,
            random_state=params.seed,
            verbose=True,
            class_weight=params.class_weight,
        )
    elif params.model_name == "KNeighborsClassifier":
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(n_jobs=params.n_jobs)
    elif params.model_name == "LogisticRegression":
        if not params.C:
            params.C = 1.0

        if not params.lr_solver:
            params.lr_solver = "sag"

        if not params.lr_tol:
            params.lr_tol = 0.0001

        if not params.lr_epochs:
            params.lr_epochs = 1

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            verbose=2,
            multi_class="auto",
            solver=params.lr_solver,
            C=params.C,
            penalty="l2",  # sag and saga do not support l1
            tol=params.lr_tol,
            dual=False,  # sag and saga do not support dual
            max_iter=params.lr_epochs,
            random_state=params.seed,
            n_jobs=params.n_jobs,
        )
    elif params.model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        # Create the estimator object
        model = RandomForestClassifier(
            random_state=params.seed,
            n_jobs=params.n_jobs,
            class_weight=params.class_weight,
            verbose=2,
        )
    elif params.model_name == "DecisionTreeClassifier":
        from sklearn.tree import DecisionTreeClassifier

        # Create the estimator object
        model = DecisionTreeClassifier(
            random_state=params.seed, class_weight=params.class_weight
        )
    elif params.model_name == "PassiveAggressiveClassifier":
        if not params.loss:
            params.loss = "hinge"

        if not params.C:
            params.C = 1.0

        from sklearn.linear_model import PassiveAggressiveClassifier

        # Create the estimator object
        model = PassiveAggressiveClassifier(
            random_state=params.seed,
            loss=params.loss,
            class_weight=params.class_weight,
            C=params.C,
            n_iter_no_change=10,
            verbose=2,
            n_jobs=params.n_jobs,
        )
    elif params.model_name == "HyperoptEstimator":
        from hpsklearn import (
            HyperoptEstimator,
            any_sparse_classifier,
            any_classifier,
            any_preprocessing,
            svc,
            random_forest,
        )
        from hyperopt import tpe

        # Create the estimator object
        model = HyperoptEstimator(
            seed=params.seed,
            # algo=tpe.suggest,
            verbose=True,
            max_evals=50,
            refit=False,
            # classifier=any_sparse_classifier("my_clf"),
            classifier=any_classifier("my_clf"),
            preprocessing=any_preprocessing("my_pre"),
            # classifier=any_sparse_classifier("my_clf"),
            # classifier=random_forest("my-model", n_jobs=params.n_jobs, verbose=True)
            # classifier=any_sparse_classifier("my_clf"),
            # trial_timeout=5000,
        )

    # Log all params to comet
    exp.comet_exp.log_parameters(params.to_dict())

    model.fit([mri_img.flatten() for mri_img in mri_imgs_train], labels_train)

    # model = estim.best_model()
    exp.artifacts["model"] = model

    exp.log.info("Training finished.")
    from sklearn.metrics import accuracy_score

    with exp.comet_exp.validate():
        y_pred = model.predict([mri_img.flatten() for mri_img in mri_imgs_val])

        accuracy = accuracy_score(labels_val, y_pred)
        exp.log.info("Validation Accuracy: " + str(accuracy))

        exp.comet_exp.log_metrics(evaluation.evaluate_model(labels_val, y_pred))

        from sklearn.metrics import confusion_matrix

        exp.comet_exp.log_confusion_matrix(
            matrix=confusion_matrix(labels_val, y_pred).tolist(),
            file_name="confusion-matrix-validate.json",
        )

    if mri_imgs_test is not None:
        # Only run test of
        with exp.comet_exp.test():
            y_pred = model.predict([mri_img.flatten() for mri_img in mri_imgs_test])

            accuracy = accuracy_score(labels_test, y_pred)
            exp.log.info("Test Accuracy: " + str(accuracy))

            exp.comet_exp.log_metrics(evaluation.evaluate_model(labels_test, y_pred))

            from sklearn.metrics import confusion_matrix

            exp.comet_exp.log_confusion_matrix(
                matrix=confusion_matrix(labels_test, y_pred).tolist(),
                file_name="confusion-matrix-test.json",
            )

    if params.save_models:
        from joblib import dump, load

        dump(model, os.path.join(exp.output_path, params.model_name + ".joblib"))


## -------------------------- Pytorch ------------------------------


def train_pytorch_model(exp: Experiment, params, artifacts):
    # new SummaryWriter for new experiment
    # Tensorflow logger
    # summary_writer = SummaryWriter(log_dir=run_dir)

    # Get artifacts for the experiment run
    mri_imgs_train, labels_train = artifacts["train_data"]
    mri_imgs_val, labels_val = artifacts["val_data"]
    mri_imgs_test, labels_test = None, None
    if "test_data" in artifacts:
        mri_imgs_test, labels_test = artifacts["test_data"]

    params = Dict(params)

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
    
    if not params.start_radius:
        params.start_radius=0.2
    
    if not params.sample_rates:
        params.sample_rates=[0.2,0.25]

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

    # TODO: use augmentation
    # assert params.sampler is not None, "Sampler is not implemented"

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
        )
        val_dataset = pytorch_utils.PyTorchGeometricDataset(
            mri_images=mri_imgs_val,
            labels=labels_val,
            root=datasets_folder,
            split="val",
        )
        test_dataset = pytorch_utils.PyTorchGeometricDataset(
            mri_images=mri_imgs_test,
            labels=labels_test,
            root=datasets_folder,
            split="test",
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

    # train_dataset.print_image()

    exp.log.info("Train dataset loaded. Length: " + str(len(train_loader.dataset)))
    exp.log.info("Validation dataset loaded. Length: " + str(len(val_loader.dataset)))

    device = torch.device("cuda" if params.use_cuda else "cpu")

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

    else:
        raise ValueError("No criterion given")

    if params.use_cuda:
        # TODO: required?
        criterion = criterion.cuda()

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
        # post_pred = Compose(
        #  [AsDiscrete(threshold_values=True)])
        # output_transform=lambda x, y, y_pred: (y_pred, y)
        # val_metrics = {
        #   "Mean_Dice": MeanDice(),
        #  "loss": Loss(criterion),
        #  }
        output_transform = lambda x: (
            x[0],
            torch.squeeze(x[1]),  # torch.squeeze(x[1], 1),
        )  # (torch.flatten(x[0], start_dim=1), torch.flatten(x[1], start_dim=1))
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

    # Add learning rate scheduler
    # from ignite.contrib.handlers import LRScheduler
    # scheduler = LRScheduler(lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98))
    # scheduler = LRScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=10))

    # Attach to the trainer
    # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    if not params.scheduler:
        # Dont use any scheduler
        pass
    elif params.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.5, patience=10
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
        return accuracy

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
                },
                epoch=engine.state.epoch,
            )
            """
            summary_writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
            summary_writer.add_scalar("training/acc", acc, engine.state.epoch)
            summary_writer.add_scalar("training/bal_acc_with_ignite", bal_acc_with_ignite, engine.state.epoch)
            summary_writer.add_scalar("training/bal_acc", bal_acc, engine.state.epoch)
            summary_writer.add_scalar("training/spec", spec, engine.state.epoch)
            summary_writer.add_scalar("training/sen", sen, engine.state.epoch)
            """
            empty_cuda_cache(engine)

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
            """
            summary_writer.add_scalar("valdation/acc", acc, engine.state.epoch)
            summary_writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
            summary_writer.add_scalar("valdation/bal_acc_with_ignite", bal_acc_with_ignite, engine.state.epoch)
            summary_writer.add_scalar("valdation/bal_acc", bal_acc, engine.state.epoch)
            summary_writer.add_scalar("validation/spec", spec, engine.state.epoch)
            summary_writer.add_scalar("validation/sen", sen, engine.state.epoch)
            """
            empty_cuda_cache(engine)

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

    if mri_imgs_test is not None:
        with exp.comet_exp.test():
            pred_classes, pred_scores = zip(
                *pytorch_utils.predict(model, test_loader, cuda=params.use_cuda)
            )
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

    # TODO not really needed?
    exp.comet_exp.end()
    # TODO: run evaluation
