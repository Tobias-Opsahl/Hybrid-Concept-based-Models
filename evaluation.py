import os
import pickle
import torch
import torch.nn as nn
from train import train_simple, train_cbm
from shapes.datasets_shapes import load_data_shapes, make_subset_shapes
from plotting import plot_training_histories, plot_subset_test_accuracies
from utils import get_hyperparameters, load_models_shapes, add_histories, seed_everything
from constants import MODEL_STRINGS, COLORS, MAX_EPOCHS


def evaluate_on_test_set(model, test_loader, device=None, non_blocking=False):
    """
    Evaluates a model on the test set and returns the test accuracy.

    Args:
        model (model): Pytorch pretrained model
        test_loader (_type_): _description_
        device (_type_, optional): _description_. Defaults to None.
        non_blocking (bool, optional): _description_. Defaults to False.
        test_loader (dataloader): The test-dataloader
        device (str): Use "cpu" for cpu training and "cuda:0" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.

    Returns:
        float: The test accuracy
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=non_blocking)
    model.eval()
    test_correct = 0
    for input, labels, attr_labels, paths in test_loader:
        input = input.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        outputs = model(input)
        if isinstance(outputs, tuple):  # concept model, returns (outputs, attributes)
            outputs = outputs[0]
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()

    test_accuracy = 100 * test_correct / len(test_loader.dataset)
    return test_accuracy


def train_and_evaluate_shapes(n_classes, n_attr, train_loader, val_loader, test_loader, sub_dir, hyperparameters=None,
                              fast=False, device=None, non_blocking=False, n_bootstrap=1, seed=None, verbose=1):
    """
    Trains the shapes models given some hyperparameters, classes, attributes and data subdirectory.
    The data subdirectory should be generated by `make_subset_shapes()`.

    Args:
        n_classes (int): Amount of classes.
        n_attr (int): Amount of attribues.
        train_loader (dataloader): The training dataloader.
        val_loader (dataloader): The validation dataloader.
        test_loader (dataloader): The test dataloader.
        sub_dir (str): Name of the subset directory.
        hyperparameters (dict of dict, optional): Dictionary of the hyperparameter-dictionaries.
            Should be read from yaml file in "hyperparameters/". If `None`, will read
            default or fast hyperparameters. Defaults to None.
        fast (bool, optional): If True, will load hyperparameters with very low `n_epochs`. This
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
            can be used for fast testing of the code. Defaults to False.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        n_bootstrap (int): The amount of bootstrap iterations. This is only used to give the saved model unique names,
            so that many different bootstrap iterations can be ran in parallel.
        seed (int): Seed to seed training.
        verbose (int): Determines how much output is printed. If 2, will print stats after every epoch.
            If 1, will print after last epoch only. If 0, will not print anything.

    Returns:
        list of dict: list of model training-histories, including test_accuracy.
    """
    if device is None or device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = hyperparameters
    if hp is None:
        n_subset = int(sub_dir.strip("/").strip("sub"))
        hp = get_hyperparameters(n_classes, n_attr, n_subset, fast=fast)

    models = load_models_shapes(n_classes, n_attr, hyperparameters=hp)
    histories = []
    criterion = nn.CrossEntropyLoss()
    attr_criterion = nn.BCEWithLogitsLoss()

    for i in range(len(models)):
        model = models[i]
        model_string = MODEL_STRINGS[i]
        print(f"Running model {model.name}:")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=hp[model_string]["learning_rate"])
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hp[model_string]["gamma"])

        class_dir = "c" + str(n_classes) + "_a" + str(n_attr) + "/"
        n_epochs = MAX_EPOCHS
        if fast:  # Make sure not to overwrite in case of fast testing
            class_dir = class_dir + "testing/"
            n_epochs = 2
        save_model_path = "saved_models/" + class_dir + sub_dir
        save_name = MODEL_STRINGS[i] + "_b" + str(n_bootstrap)
        os.makedirs(save_model_path, exist_ok=True)
        if seed is not None:  # Seed before training, for reprorudibility
            seed_everything(seed)
        if model.name == "ShapesCNN":
            history = train_simple(model, criterion, optimizer, train_loader, val_loader, scheduler=exp_lr_scheduler,
                                   n_epochs=n_epochs, n_early_stop=None, device=device,
                                   non_blocking=non_blocking, verbose=verbose, model_dir=save_model_path,
                                   model_save_name=save_name, history_save_name=None)
        else:
            history = train_cbm(model, criterion, attr_criterion, optimizer, train_loader, val_loader,
                                n_epochs=n_epochs, attr_weight=hp[model_string]["attr_weight"],
                                scheduler=exp_lr_scheduler, n_early_stop=None, device=device, non_blocking=non_blocking,
                                verbose=verbose, model_dir=save_model_path, model_save_name=save_name,
                                history_save_name=None)

        state_dict = torch.load(save_model_path + save_name + "_loss.pth")
        model.load_state_dict(state_dict)
        test_accuracy = evaluate_on_test_set(model, test_loader, device=device, non_blocking=non_blocking)
        history["test_accuracy"] = [test_accuracy]
        histories.append(history)

    return histories


def run_models_on_subsets_and_plot(
        dataset_dir, n_classes, n_attr, subsets, n_bootstrap=1, fast=False, batch_size=16, device=None,
        non_blocking=False, num_workers=0, pin_memory=False, persistent_workers=False, base_seed=57, verbose=1):
    """
    Run all available models (from MODEL_STRINGS) on different subsets. For each subset, plots the models
    training-history together, and also plots the test error for each subset.
    To even out variance, the subsets can be bootstrapped with "n_bootstrap".
    Note that this will only bootstrap the training and validation set, while the test-set remains the same.

    Args:
        dataset_dir (str): Path to the dataset_directory
        n_classes (int): The amount of classes in the dataset.
        n_attr (int): The amount of attributes in the dataset
        subsets (list of int): List of the subsets to run on.
        n_bootstrap (int, optional): The amount of times to draw new subset and run models. Defaults to 1.
        fast (bool, optional): If True, will load hyperparameters with low `n_epochs`. Defaults to False.
        batch_size (int, optional): Batch-size of the training. Defaults to 16.
        device (str): Use "cpu" for cpu training and "cuda" for gpu training.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        num_workers (int): The amount of subprocesses used to load the data from disk to RAM.
            0, default, means that it will run as main process.
        pin_memory (bool): Whether or not to pin RAM memory (make it non-pagable).
            This can increase loading speed from RAM to VRAM (when using `to("cuda:0")`,
            but also increases the amount of RAM necessary to run the job. Should only
            be used with GPU training.
        persistent_workers (bool): If `True`, will not shut down workers between epochs.
        base_seed (int, optional): Seed for the subset generation. Will iterate with 1 for every bootstrap.
            Defaults to 57.
        verbose (int, optional): Controls the verbosity. Defaults to 1.
    """
    class_dir = "c" + str(n_classes) + "_a" + str(n_attr) + "/"
    if fast:  # Make sure not to overwrite in case of fast testing
        class_dir = class_dir + "testing/"
    test_accuracies_lists = [[] for _ in range(len(MODEL_STRINGS))]  # Test accuracies for every model for every subset

    for subset in subsets:
        seed = base_seed
        subset_dir = "sub" + str(subset) + "/"
        if verbose != 0:
            print(f"Beginning subset {subset}:")
        histories_total = None
        # Load test-set for full dataset (not subset)
        test_loader = load_data_shapes(
            mode="test", path=dataset_dir, subset_dir="", batch_size=batch_size,
            drop_last=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        for i in range(n_bootstrap):
            make_subset_shapes(dataset_dir, subset, n_classes, seed=seed)
            seed += 1  # Change seed so that subset will be different for the bootstrapping

            train_loader, val_loader = load_data_shapes(
                mode="train-val", path=dataset_dir, subset_dir=subset_dir, batch_size=batch_size, drop_last=False,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

            histories = train_and_evaluate_shapes(
                n_classes, n_attr, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                sub_dir=subset_dir, hyperparameters=None, fast=fast, device=device, non_blocking=non_blocking,
                n_bootstrap=n_bootstrap, seed=base_seed, verbose=verbose)
            histories_total = add_histories(histories_total, histories, n_bootstrap)

        for i in range(len(MODEL_STRINGS)):
            test_accuracies_lists[i].append(histories_total[i]["test_accuracy"][0])

        save_name = class_dir + "c" + str(n_classes) + "_a" + str(n_attr) + "_b" + str(n_bootstrap)
        save_name += "_sub" + str(subset) + ".png"
        os.makedirs("plots/" + class_dir, exist_ok=True)
        plot_training_histories(histories=histories_total, names=MODEL_STRINGS, colors=COLORS, attributes=False,
                                title=save_name, save_dir="plots/", save_name=save_name)
        pickle_save_name = "history/" + class_dir + "histories_sub" + str(subset) + "_b" + str(n_bootstrap) + ".pkl"
        os.makedirs("history/" + class_dir, exist_ok=True)
        with open(pickle_save_name, "wb") as outfile:
            pickle.dump(histories_total, outfile)
        make_subset_shapes(dataset_dir, subset, n_classes, seed=base_seed)  # Reset subset to base-seed

    plot_subset_test_accuracies(x_values=subsets, test_accuracies_lists=test_accuracies_lists, names=MODEL_STRINGS,
                                colors=COLORS, title=None,
                                save_name=class_dir + "test_accuracies_b" + str(n_bootstrap) + ".png")
    with open("history/" + class_dir + "test_accuracies_b" + str(n_bootstrap) + ".pkl", "wb") as outfile:
        pickle.dump(test_accuracies_lists, outfile)
