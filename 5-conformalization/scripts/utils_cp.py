import random
import pickle
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from utils_classCP import *
from classCP import *
import pickle
from collections import Counter
import json
import logging
import time
import pathlib


class ImageNet_WRN:
    data_directory = f'{pathlib.Path(__file__).parent.resolve()}/data'
    label_filename = 'imagenet-simple-labels.json'
    cmodel_marginal_path = f'{pathlib.Path(__file__).parent.resolve()}/output/model/cp-model-50-marginal.pkl'
    cmodel_conditional_path = f'{pathlib.Path(__file__).parent.resolve()}/output/model/cp-model-50-conditional.pkl'
    rand_generator = torch.Generator().manual_seed(666)
    input_transformer = transforms.Compose([
        # resize to 256x256 pixel
        transforms.Resize(256),
        # crop the image at the center (size needs to be at least 224 pixels)
        transforms.CenterCrop(224),
        # convert to 3-dimensional tensor
        transforms.ToTensor(),
        # normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, model):
        self.model = model

    def load_test_data(self, props):
        """_summary_

        Args:
            prop (tuple): in the format of calibration size, validation size, 

        Returns:
            _type_: _description_
        """
        # Load the full dataset
        self.imagenet_validation_dataset_raw = torchvision.datasets.ImageNet(root=self.data_directory,
                                                                             split='val')

        imagenet_validation_dataset = torchvision.datasets.ImageNet(root=self.data_directory,
                                                                    split='val',
                                                                    transform=self.input_transformer)
        self.imagenet_validation_dataset = imagenet_validation_dataset
        print(
            f"=> Loaded {imagenet_validation_dataset.__len__()} images from ImageNet validation set.")

        if len(props) == 2:
            calib_prop, val_prop = props
            calib_size = int(
                imagenet_validation_dataset.__len__() * calib_prop)
            val_size = int(imagenet_validation_dataset.__len__() * val_prop)
            print(f"{' '* 4} Perform {int(calib_prop*100)}-{val_prop*100} split:")
            print(f"{' '*8} - Calibration size: {calib_size}")
            print(f"{' '*8} - Validation size: {val_size}")
            set_calibration_dataset, set_test_dataset = torch.utils.data.random_split(
                imagenet_validation_dataset,
                props,
                self.rand_generator
            )
        if len(props) == 3:
            self.dataLoader_messages(props)
            set_calibration_dataset, set_test_dataset, _ = torch.utils.data.random_split(
                imagenet_validation_dataset,
                props,
                self.rand_generator
            )
        self.set_calibration_dataset = set_calibration_dataset
        self.set_test_dataset = set_test_dataset
        self.set_calibration_loader = self.loader(set_calibration_dataset)
        self.set_test_loader = self.loader(set_test_dataset)

        return self

    def dataLoader_messages(self, props):
        total_images = self.imagenet_validation_dataset.__len__()
        calib_prop, val_prop, _ = props
        keep_prop = int((calib_prop + val_prop) * 100)
        keep_size = int(keep_prop/100 * total_images)
        calib_size = int(calib_prop * total_images)
        val_size = int(val_prop * total_images)
        print(
            f"{' '* 4} Keep {keep_prop}% of the images, that is {keep_size}")
        print(f"{' '*8} - Calibration size: {calib_size}")
        print(f"{' '*8} - Validation size: {val_size}")

    def load_class_labels(self):
        file_path = f'{self.data_directory}/{self.label_filename}'
        with open(file_path) as f:
            imagenet_classes = json.load(f)
            f.close()

        class_keys = range(len(imagenet_classes))
        labelDict = dict(zip(class_keys, imagenet_classes))
        self.imagenet_classes = labelDict
        self.imagenet_classes_rev = {
            value: key for key, value in labelDict.items()}
        return self

    def loader(self, data, batch_size=128):
        dataLoader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, pin_memory=True)
        return dataLoader

    def inspect_images(self, num_images):
        _, ax = plt.subplots(nrows=1, ncols=num_images,
                             figsize=(2 * num_images, 2))
        data_size = self.set_calibration_dataset.__len__()
        indices = random.sample(range(data_size), num_images)

        for i, index in enumerate(indices):
            image, label = self.set_calibration_dataset.__getitem__(index)
            true_className = self.imagenet_classes[label]
            # fill each subplot
            ax[i].imshow(image.numpy().transpose((1, 2, 0)))
            ax[i].set_title(true_className)
            ax[i].axis('off')
        plt.show()

    def subset_class_maps(self, subset_str):
        val_targets = np.array(self.imagenet_validation_dataset.targets)
        indices = np.array(self.set_calibration_dataset.indices) if subset_str == 'calib' else np.array(
            self.set_test_dataset.indices)
        subset_targets = [self.imagenet_classes[i]
                          for i in val_targets[indices]]

        container = {}
        for i, target_class in enumerate(subset_targets):
            if target_class in container:
                container[target_class].append(i)
            else:
                container[target_class] = [i]
        return container

    def generate_subset_class_maps(self):
        self.set_calibration_map = self.subset_class_maps('calib')
        self.set_test_map = self.subset_class_maps('test')

        return self

    def show_subset_dist(self, subset_str, plot=True):
        val_targets = np.array(self.imagenet_validation_dataset.targets)
        indices = np.array(self.set_calibration_dataset.indices) if subset_str == 'calib' else np.array(
            self.set_test_dataset.indices)

        subset_targets = val_targets[indices]
        calib_labels = Counter([self.imagenet_classes[i]
                               for i in subset_targets])
        counter_df = pd.DataFrame.from_dict(
            calib_labels, orient='index').reset_index().rename(columns={'index': 'class', 0: 'count'})

        if plot:
            fig = px.bar(counter_df, x='class', y='count')
            fig.show()

        return counter_df


def train_conformal(wrn, vault, alpha, lamda_criterion, randomized=True, allow_zero_sets=True):
    """
    Train a conformal model that can produce RAPS predictive set

    Args:
        wrn (_torchvision.models_): pre-trained model from pytorch
        vault (_class_): ImageNet_WRN class object
        alpha (_float_): error rate 
        lamda_criterion (_str_): either "size" or "adaptiveness" to indicate types of coverage
                                 "size" => marginal, "adaptiveness" => conditional
        randomized (_bool_): when setting to `true`, at test-time, the sets will not be randomized. 
        allow_zero_sets (_bool_): when setting to `true`, at test-time, sets of size zero are disallowed. 

    Returns:
        _tuple_: (cmodel, cmodel_eval) to store model and model evaluation
    """
    cp_type = 'marginal' if lamda_criterion == 'size' else 'conditional'

    wrn.eval()
    print(f"{' '*4} Model is in evaluation mode? {not wrn.training}")

    # Training model
    start_time = time.time()
    print(
        f'=> Training conformal model that generate sets of {cp_type} coverage...')
    cmodel = ConformalModel(model=wrn,
                            calib_loader=vault.set_calibration_loader,
                            alpha=alpha,
                            lamda_criterion=lamda_criterion,
                            randomized=randomized,
                            allow_zero_sets=allow_zero_sets)

    end_time = time.time()
    print(
        f"--- {cp_type.capitalize()} conformalization took {convert_time(end_time - start_time)} ---")

    # validate the coverage of the conformal model
    start_time = time.time()
    print('=> Evaluating model coverage...')
    top1, top5, coverage, size = validate(
        vault.set_test_loader, cmodel, print_bool=True)
    model_eval = {'top1': top1,
                  'top5': top5,
                  'coverage': coverage,
                  'size': size}
    end_time = time.time()
    print(
        f"--- Evaluation of {cp_type} conformalization took {convert_time(end_time - start_time)} ---")

    print('=> Saving the model and its evaluation to the directory.')
    output = (cmodel, model_eval)
    model_path = vault.cmodel_marginal_path if lamda_criterion == 'size' else vault.cmodel_conditional_path
    save_cmodel(output, model_path)

    return output


def imagenet_prediction_set(randInt, vault, cmodel_coverage, plot=False, load_conformal=True, verbose=True):
    img, label = vault.set_test_dataset.__getitem__(randInt)
    cp = prediction_set(img, label, vault, cmodel_coverage,
                        plot, load_conformal, verbose)
    return cp


def prediction_set(input_img: torch.tensor, input_label: int, vault: "ImageNet_WRN", cmodel_coverage: str, plot: bool = False, load_conformal: bool = True, verbose: bool = True, **kwargs) -> dict:
    """                             
    Generate a prediction set using a conformal model for the input image.

    Args:                           
        input_img (torch.tensor): Image tensor from ImageNet-val.
        input_label (int): Image class (unreadable).
        vault (class): ImageNet_WRN class object.
        cmodel_coverage (str): Indicate which conformal model used to make the prediction set.
        plot (bool, optional): Flag to plot the image and the set. Defaults to False.
        load_conformal (bool): Flag to state whether to load the conformal model.
        verbose (bool): whether to print log messages.
        **kwargs (class): If `load_conformal` is False, then need to input a cmodel.

    Returns:
        dict: A dictionary with the following keys:
            'score': Conformal scores
            'true_class': True class of the input image
            'pred_class': Predicted class of the input image
            'model_output': A pandas DataFrame showing all classes and their corresponding softmax
            'set': A list of the prediction set
            'setDF': A pandas DataFrame showing classes in the prediction set and the corresponding softmax
            'img_cooked': Input image that is standardized
            'img_raw': Unnormalized image
    """
    # Create the logger
    logger = create_logger(verbose)

    # Load the conformal model trained on
    if load_conformal:
        logger.debug(
            f'=> Loading conformal model trained on 25k ImageNet-val \n   that can produce {cmodel_coverage} coverage')
        cmodel, cmodel_eval = load_cmodel(vault, cmodel_coverage)
        logger.debug(cmodel_eval)
    else:
        logger.debug(
            '=> Using the trained conformal model from the function input.')
        cmodel = kwargs.get('conformal_model')

    logger.debug(
        f'=> Creating prediction set using conformal model for the input image.')

    # Setting seed to ensure reproducibility
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)

    logits, raps_set = cmodel(input_img.view(1, 3, 224, 224))
    raps_set_unreadable = raps_set[0]
    raps_set_readable = [vault.imagenet_classes[el]
                         for el in raps_set_unreadable]

    # get probabilities for each class in the prediction set
    logger.debug(
        f'=> Extracting probabilities for each class in the prediction set.')
    pred_logits = logits[0].detach()
    pred_softmax = torch.nn.functional.softmax(pred_logits, dim=0).numpy()
    pred_labels = [*vault.imagenet_classes.values()]
    model_output_df = pd.DataFrame({'class': pred_labels,
                                    'logit': pred_logits.numpy(),
                                    'softmax': pred_softmax}).sort_values(
        by='logit', ascending=False)

    set_df = model_output_df[model_output_df['class'].isin(
        raps_set_readable)].rename_axis("id").reset_index()

    output = {'true_class': vault.imagenet_classes[input_label],
              'pred_class': model_output_df.iloc[0]['class'],
              'model_output': model_output_df,
              'set': raps_set_readable,
              'setDF': set_df,
              'img_cooked': input_img,
              'img_raw': unnormalize_img(input_img)}
    logger.debug('=> Done!')

    if plot:
        logger.debug('=> Plotting...')
        plot_set(output)

    return output


def load_cmodel(vault, coverage_type):
    model_path = vault.cmodel_marginal_path if coverage_type == 'marginal' else vault.cmodel_conditional_path
    cmodel, cmodel_eval = load_pickle(model_path)
    return (cmodel, cmodel_eval)


def unnormalize_img(input_img):
    unnormalized = (input_img * torch.Tensor(
        [0.229, 0.224, 0.225]).view(-1, 1, 1))+torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    return unnormalized


def generate_plotTitle(cp):
    # Get the true class
    true_class = cp['true_class']
    pred_class = cp['pred_class']
    model_output_df = cp['model_output']

    true_class_prob = model_output_df[model_output_df['class']
                                      == true_class]['softmax'].item()
    pred_class_prob = model_output_df[model_output_df['class']
                                      == pred_class]['softmax'].item()
    success = "Yes" if true_class in cp['set'] else "No"
    # Generate plot title
    if true_class == pred_class:
        plot_title = f"True class is Top-1: {true_class} ({round(true_class_prob, 5)}) <br><sup>Is true class in conformal set? {success}</sup>"
    else:
        plot_title = f"True class: {true_class} ({round(true_class_prob, 5)}) <br> Top-1: {pred_class} ({round(pred_class_prob, 5)}) <br><sup>Is true class in the RAPS set? {success}</sup>"

    output = {'true_class': true_class,
              'pred_class': pred_class,
              'true_class_prob': true_class_prob,
              'pred_class_prob': pred_class_prob,
              'success': success,
              'plot_title': plot_title}

    return output


def generate_plotTitle_corrupted(cp_corrupted):
    corruption_name = cp_corrupted['corruption_name']
    severity = cp_corrupted['corruption_severity']
    add_title_part = f"Under {corruption_name} with severity {severity}/5 <br>"
    titleObj = generate_plotTitle(cp_corrupted)
    titleObj['plot_title'] = add_title_part + titleObj['plot_title']
    return titleObj


def plot_set(cp):
    plot_title_obj = generate_plotTitle(cp)
    set_size = len(cp['set'])
    set_table = cp['setDF'].drop(['logit'], axis=1)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "image"}, {"type": "table"}]],
                        subplot_titles=(plot_title_obj['plot_title'],
                                        f'RAPS Predictive Set: size = {set_size}'),
                        column_widths=(0.4, 0.6))
    fig.add_trace(px.imshow(cp['img_raw'].numpy().transpose((1, 2, 0))).data[0],
                  row=1, col=1)
    fig.add_trace(go.Table(
        header={"values": set_table.columns},
        cells={"values": set_table.T.values}),
        row=1, col=2)

    fig.update_layout(
        autosize=False,
        width=1200,
        height=650,
        yaxis_visible=False, yaxis_showticklabels=False,
        xaxis_visible=False, xaxis_showticklabels=False)

    fig.show()


def save_cmodel(model, filename):
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as input:
        cmodel = pickle.load(input)
    return cmodel


def convert_time(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def create_logger(verbose):
    logger = logging.getLogger("verbose")

    # Clear all previously added handlers (if any)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    level = logging.DEBUG if verbose else logging.INFO

    logger.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    return logger
