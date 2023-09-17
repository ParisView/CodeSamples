import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn as nn

from whcImageLoading import WHCSegmenterImageLoader
from widthHeightClassTargetMarking import whc_targets

PRECISION_INDEX = 0
RECALL_INDEX = 1
F_SCORE_INDEX = 2
P_R_F_INVALID_VALUE = 1.1

STATE_DICTIONARY = 'state_dictionary'
OPTIMIZER_DICTIONARY = 'optimizer_dictionary'


def save_checkpoint(path, model, optimizer):
    state = {
        STATE_DICTIONARY: model.state_dict(),
        OPTIMIZER_DICTIONARY: optimizer.state_dict()
    }
    torch.save(state, path)
    print('model saved to %s' % path)


def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state[STATE_DICTIONARY])
    optimizer.load_state_dict(state[OPTIMIZER_DICTIONARY])
    print('model loaded from %s' % path)


class WHCCharSegmentationConvolutionalModel(nn.Module):
    """
    This is a model for segmentation of an image with a string of characters
    """

    def __init__(self):
        super(self.__class__, self).__init__()

        NUMBER_OF_INPUT_CHANNELS = 1
        LAYER_1_NUMBER_OF_FEATURE_MAPS = 8
        LAYER_1_WEIGHTS_WIDTH = 3
        LAYER_2_NUMBER_OF_FEATURE_MAPS = 8
        LAYER_2_WEIGHTS_WIDTH = 3
        # layer 3 was excluded from model structure based on results of training experiments
        LAYER_4_NUMBER_OF_INPUT_FEATURES = 8
        LAYER_4_NUMBER_OF_FEATURES = 8
        # layer 5 was excluded from model structure based on results of training experiments
        # layer 6 was excluded from model structure based on results of training experiments
        LAYER_7_NUMBER_OF_FEATURES = len(whc_targets.keys())

        # convolutional layers
        self.l1_convolution = nn.Conv2d(
            in_channels=NUMBER_OF_INPUT_CHANNELS,
            out_channels=LAYER_1_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1_WEIGHTS_WIDTH, LAYER_1_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l1_batch_norm = nn.BatchNorm2d(LAYER_1_NUMBER_OF_FEATURE_MAPS)

        self.l2_convolution = nn.Conv2d(
            in_channels=LAYER_1_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_2_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_2_WEIGHTS_WIDTH, LAYER_2_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l2_batch_norm = nn.BatchNorm2d(LAYER_2_NUMBER_OF_FEATURE_MAPS)

        # after convolution the resulting feature maps are considered as a batch of columns
        # of pixels. each column is considered as a single target. targets can belong to the
        # following classes: character start, character middle, character end or space.

        # linear layers (at this point image width is taken as batch size)
        self.l4_input_batch_norm = nn.BatchNorm1d(LAYER_4_NUMBER_OF_INPUT_FEATURES)
        self.l4_linear = nn.Linear(
            in_features=LAYER_4_NUMBER_OF_INPUT_FEATURES,
            out_features=LAYER_4_NUMBER_OF_FEATURES
        )
        self.l4_batch_norm = nn.BatchNorm1d(LAYER_4_NUMBER_OF_FEATURES)

        self.l7_linear = nn.Linear(
            in_features=LAYER_4_NUMBER_OF_FEATURES,
            out_features=LAYER_7_NUMBER_OF_FEATURES
        )

    def forward(self, input_data):
        """
        This method calculates the result of forward propagation of the input data
        through the model
        :param input_data - model input data, should be an image with a string of characters
        """

        # perform computations of convolution layers
        l1_out = nn.functional.relu(self.l1_batch_norm(self.l1_convolution(input_data)))
        l2_out = nn.functional.relu(self.l2_batch_norm(self.l2_convolution(l1_out)))

        # reshape feature maps to use image width as batch size in further computations
        l2_out = l2_out.permute(3, 0, 1, 2)
        l2_out = l2_out.reshape(l2_out.shape[0], -1, l2_out.shape[3])
        # perform max pool computation on image height
        l2_out = nn.functional.adaptive_max_pool1d(input=l2_out, output_size=1)
        l2_out = l2_out.reshape(l2_out.shape[0], -1)

        # perform computations of linear layers
        result = self.l4_input_batch_norm(l2_out)
        result = nn.functional.relu(self.l4_batch_norm(self.l4_linear(result)))
        result = nn.functional.log_softmax(self.l7_linear(result), dim=1)

        return result


class TrainProcessVisualization:
    def __init__(self, n_of_cross_validation_sets=1):
        plt.ion()

        # creating list of loss figures for all cross validation sets
        self.loss_figure = []
        self.loss_ax1 = []
        self.loss_ax2 = []

        # creating list of precession-recall-f_score figures for all cross validation sets
        self.prf_figure = []
        self.prf_axes = []
        self.first_time_shown = []

        # filling lists of figures with figures
        for i in range(n_of_cross_validation_sets):
            # creating figure
            figure = plt.figure(figsize=(13.0, 5.0), constrained_layout=True)
            figure.set_constrained_layout_pads(w_pad=0.1, h_pad=0, wspace=0, hspace=0)
            # appending figure to the list
            self.loss_figure.append(figure)
            self.loss_ax1.append(figure.add_subplot(121))
            self.loss_ax2.append(figure.add_subplot(122))
            self.loss_ax1[-1].set_title(f'Loss VSet {i}')
            self.loss_ax2[-1].set_title(f'Accuracy VSet {i}')
            # creating figure
            figure, axes = plt.subplots(2, 3, figsize=(10.0, 5.0), sharex='all', sharey='all', constrained_layout=True)
            figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
            figure.suptitle(f'Precision, Recall, F-score VSet {i}')
            # appending figure to the list
            self.prf_figure.append(figure)
            self.prf_axes.append(axes.flat)
            for j in range(len(whc_targets.keys())):
                ax_title = list(whc_targets.keys())[list(whc_targets.values()).index(j)] + f', {j}'
                self.prf_axes[-1][j].set_title(ax_title)
            self.first_time_shown.append(True)

    def plot_train_process(self, plotted_train_loss, plotted_validation_loss, plotted_validation_accuracy,
                           plotted_precision_recall_f, cross_validation_set=0):
        self.loss_ax1[cross_validation_set].plot(plotted_train_loss[cross_validation_set],
                                                 color='b', label='train')
        self.loss_ax1[cross_validation_set].plot(plotted_validation_loss[cross_validation_set],
                                                 color='r', label='validation')
        self.loss_ax2[cross_validation_set].plot(plotted_validation_accuracy[cross_validation_set],
                                                 color='k')
        for j in range(len(plotted_precision_recall_f[cross_validation_set])):
            self.prf_axes[cross_validation_set][j].plot(
                plotted_precision_recall_f[cross_validation_set][j][PRECISION_INDEX], color='g', label='precision')
            self.prf_axes[cross_validation_set][j].plot(
                plotted_precision_recall_f[cross_validation_set][j][RECALL_INDEX], color='b', label='recall')
            self.prf_axes[cross_validation_set][j].plot(
                plotted_precision_recall_f[cross_validation_set][j][F_SCORE_INDEX], color='r', label='f-score')
        if self.first_time_shown[cross_validation_set]:
            self.first_time_shown[cross_validation_set] = False
            self.loss_ax1[cross_validation_set].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
            self.prf_axes[cross_validation_set][-1].plot(0, color='g', label='precision')
            self.prf_axes[cross_validation_set][-1].plot(0, color='b', label='recall')
            self.prf_axes[cross_validation_set][-1].plot(0, color='r', label='f-score')
            self.prf_axes[cross_validation_set][-1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
        self.loss_figure[cross_validation_set].canvas.flush_events()
        self.prf_figure[cross_validation_set].canvas.flush_events()
        plt.show(block=False)

    def save_figures(self, loss_fig_file_path, prf_fig_file_path, cross_validation_set=0):
        self.loss_figure[cross_validation_set].savefig(loss_fig_file_path)
        self.prf_figure[cross_validation_set].savefig(prf_fig_file_path)


if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=1000)

    # creating directory paths and file names
    IMAGE_DIRECTORY_PATH = "Dataset/"
    SAVE_MODEL_PATH = "Checkpoints/"
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    main_part = "segmenter_checkpoint" + date
    main_path = SAVE_MODEL_PATH + main_part
    cp_suffix = ".pth"
    loss_fig_suffix = "_l.png"
    prf_fig_suffix = "_f.png"
    log_suffix = ".txt"
    log_file_path = main_path + log_suffix
    log_file = open(log_file_path, 'w', encoding='utf8')

    # setting train process parameters
    N_EPOCHS = 20
    learning_rate = 3e-4
    PRINT_MODEL_SUMMARY_IS_ON = True
    TRAINING_PROCESS_IS_ON = True
    SAVING_CHECKPOINTS_IS_ON = True

    data_loader = WHCSegmenterImageLoader(IMAGE_DIRECTORY_PATH)
    print(f"\nfound {len(data_loader.dataset_list)} image files with marked targets "
          f"in directory: {IMAGE_DIRECTORY_PATH}", file=log_file)
    print(f"test dataset list lengths: {data_loader.total_images_in_test_dataset}", file=log_file)
    print(f"train dataset list lengths: {data_loader.total_images_in_train_dataset}", file=log_file)

    n_of_cross_validation_sets = data_loader.n_of_cross_validation_datasets

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("\n\ndevice =", device)
    print(f"cuda version = {torch.version.cuda}")

    print("\n\ndevice =", device, file=log_file)
    print(f"cuda version = {torch.version.cuda}", file=log_file)
    print(f'learning rate: {learning_rate}', file=log_file)

    # creating loss function
    loss_function = nn.NLLLoss()
    loss_function.to(device, torch.float32)

    # creating lists for training process of multiple cross validation sets
    whc_char_segmentation_model = []
    model_optimizer = []
    train_loss = []
    validation_loss = []
    validation_accuracy = []
    precision_recall_f = []
    min_loss = []
    checkpoint_to_save = []

    # for each cross validation set filling lists with their items
    for cross_validation_set in range(n_of_cross_validation_sets):
        whc_char_segmentation_model.append(WHCCharSegmentationConvolutionalModel().to(device, torch.float32))

        model_optimizer.append(torch.optim.Adam(whc_char_segmentation_model[-1].parameters(), lr=learning_rate))
        train_loss.append([])
        validation_loss.append([])
        validation_accuracy.append([])
        precision_recall_f.append([])
        for i in range(len(whc_targets.keys())):
            precision_recall_f[-1].append([[], [], []])
        min_loss.append(0.0)
        checkpoint_to_save.append({})

    # printing model summary
    if PRINT_MODEL_SUMMARY_IS_ON:
        print('', end='\n\n')
        print('-------------------------  model summary  -------------------------', end='\n\n')
        for parameter in whc_char_segmentation_model[0].state_dict().keys():
            print(f'parameter name:\t\t{parameter},\t\tsize:'
                  f' {whc_char_segmentation_model[0].state_dict()[parameter].shape}')
        print('\n---------------------  end of model summary  ---------------------', end='\n\n')

        print('', end='\n\n', file=log_file)
        print('-------------------------  model summary  -------------------------', end='\n\n', file=log_file)
        for parameter in whc_char_segmentation_model[0].state_dict().keys():
            print(f'parameter name:\t\t{parameter},\t\tsize:'
                  f' {whc_char_segmentation_model[0].state_dict()[parameter].shape}', file=log_file)
        print('\n---------------------  end of model summary  ---------------------', end='\n\n', file=log_file)

    # train loop
    if TRAINING_PROCESS_IS_ON:
        start_time = time.time()
        visualization_figure = TrainProcessVisualization(n_of_cross_validation_sets)

        for epoch in range(N_EPOCHS):
            print(f'\n=================================================\n'
                  f'training epoch: {epoch + 1}\n')
            print(f'\n=================================================\n'
                  f'training epoch: {epoch + 1}\n', file=log_file)

            for cross_validation_set in range(n_of_cross_validation_sets):
                epoch_train_loss = []
                epoch_validation_loss = []
                epoch_validation_accuracy = []
                validation_targets_vs_predictions = torch.zeros(
                    [len(whc_targets.keys()), len(whc_targets.keys())], dtype=torch.int32)
                epoch_start_time = time.time()
                progress = 0

                # training the model on training dataset
                print('\n-------------------------------------')
                print(f'-------------------------------------\n'
                      f'total train images {data_loader.total_images_in_train_dataset[cross_validation_set]}'
                      f' in cross-validation set {cross_validation_set}', file=log_file)
                whc_char_segmentation_model[cross_validation_set].train()
                data_loader.reset_train_dataset_index(cross_validation_set)
                data_loader.reset_test_dataset_index(cross_validation_set)

                while not data_loader.reached_train_dataset_end(cross_validation_set):
                    whc_char_segmentation_model[cross_validation_set].zero_grad()
                    image_sample, targets_sample = data_loader.next_train_sample(cross_validation_set, device)
                    model_output = whc_char_segmentation_model[cross_validation_set](image_sample)

                    loss = loss_function(model_output, targets_sample)
                    loss.backward()
                    epoch_train_loss.append(loss.item())

                    progress += 1
                    if progress % 100 == 0 or (
                            progress == data_loader.total_images_in_train_dataset[cross_validation_set]):
                        print(f'\rprocessed {progress} images', end='')
                    model_optimizer[cross_validation_set].step()

                print(f'processed {progress} images', file=log_file)

                # testing the model on validation dataset
                print('\nvalidating')
                print('\nvalidating', file=log_file)
                print(f'total test images {data_loader.total_images_in_test_dataset[cross_validation_set]}'
                      f' in cross-validation set {cross_validation_set}', file=log_file)
                whc_char_segmentation_model[cross_validation_set].eval()
                progress = 0

                with torch.no_grad():
                    while not data_loader.reached_test_dataset_end(cross_validation_set):
                        image_sample, targets_sample = data_loader.next_test_sample(
                            cross_validation_set, device)
                        model_output = whc_char_segmentation_model[cross_validation_set](image_sample)

                        loss = loss_function(model_output, targets_sample)
                        epoch_validation_loss.append(loss.item())
                        prediction = model_output.max(1)[1].data
                        epoch_validation_accuracy.append(np.mean((targets_sample.cpu() ==
                                                                  prediction.cpu()).numpy()))

                        progress += 1
                        if progress % 100 == 0 or (
                                progress == data_loader.total_images_in_train_dataset[cross_validation_set]):
                            print(f'\rprocessed {progress} images', end='')

                        for k in range(targets_sample.shape[0]):
                            validation_targets_vs_predictions[targets_sample[k].item()][prediction[k].item()] += 1

                print(f'processed {progress} images', file=log_file)

                # printing targets vs predictions and calculating precision and recall for each class
                print(f'\tvalidation targets (vertical) vs predictions (horizontal):\n'
                      f'{validation_targets_vs_predictions}')
                print(f'\tvalidation targets (vertical) vs predictions (horizontal):\n'
                      f'{validation_targets_vs_predictions}', file=log_file)
                tp_fp_sum_for_precision = validation_targets_vs_predictions.sum(0)
                tp_fn_sum_for_recall = validation_targets_vs_predictions.sum(1)
                for i in range(validation_targets_vs_predictions.shape[0]):
                    precision_value = P_R_F_INVALID_VALUE if tp_fp_sum_for_precision[i] == 0 else (
                            validation_targets_vs_predictions[i][i] / tp_fp_sum_for_precision[i]).item()
                    precision_recall_f[cross_validation_set][i][PRECISION_INDEX].append(precision_value)
                    recall_value = P_R_F_INVALID_VALUE if tp_fn_sum_for_recall[i] == 0 else (
                            validation_targets_vs_predictions[i][i] / tp_fn_sum_for_recall[i]).item()
                    precision_recall_f[cross_validation_set][i][RECALL_INDEX].append(recall_value)
                    f_score_value = P_R_F_INVALID_VALUE if precision_value + recall_value == 0 else (
                            2 * precision_value * recall_value / (precision_value + recall_value))
                    precision_recall_f[cross_validation_set][i][F_SCORE_INDEX].append(f_score_value)
                    print(f'Class {i}: \tPrecision: {precision_value:.6f}, '
                          f'\t\tRecall: {recall_value:.6f}, \t\tF-score: {f_score_value:.6f}')
                    print(f'Class {i}: \tPrecision: {precision_value:.6f}, '
                          f'\t\tRecall: {recall_value:.6f}, \t\tF-score: {f_score_value:.6f}', file=log_file)

                # printing epoch time and loss
                print(f'\nEpoch {epoch + 1} of {N_EPOCHS} took {time.time() - epoch_start_time:.3f}s')
                print(f'\nEpoch {epoch + 1} of {N_EPOCHS} took {time.time() - epoch_start_time:.3f}s', file=log_file)
                train_loss[cross_validation_set].append(np.mean(epoch_train_loss))
                validation_loss[cross_validation_set].append(np.mean(epoch_validation_loss))
                validation_accuracy[cross_validation_set].append(np.mean(epoch_validation_accuracy))
                print(f'\t   training loss: {train_loss[cross_validation_set][-1]:.6f}')
                print(f'\tvalidation loss: {validation_loss[cross_validation_set][-1]:.6f}')
                print(f'\tvalidation accuracy: {validation_accuracy[cross_validation_set][-1]:.6f}')
                print(f'\t   training loss: {train_loss[cross_validation_set][-1]:.6f}', file=log_file)
                print(f'\tvalidation loss: {validation_loss[cross_validation_set][-1]:.6f}', file=log_file)
                print(f'\tvalidation accuracy: {validation_accuracy[cross_validation_set][-1]:.6f}', file=log_file)

                # plotting the train process
                visualization_figure.plot_train_process(train_loss, validation_loss, validation_accuracy,
                                                        precision_recall_f, cross_validation_set)

                # saving the model in memory on each step, at which loss has decreased
                if SAVING_CHECKPOINTS_IS_ON:
                    if len(validation_loss[cross_validation_set]) < 2:
                        min_loss[cross_validation_set] = validation_loss[cross_validation_set][0]
                    elif validation_loss[cross_validation_set][-1] < min_loss[cross_validation_set]:
                        min_loss[cross_validation_set] = validation_loss[cross_validation_set][-1]
                        checkpoint_to_save[cross_validation_set] = {
                            STATE_DICTIONARY: whc_char_segmentation_model[cross_validation_set].state_dict(),
                            OPTIMIZER_DICTIONARY: model_optimizer[cross_validation_set].state_dict()
                        }

        # saving the model to a file
        if SAVING_CHECKPOINTS_IS_ON:
            for cross_validation_set in range(n_of_cross_validation_sets):
                if checkpoint_to_save[cross_validation_set]:
                    cp_path = main_path + f"_{cross_validation_set}" + cp_suffix
                    torch.save(checkpoint_to_save[cross_validation_set], cp_path)
                    print(f'model for cross-validation set {cross_validation_set} saved to {cp_path}')
                    print(f'model for cross-validation set {cross_validation_set} saved to {cp_path}', file=log_file)

        # saving the figures
        for cross_validation_set in range(n_of_cross_validation_sets):
            loss_fig_file_path = main_path + f"_{cross_validation_set}" + loss_fig_suffix
            prf_fig_file_path = main_path + f"_{cross_validation_set}" + prf_fig_suffix
            visualization_figure.save_figures(loss_fig_file_path, prf_fig_file_path, cross_validation_set)

        # printing overall time
        print(f'\ntrain process took {time.time() - start_time:.1f}s')
        print(f'\ntrain process took {time.time() - start_time:.1f}s', file=log_file)

    log_file.flush()
    log_file.close()

    # blocking the last plot of the train process
    plt.show(block=True)

