import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn as nn

from whcImageLoading import classes_dict
from whcRecognizerImageLoading import WHCImageLoader

PRECISION_INDEX = 0
RECALL_INDEX = 1
F_SCORE_INDEX = 2
P_R_F_INVALID_VALUE = 1.1


class WHCCharRecognitionModel(nn.Module):
    """
    This is a model for character recognition.
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        NUMBER_OF_INPUT_CHANNELS = 1
        LAYER_1_NUMBER_OF_FEATURE_MAPS = 16
        LAYER_1_WEIGHTS_WIDTH = 3
        LAYER_1A_NUMBER_OF_FEATURE_MAPS = 16
        LAYER_1A_WEIGHTS_WIDTH = 3
        LAYER_1B_NUMBER_OF_FEATURE_MAPS = 16
        LAYER_1B_WEIGHTS_WIDTH = 3
        LAYER_1C_NUMBER_OF_FEATURE_MAPS = 32
        LAYER_1C_WEIGHTS_WIDTH = 3
        LAYER_2_NUMBER_OF_FEATURE_MAPS = 32
        LAYER_2_WEIGHTS_WIDTH = 3
        LAYER_2A_NUMBER_OF_FEATURE_MAPS = 64
        LAYER_2A_WEIGHTS_WIDTH = 3
        LAYER_3_NUMBER_OF_FEATURE_MAPS = 128
        LAYER_3_WEIGHTS_WIDTH = 3
        LAYER_4_NUMBER_OF_FEATURES = 128
        LAYER_5_NUMBER_OF_FEATURES = len(classes_dict.keys())

        self.l1_convolution = nn.Conv2d(
            in_channels=NUMBER_OF_INPUT_CHANNELS,
            out_channels=LAYER_1_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1_WEIGHTS_WIDTH, LAYER_1_WEIGHTS_WIDTH)
        )
        self.l1_batch_norm = nn.BatchNorm2d(LAYER_1_NUMBER_OF_FEATURE_MAPS)
        self.l1a_convolution = nn.Conv2d(
            in_channels=LAYER_1_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1A_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1A_WEIGHTS_WIDTH, LAYER_1A_WEIGHTS_WIDTH),
            padding=1
        )
        self.l1a_batch_norm = nn.BatchNorm2d(LAYER_1A_NUMBER_OF_FEATURE_MAPS)
        self.l1b_convolution = nn.Conv2d(
            in_channels=LAYER_1A_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1B_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1B_WEIGHTS_WIDTH, LAYER_1B_WEIGHTS_WIDTH),
            padding=1
        )
        self.l1b_batch_norm = nn.BatchNorm2d(LAYER_1B_NUMBER_OF_FEATURE_MAPS)
        self.l1c_convolution = nn.Conv2d(
            in_channels=LAYER_1B_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1C_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1C_WEIGHTS_WIDTH, LAYER_1C_WEIGHTS_WIDTH),
            padding=1
        )
        self.l1c_batch_norm = nn.BatchNorm2d(LAYER_1C_NUMBER_OF_FEATURE_MAPS)
        self.l2_convolution = nn.Conv2d(
            in_channels=LAYER_1C_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_2_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_2_WEIGHTS_WIDTH, LAYER_2_WEIGHTS_WIDTH)
        )
        self.l2_batch_norm = nn.BatchNorm2d(LAYER_2_NUMBER_OF_FEATURE_MAPS)
        self.l2a_convolution = nn.Conv2d(
            in_channels=LAYER_2_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_2A_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_2A_WEIGHTS_WIDTH, LAYER_2A_WEIGHTS_WIDTH),
            padding=1
        )
        self.l2a_batch_norm = nn.BatchNorm2d(LAYER_2A_NUMBER_OF_FEATURE_MAPS)
        self.l3_convolution = nn.Conv2d(
            in_channels=LAYER_2A_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_3_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_3_WEIGHTS_WIDTH, LAYER_3_WEIGHTS_WIDTH)
        )
        self.l3_batch_norm = nn.BatchNorm2d(LAYER_3_NUMBER_OF_FEATURE_MAPS)
        self.l4_linear = nn.Linear(
            in_features=LAYER_3_NUMBER_OF_FEATURE_MAPS,
            out_features=LAYER_4_NUMBER_OF_FEATURES
        )
        self.l5_linear = nn.Linear(
            in_features=LAYER_4_NUMBER_OF_FEATURES,
            out_features=LAYER_5_NUMBER_OF_FEATURES
        )

    def forward(self, input_data):
        """
        This method calculates the result of forward propagation of the input data
        through the model.
        :param input_data - model input data, should be a 1-channel (grayscale) image of a
        character. Width and height of the image should be from 18 to 38 pixels with 4 pixel step.
        """

        result = nn.functional.relu(self.l1_batch_norm(self.l1_convolution(input_data)))
        result = nn.functional.relu(self.l1a_batch_norm(self.l1a_convolution(result)))
        result = nn.functional.relu(self.l1b_batch_norm(self.l1b_convolution(result)))
        result = nn.functional.relu(self.l1c_batch_norm(self.l1c_convolution(result)))
        result = nn.functional.max_pool2d(input=result, kernel_size=(2, 2))
        result = nn.functional.relu(self.l2_batch_norm(self.l2_convolution(result)))
        result = nn.functional.relu(self.l2a_batch_norm(self.l2a_convolution(result)))
        result = nn.functional.max_pool2d(input=result, kernel_size=(2, 2))
        result = nn.functional.relu(self.l3_batch_norm(self.l3_convolution(result)))

        max_pool_height = result.shape[2]
        max_pool_width = result.shape[3]
        result = nn.functional.max_pool2d(input=result, kernel_size=(max_pool_height, max_pool_width))

        result = result.reshape(result.shape[0], result.shape[1])
        result = nn.functional.relu(self.l4_linear(result))
        result = nn.functional.relu(self.l5_linear(result))
        result = nn.functional.log_softmax(result, dim=1)

        return result


class TrainProcessVisualization:
    def __init__(self, n_of_cross_validation_sets=1):
        plt.ion()
        self.loss_figure = []
        self.loss_ax1 = []
        self.loss_ax2 = []
        self.prf_figure = []
        self.prf_axes = []
        self.first_time_shown = []
        for i in range(n_of_cross_validation_sets):
            figure = plt.figure(figsize=(13.0, 5.0), constrained_layout=True)
            figure.set_constrained_layout_pads(w_pad=0.1, h_pad=0, wspace=0, hspace=0)
            self.loss_figure.append(figure)
            self.loss_ax1.append(figure.add_subplot(121))
            self.loss_ax2.append(figure.add_subplot(122))
            self.loss_ax1[-1].set_title(f'Loss VSet {i}')
            self.loss_ax2[-1].set_title(f'Accuracy VSet {i}')
            figure, axes = plt.subplots(4, 10, figsize=(16.0, 7.0), sharex='all', sharey='all', constrained_layout=True)
            figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
            figure.suptitle(f'Precision, Recall, F-score VSet {i}')
            self.prf_figure.append(figure)
            self.prf_axes.append(axes.flat)
            self.first_time_shown.append(True)

    def plot_train_process(self, plotted_train_loss, plotted_validation_loss, plotted_validation_accuracy,
                           plotted_precision_recall_f, cross_validation_set=0):
        self.loss_ax1[cross_validation_set].cla()
        self.loss_ax2[cross_validation_set].cla()
        self.loss_ax1[cross_validation_set].plot(plotted_train_loss[cross_validation_set],
                                                 color='b', label='train')
        self.loss_ax1[cross_validation_set].plot(plotted_validation_loss[cross_validation_set],
                                                 color='r', label='validation')
        self.loss_ax2[cross_validation_set].plot(plotted_validation_accuracy[cross_validation_set],
                                                 color='k')
        for j in range(len(plotted_precision_recall_f[cross_validation_set])):
            self.prf_axes[cross_validation_set][j].cla()
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


if __name__ == "__main__":
    # This training process is designed for preliminary assessment of the complexity of the task
    # of character recognition and for experimenting with model parameters.
    # It has no intent or means for checkpoint saving and should not be used for production
    # model training or final tuning.

    torch.set_printoptions(profile="full", linewidth=1000)

    IMAGE_DIRECTORY_PATH = "Dataset/"

    N_EPOCHS = 100
    MAX_BATCH_SIZE = 500
    learning_rate = 3e-4
    PRINT_MODEL_SUMMARY_IS_ON = True
    TRAINING_PROCESS_IS_ON = True

    MAX_SAMPLES_IN_CLASS = 3000

    train_dataset_shuffle = True
    train_dataset_sorted = True
    train_dataset_multiply = False
    test_dataset_sorted = True

    data_loader = WHCImageLoader(
        IMAGE_DIRECTORY_PATH, max_batch_size=MAX_BATCH_SIZE, max_samples_in_class=MAX_SAMPLES_IN_CLASS,
        train_dataset_shuffle=train_dataset_shuffle, train_dataset_sorted=train_dataset_sorted,
        train_dataset_multiply=train_dataset_multiply, test_dataset_sorted=test_dataset_sorted
    )

    print(f"\n\nMAX_BATCH_SIZE: {MAX_BATCH_SIZE}, train_dataset_shuffle: {train_dataset_shuffle},"
          f" train_dataset_sorted: {train_dataset_sorted}, train_dataset_multiply: {train_dataset_multiply}, "
          f"test_dataset_sorted: {test_dataset_sorted}")
    print(f"max characters in class limitation: {MAX_SAMPLES_IN_CLASS}")

    n_of_cross_validation_sets = data_loader.n_of_cross_validation_datasets

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("\n\ndevice =", device)
    print(f"cuda version = {torch.version.cuda}")
    print(f'learning rate: {learning_rate}')

    loss_function = nn.NLLLoss()
    loss_function.to(device, torch.float32)
    whc_char_recognition_model = []
    model_optimizer = []
    train_loss = []
    validation_loss = []
    validation_accuracy = []
    precision_recall_f = []
    min_loss = []

    for cross_validation_set in range(n_of_cross_validation_sets):
        whc_char_recognition_model.append(WHCCharRecognitionModel().to(device, torch.float32))
        model_optimizer.append(torch.optim.Adam(whc_char_recognition_model[-1].parameters(), lr=learning_rate))
        train_loss.append([])
        validation_loss.append([])
        validation_accuracy.append([])
        precision_recall_f.append([])
        for i in range(len(classes_dict.keys())):
            precision_recall_f[-1].append([[], [], []])
        min_loss.append(0.0)

    if PRINT_MODEL_SUMMARY_IS_ON:
        print('', end='\n\n')
        print('-------------------------  model summary  -------------------------', end='\n\n')
        for parameter in whc_char_recognition_model[0].state_dict().keys():
            print(f'parameter name:\t\t{parameter},\t\tsize:'
                  f' {whc_char_recognition_model[0].state_dict()[parameter].shape}')
        print('\n---------------------  end of model summary  ---------------------', end='\n\n')

    # train loop
    if TRAINING_PROCESS_IS_ON:
        start_time = time.time()
        visualization_figure = TrainProcessVisualization(n_of_cross_validation_sets)

        for epoch in range(N_EPOCHS):
            print(f'\n=================================================\n'
                  f'training epoch: {epoch + 1}\n')

            for cross_validation_set in range(n_of_cross_validation_sets):
                epoch_train_loss = []
                epoch_validation_loss = []
                epoch_validation_accuracy = []
                validation_targets_vs_predictions = torch.zeros(
                    [len(classes_dict.keys()), len(classes_dict.keys())], dtype=torch.int32)
                epoch_start_time = time.time()
                progress = 0

                # training the model on training dataset
                print(f'-------------------------------------\n'
                      f'total train images {data_loader.total_images_in_train_dataset[cross_validation_set]}'
                      f' in cross-validation set {cross_validation_set}')
                whc_char_recognition_model[cross_validation_set].train()
                data_loader.reset_train_dataset_index(cross_validation_set)
                data_loader.reset_test_dataset_index(cross_validation_set)

                while not data_loader.reached_train_dataset_end(cross_validation_set):
                    whc_char_recognition_model[cross_validation_set].zero_grad()
                    image_batch, target_batch = data_loader.next_train_batch(cross_validation_set, device)
                    model_output = whc_char_recognition_model[cross_validation_set](image_batch)
                    loss = loss_function(model_output, target_batch)
                    loss.backward()
                    epoch_train_loss.append(loss.item())
                    progress += image_batch.shape[0]
                    print(f'\rprocessed {progress} images', end='')
                    model_optimizer[cross_validation_set].step()

                # testing the model on validation dataset
                print('\nvalidating')
                print(f'total test images {data_loader.total_images_in_test_dataset[cross_validation_set]}'
                      f' in cross-validation set {cross_validation_set}')
                whc_char_recognition_model[cross_validation_set].eval()
                progress = 0

                with torch.no_grad():
                    while not data_loader.reached_test_dataset_end(cross_validation_set):
                        image_batch, target_batch = data_loader.next_test_batch(cross_validation_set, device)
                        model_output = whc_char_recognition_model[cross_validation_set](image_batch)
                        loss = loss_function(model_output, target_batch)
                        epoch_validation_loss.append(loss.item())
                        prediction = model_output.max(1)[1].data
                        epoch_validation_accuracy.append(np.mean((target_batch.cpu() == prediction.cpu()).numpy()))
                        progress += image_batch.shape[0]
                        print(f'\rprocessed {progress} images', end='')

                        for k in range(image_batch.shape[0]):
                            validation_targets_vs_predictions[target_batch[k].item()][prediction[k].item()] += 1

                # printing targets vs predictions and calculating precision and recall for each class
                print(f'\tvalidation targets vs predictions:\n{validation_targets_vs_predictions}')
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

                # printing epoch time and loss
                print(f'\nEpoch {epoch + 1} of {N_EPOCHS} took {time.time() - epoch_start_time:.3f}s')
                train_loss[cross_validation_set].append(np.mean(epoch_train_loss))
                validation_loss[cross_validation_set].append(np.mean(epoch_validation_loss))
                validation_accuracy[cross_validation_set].append(np.mean(epoch_validation_accuracy))
                print(f'\t   training loss: {train_loss[cross_validation_set][-1]:.6f}')
                print(f'\tvalidation loss: {validation_loss[cross_validation_set][-1]:.6f}')
                print(f'\tvalidation accuracy: {validation_accuracy[cross_validation_set][-1]:.6f}')

                # plotting the train process
                visualization_figure.plot_train_process(train_loss, validation_loss, validation_accuracy,
                                                        precision_recall_f, cross_validation_set)

        # printing overall time
        print(f'\ntrain process took {time.time() - start_time:.1f}s')

    # blocking the last plot of the train process
    plt.show(block=True)

