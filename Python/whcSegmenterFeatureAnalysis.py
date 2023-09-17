from os.path import join
from matplotlib import pyplot as plt
import torch

from whcImageLoading import WHCSegmenterExtractedFeaturesLoader
from whcSegmenterDataPreparation import WHCCRFeatureExtractor, load_feature_extractor_checkpoint


N_OF_FEATURES_TO_PLOT = 64
N_OF_AXES_X = 5
N_OF_AXES_Y = 4

SAMPLES_LIMIT = 10


class FeatureVisualization:
    def __init__(self):
        plt.ion()
        n_figures = N_OF_FEATURES_TO_PLOT // (N_OF_AXES_X * N_OF_AXES_Y)
        if N_OF_FEATURES_TO_PLOT % (N_OF_AXES_X * N_OF_AXES_Y) != 0:
            n_figures += 1
        self.feature_figures = []
        self.feature_figures_to_save = []
        for i in range(n_figures):
            figure, axes = plt.subplots(N_OF_AXES_Y, N_OF_AXES_X, figsize=(15.0, 10.0),
                                        sharex='all', sharey='all', constrained_layout=True)
            figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
            figure.suptitle(f'Features vs Classes')
            self.feature_figures.append(axes.flat)
            self.feature_figures_to_save.append(figure)
        for i in range(N_OF_FEATURES_TO_PLOT):
            figure_index = i // (N_OF_AXES_X * N_OF_AXES_Y)
            ax_index = i % (N_OF_AXES_X * N_OF_AXES_Y)
            self.feature_figures[figure_index][ax_index].set_title(f"{i}")

    def plot_features(self, plotted_features, plotted_targets):
        for i in range(plotted_features.shape[1]):
            figure_index = i // (N_OF_AXES_X * N_OF_AXES_Y)
            ax_index = i % (N_OF_AXES_X * N_OF_AXES_Y)
            self.feature_figures[figure_index][ax_index].plot(plotted_features[:, i], plotted_targets, '.')
            plt.show(block=False)

    def save_figures(self, figure_file_path):
        for i in range(n_figures):
            path = figure_file_path + f"_{i:02d}.png"
            self.feature_figures_to_save[i].savefig(path)


if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=1000)

    IMAGE_DIRECTORY_PATH = "Dataset/"
    FEATURE_EXTRACTOR_CHECKPOINT_PATH = "Checkpoints/"
    FEATURE_EXTRACTOR_CHECKPOINT_NAME = "recognizer_checkpoint.pth"

    data_loader = WHCSegmenterExtractedFeaturesLoader(IMAGE_DIRECTORY_PATH)

    whc_feature_extractor = WHCCRFeatureExtractor()
    cp_path_to_load = join(FEATURE_EXTRACTOR_CHECKPOINT_PATH, FEATURE_EXTRACTOR_CHECKPOINT_NAME)
    load_feature_extractor_checkpoint(cp_path_to_load, whc_feature_extractor)
    whc_feature_extractor.eval()

    visualization_figure = FeatureVisualization()

    progress = 0

    while ((not data_loader.reached_train_dataset_end())
           and progress < SAMPLES_LIMIT):
        features_sample, targets_sample = data_loader.next_train_batch(whc_feature_extractor)
        visualization_figure.plot_features(features_sample, targets_sample)

        progress += 1
        print(f'\rprocessed {progress} images', end='')

    # blocking the last plot
    plt.show(block=True)
