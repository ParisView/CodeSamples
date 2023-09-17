from os.path import isfile, join
import torch
from torch import nn as nn

from whcImageLoading import classes_dict, WHCSegmenterExtractedFeaturesLoader


def load_feature_extractor_checkpoint(path, model, device=torch.device("cpu")):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['state_dictionary'])
    print('feature extractor loaded from %s' % path)


class WHCCRFeatureExtractor(nn.Module):
    """
    This is a feature extraction class for WHCCharRecognitionModel.
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
            kernel_size=(LAYER_1_WEIGHTS_WIDTH, LAYER_1_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l1_batch_norm = nn.BatchNorm2d(LAYER_1_NUMBER_OF_FEATURE_MAPS)
        self.l1a_convolution = nn.Conv2d(
            in_channels=LAYER_1_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1A_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1A_WEIGHTS_WIDTH, LAYER_1A_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l1a_batch_norm = nn.BatchNorm2d(LAYER_1A_NUMBER_OF_FEATURE_MAPS)
        self.l1b_convolution = nn.Conv2d(
            in_channels=LAYER_1A_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1B_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1B_WEIGHTS_WIDTH, LAYER_1B_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l1b_batch_norm = nn.BatchNorm2d(LAYER_1B_NUMBER_OF_FEATURE_MAPS)
        self.l1c_convolution = nn.Conv2d(
            in_channels=LAYER_1B_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_1C_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_1C_WEIGHTS_WIDTH, LAYER_1C_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l1c_batch_norm = nn.BatchNorm2d(LAYER_1C_NUMBER_OF_FEATURE_MAPS)
        self.l2_convolution = nn.Conv2d(
            in_channels=LAYER_1C_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_2_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_2_WEIGHTS_WIDTH, LAYER_2_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l2_batch_norm = nn.BatchNorm2d(LAYER_2_NUMBER_OF_FEATURE_MAPS)
        self.l2a_convolution = nn.Conv2d(
            in_channels=LAYER_2_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_2A_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_2A_WEIGHTS_WIDTH, LAYER_2A_WEIGHTS_WIDTH),
            padding=(0, 1)
        )
        self.l2a_batch_norm = nn.BatchNorm2d(LAYER_2A_NUMBER_OF_FEATURE_MAPS)
        self.l3_convolution = nn.Conv2d(
            in_channels=LAYER_2A_NUMBER_OF_FEATURE_MAPS,
            out_channels=LAYER_3_NUMBER_OF_FEATURE_MAPS,
            kernel_size=(LAYER_3_WEIGHTS_WIDTH, LAYER_3_WEIGHTS_WIDTH),
            padding=(0, 1)
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
        through the model
        input_data - model input data
        """
        concatenated_tensors = []

        l1_out = nn.functional.relu(self.l1_batch_norm(self.l1_convolution(input_data)))
        l1a_out = nn.functional.relu(self.l1a_batch_norm(self.l1a_convolution(l1_out)))

        l1_out = l1_out[0].permute(2, 0, 1)
        l1_out = nn.functional.adaptive_max_pool1d(input=l1_out, output_size=1)
        l1_out = l1_out.reshape(l1_out.shape[0], -1)
        l1_out = l1_out.detach().clone()
        concatenated_tensors.append(l1_out)

        l1a_out = l1a_out[0].permute(2, 0, 1)
        l1a_out = nn.functional.adaptive_max_pool1d(input=l1a_out, output_size=1)
        l1a_out = l1a_out.reshape(l1a_out.shape[0], -1)
        l1a_out = l1a_out.detach().clone()
        concatenated_tensors.append(l1a_out)

        result = torch.cat(concatenated_tensors, 1).detach().clone()
        return result


if __name__ == "__main__":
    IMAGE_DIRECTORY_PATH = "Dataset/"

    FEATURE_EXTRACTOR_CHECKPOINT_PATH = "Checkpoints/"
    FEATURE_EXTRACTOR_CHECKPOINT_NAME = "segmenter_checkpoint.pth"

    whc_feature_extractor = WHCCRFeatureExtractor()
    cp_path_to_load = join(FEATURE_EXTRACTOR_CHECKPOINT_PATH, FEATURE_EXTRACTOR_CHECKPOINT_NAME)
    load_feature_extractor_checkpoint(cp_path_to_load, whc_feature_extractor)
    whc_feature_extractor.eval()

    features_loader = WHCSegmenterExtractedFeaturesLoader(IMAGE_DIRECTORY_PATH)

    features_batch, targets_batch = features_loader.next_train_batch(
        whc_feature_extractor, images_in_batch=3, cross_validation_set=0, output_device="cpu")
    print(f"\n\ntrain targets batch size {targets_batch.shape}")
    print(f"train features batch size {features_batch.shape}")

    features_batch, targets_batch = features_loader.next_test_batch(
        whc_feature_extractor, images_in_batch=3, cross_validation_set=0, output_device="cpu")
    print(f"\n\ntest targets batch size {targets_batch.shape}")
    print(f"test features batch size {features_batch.shape}")



