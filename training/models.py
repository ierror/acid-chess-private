import torch.nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_board_segmentation_model_instance(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class SquareClassificationModel(torch.nn.Module):
    def __init__(self):
        super(SquareClassificationModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 5, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, kernel_size=3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(5, 10, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(10, 10, kernel_size=3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(10, 20, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(20, 20, kernel_size=3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(20, 80, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(80, 80, kernel_size=3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.25),
            # torch.nn.Linear(3, 80),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(80, 3)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
