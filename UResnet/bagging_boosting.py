import torch
import argparse
from torchvision import transforms
from PIL import Image
import os
import torchvision
import torch.nn as nn

# Bagging Ensemble Model
class BaggingEnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(BaggingEnsembleModel, self).__init__()
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def forward(self, x):
        outputs = [weight * model(x)[0] for model, weight in zip(self.models, self.weights)]
        ensemble_output = sum(outputs)  # Weighted sum of outputs
        return ensemble_output

# Boosting Ensemble Model
class BoostingEnsembleModel(nn.Module):
    def __init__(self, models, alphas):
        super(BoostingEnsembleModel, self).__init__()
        self.models = models
        self.alphas = alphas

    def forward(self, x):
        outputs = [alpha * model(x)[0] for model, alpha in zip(self.models, self.alphas)]
        ensemble_output = sum(outputs)  # Weighted sum of outputs
        return ensemble_output

# Test function
def test(config):
    device = torch.device("cuda:" + str(config.cuda_id) if torch.cuda.is_available() else "cpu")

    # Load models and move them to the device
    model1 = torch.load(config.snapshot_pth1).to(device)
    model2 = torch.load(config.snapshot_pth2).to(device)
    model3 = torch.load(config.snapshot_pth3).to(device)

    model1.eval()
    model2.eval()
    model3.eval()

    models = [model1, model2, model3]

    # Example weights for Bagging and alphas for Boosting
    bagging_weights = [0.3, 0.4, 0.3]
    boosting_alphas = [0.5, 0.3, 0.2]

    # Create Bagging and Boosting ensemble models
    bagging_model = BaggingEnsembleModel(models, bagging_weights).to(device)
    boosting_model = BoostingEnsembleModel(models, boosting_alphas).to(device)

    # Define image transformation
    tsfm = transforms.Compose([transforms.ToTensor()])

    # Ensure output directory exists
    os.makedirs(config.output_pth, exist_ok=True)

    # Process each image in the input directory
    for img_name in os.listdir(config.test_pth):
        if img_name.endswith((".png", ".jpg", ".jpeg", '.JPEG')):
            img_path = os.path.join(config.test_pth, img_name)

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            input_image = tsfm(image).unsqueeze(0).to(device)

            # Perform enhancement using Bagging and Boosting models
            with torch.no_grad():
                bagging_output = bagging_model(input_image)
                boosting_output = boosting_model(input_image)

            # Save the enhanced images
            bagging_output_path = os.path.join(config.output_pth, img_name.split('.')[0] + '-bagging-output.png')
            boosting_output_path = os.path.join(config.output_pth, img_name.split('.')[0] + '-boosting-output.png')

            torchvision.utils.save_image(bagging_output, bagging_output_path)
            torchvision.utils.save_image(boosting_output, boosting_output_path)

            print(f"Processed {img_name} with Bagging and Boosting ensembles. Saved results.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0, help='default:0')
    parser.add_argument('--snapshot_pth1', type=str, required=True, help='path to snapshot of model1')
    parser.add_argument('--snapshot_pth2', type=str, required=True, help='path to snapshot of model2')
    parser.add_argument('--snapshot_pth3', type=str, required=True, help='path to snapshot of model3')
    parser.add_argument('--test_pth', type=str, default='./input/', help='path of test images')
    parser.add_argument('--output_pth', type=str, default='./results/', help='path to save generated image')

    config = parser.parse_args()
    test(config)
