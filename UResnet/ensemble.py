import torch
import argparse
from torchvision import transforms
from PIL import Image
import os
import torchvision
import torch.nn as nn

# Define the ensemble model that combines outputs of the three models
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        output1, _ = self.model1(x)
        output2, _ = self.model2(x)
        output3, _ = self.model3(x)
        # Averaging the outputs from the three models
        ensemble_output = (output1 + output2 + output3) / 3
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

    # Create the ensemble model
    ensemble_model = EnsembleModel(model1, model2, model3).to(device)

    # Define image transformation
    tsfm = transforms.Compose([transforms.ToTensor()])

    # Ensure output directory exists
    os.makedirs(config.output_pth, exist_ok=True)

    # Process each image in the input directory
    for img_name in os.listdir(config.test_pth):
        if img_name.endswith((".png", ".jpg", ".jpeg",'.JPEG')):
            img_path = os.path.join(config.test_pth, img_name)
            
            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            input_image = tsfm(image).unsqueeze(0).to(device)

            # Perform enhancement using the ensemble model
            with torch.no_grad():
                enhanced_output = ensemble_model(input_image)

            # Save the enhanced image
            output_image_path = os.path.join(config.output_pth, img_name.split('.')[0] + '-output.png')
            torchvision.utils.save_image(enhanced_output, output_image_path)
            print(f"Processed {img_name} and saved enhanced image to {output_image_path}")

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
