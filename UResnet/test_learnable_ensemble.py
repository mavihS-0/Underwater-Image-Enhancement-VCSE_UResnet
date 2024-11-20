import torch
import argparse
from torchvision import transforms
from PIL import Image
import os
import torchvision
from train_learnable_ensemble import LearnableEnsembleModel

# Test Function for the Learnable Ensemble Model
def test_ensemble(config):
    device = torch.device("cuda:" + str(config.cuda_id) if torch.cuda.is_available() else "cpu")

    # Load the saved ensemble model
    ensemble_model = torch.load(config.snapshot_pth).to(device)
    ensemble_model.eval()

    # Define image transformation
    tsfm = transforms.Compose([transforms.ToTensor()])

    # Ensure output directory exists
    os.makedirs(config.output_pth, exist_ok=True)

    # Process each image in the input directory
    for img_name in os.listdir(config.test_pth):
        if img_name.endswith((".png", ".jpg", ".jpeg", ".JPEG")):
            img_path = os.path.join(config.test_pth, img_name)

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            input_image = tsfm(image).unsqueeze(0).to(device)

            # Perform enhancement using the ensemble model
            with torch.no_grad():
                enhanced_image = ensemble_model(input_image)

            # Save the enhanced image
            output_path = os.path.join(config.output_pth, img_name.split('.')[0] + '-output.png')
            torchvision.utils.save_image(enhanced_image, output_path)

            print(f"Processed {img_name} and saved result as {output_path}")


# Main Function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0,help='default:0')
    parser.add_argument('--snapshot_pth',type=str,default=None,help='snapshot path,such as :xxx/snapshots/model.ckpt default:None')
    parser.add_argument('--test_pth',type=str,default='./data/test/',help='path of test images. default:./data/test/ ')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--output_pth',type=str,default='./results/',help='path to save generated image. default:./results/')
    parser.add_argument('--resize', type=int, default=256, help='Resize dimension for images')

    config = parser.parse_args()
    test_ensemble(config)
