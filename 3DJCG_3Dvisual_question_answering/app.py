from uuid import uuid4
from pathlib import Path
from typing import Optional

import aiofiles
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image

# from .models import ImageEntityRecognition
from .model_util_scannet import ScannetDatasetConfig
from models.vqanet.vqanet import VqaNet

app = FastAPI()
model = None
DC = ScannetDatasetConfig()
# model: Optional[ImageEntityRecognition] = None


# def get_model() -> ImageEntityRecognition:
#     global model

#     if model is None:
#         model = ImageEntityRecognition()

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device)
#         model.eval()

#     return model

def get_model(args):
    global model

    if model is None:
        # initiate model
        input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
        model = VqaNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            no_reference=args.no_reference,
            dataset_config=DC
        )

        # trainable model
        if args.use_pretrained:
            # load model
            print("loading pretrained VoteNet...")

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            load_result = model.load_state_dict(torch.load(pretrained_path), strict=False)
            print(load_result, flush=True)

            # mount
            # model.backbone_net = pretrained_model.backbone_net
            # model.vgen = pretrained_model.vgen
            # model.proposal = pretrained_model.proposal

            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False

                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False

                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False

        # to CUDA
        model = model.cuda()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-3)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--use_generated_dataset", action="store_true", help="Use Generated Dataset.")
    parser.add_argument("--use_generated_masked_dataset", action="store_true", help="Use Generated Masked Dataset.")
    args = parser.parse_args()
    get_model(args)


@app.get("/")
async def root():
    return {"message": "Hello World"}


async def save_upload_file(upload_file: UploadFile) -> str:
    file_path = f"uploaded_images/{uuid4()}{Path(upload_file.filename).suffix}"
    async with aiofiles.open(file_path, "wb") as f:
        while content := await upload_file.read(4 * 1024):
            await f.write(content)

    return file_path


@app.post("/inference")
async def inference(image: UploadFile):
    image_path = await save_upload_file(image)
    image = Image.open(image_path).convert("RGB")

    model = get_model()
    label_id, label, prob = model.inference(image)

    return {
        "label_id": label_id,
        "label": label,
        "prob": prob,
    }