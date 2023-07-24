import torch
import pickle
import os
import numpy as np
from vdzh.data.visdrone import VisDroneTestSet
from vdzh.models.owlvit import OwlViT
from PIL import ImageDraw, Image
from tqdm import tqdm
from ezdl.transforms import Denormalize

from transformers import OwlViTProcessor
from albumentations import Compose, Resize

def get_model(model_name, model_params, device):
    model = OwlViT(model_name, model_params)
    model = model.to(device)
    model.eval()
    return model


def detect_testset(model_params, dataset_params, output, device="cuda", threshold=0.1):
    os.makedirs(output, exist_ok=True)
    params = f"threshold_{threshold}"
    outdir = os.path.join(output, params)
    os.makedirs(os.path.join(output, params), exist_ok=True)
    image_dir = os.path.join(output, params, "images")
    os.makedirs(image_dir, exist_ok=True)

    model = get_model(model_params, model_params, device)
    transforms = Compose([
        Resize(768, 768)
    ])
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    def transform_fn(sample):
        image_ids = sample["image_id"]
        text = VisDroneTestSet.classes
        outs = transforms(image=sample['image'])
        inputs = processor(images=outs['image'], text=text, return_tensors="pt", padding=True)
        return (inputs['input_ids'], inputs["pixel_values"], inputs["attention_mask"]), {'input_name': sample['input_name']}
    
    
    testset = VisDroneTestSet(**dataset_params, transforms=transform_fn)
    testset.sample_list.sort()
    denorm = Denormalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)

    res_boxes = []
    for sample in tqdm(testset):
        (ids, pixel_values, attention_mask), additionals = sample
        ids, pixel_values, attention_mask = ids.to(device), pixel_values.to(device), attention_mask.to(device)
        outputs = model((ids, pixel_values, attention_mask))
        target_sizes = torch.tensor([pixel_values.shape[::-1][:-2]], device=device)
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = VisDroneTestSet.classes
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        unnormalized = (denorm(pixel_values) * 255).type(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
        draw = ImageDraw.Draw(Image.fromarray(unnormalized))
        
        img_res_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                xmin, ymin, xmax, ymax = box
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                res_box = torch.tensor([[xmin, ymin, xmax, ymax, score]]).numpy().astype("float32")
                draw.text((xmin, ymin), f"{label}: {round(score.item(),2)}", fill="white")
                box = [round(i, 2) for i in box.tolist()]
                draw._image.save(os.path.join(image_dir, f"{additionals['input_name']}"))
                img_res_boxes.append(res_box)
        res_boxes.append(img_res_boxes)

    lens = np.array([len(img) for img in res_boxes])
    max_len = lens.max()
    max_len = max_len if max_len > 20 else 20
    res_boxes = [
        img + [np.array([], dtype="float32")] * (max_len - len(img)) if len(img) < max_len else img[:max_len]
        for img in res_boxes
    ]
        
    f = open(os.path.join(outdir, "detection_results_gzsd.pkl"), "wb")
    pickle.dump(res_boxes, f)
    f.close()
