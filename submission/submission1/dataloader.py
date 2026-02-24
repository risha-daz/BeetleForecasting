import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image

class VariableSizeTransform:
    """
    Resize the shortest side to `min_size` (preserving aspect ratio) then
    pad to a fixed canvas of `canvas_size × canvas_size`.

    This keeps every image's aspect ratio intact and avoids squishing, while
    still producing tensors of identical shape that can be stacked.
    """

    def __init__(self, min_size: int = 224, canvas_size: int = 224):
        self.min_size = min_size
        self.canvas_size = canvas_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize shortest side → min_size
        w, h = img.size
        scale = self.min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Clamp so neither side exceeds canvas
        if new_w > self.canvas_size or new_h > self.canvas_size:
            scale = self.canvas_size / max(new_w, new_h)
            new_w, new_h = int(new_w * scale), int(new_h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        tensor = self.to_tensor(img)  # (3, H, W)

        # Pad to canvas_size × canvas_size (bottom-right padding)
        pad_h = self.canvas_size - tensor.shape[1]
        pad_w = self.canvas_size - tensor.shape[2]
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)

        return self.normalize(tensor)


# Module-level transform — instantiated once, reused across all batches
_transform = VariableSizeTransform(min_size=224, canvas_size=224)


def beetle_collate_fn(batch):
    """
    Custom collate that handles:
    - Variable-size PIL images for beetle / colorpicker / scalebar
    - Stacks them into (B, 3, H, W) tensors
    - Passes through scalars and strings unchanged

    `batch` is a list of items yielded by the dataset __iter__:
        ([image, colorpicker_img, scalebar_img, scientificName, domainID])
    """
    beetle_imgs, color_imgs, scale_imgs = [], [], []
    sci_names, domain_ids = [], []

    for (image, colorpicker_img, scalebar_img, sci_name, domain_id) in batch:
        def to_tensor(x):
            if isinstance(x, torch.Tensor) and x.shape[-2:] == (224, 224):
                return x  # already correct size, skip transform
            if isinstance(x, torch.Tensor):
                x = TF.to_pil_image(x)
            return _transform(x)

        beetle_imgs.append(to_tensor(image))
        color_imgs.append(to_tensor(colorpicker_img))
        scale_imgs.append(to_tensor(scalebar_img))
        sci_names.append(sci_name)
        domain_ids.append(domain_id)

    beetle_imgs = torch.stack(beetle_imgs)  # (B, 3, 224, 224)
    color_imgs  = torch.stack(color_imgs)   # (B, 3, 224, 224)
    scale_imgs  = torch.stack(scale_imgs)   # (B, 3, 224, 224)

    # Pin memory for faster CPU→GPU transfers
    if torch.cuda.is_available():
        beetle_imgs = beetle_imgs.pin_memory()
        color_imgs  = color_imgs.pin_memory()
        scale_imgs  = scale_imgs.pin_memory()

    return (
        beetle_imgs,
        color_imgs,
        scale_imgs,
        sci_names,   # list[str]
        domain_ids,  # list[str]
    )


def get_sentinel_beetles_loader_with_collate(datapoints):
    """
    Convert a list of beetle dictionaries into model-ready tensors.
    
    Parameters
    ----------
    datapoints : list of dict
        Each dict must contain:
        - relative_image (tensor)
        - colorpicker_img (tensor)
        - scalebar_img (tensor)
        - scientificName
        - domainID

    Returns
    -------
    tuple
        (beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids)
    """
    batch = []

    for example in datapoints:
        batch.append((
            example["relative_img"],
            example["colorpicker_img"],
            example["scalebar_img"],
            example["scientificName"],
            example["domainID"],
        ))

    return beetle_collate_fn(batch)