from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensor
import albumentations as A
import cv2


transformAug = A.Compose([
    A.HorizontalFlip(p=0.9),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #A.RandomContrast(limit=0.2, p=1.0),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
    A.Cutout(num_holes=2, max_h_size=int(0.25*224), max_w_size=int(0.25*224), fill_value=0, always_apply=True, p=0.9),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

transformNorm = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
