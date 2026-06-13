from pathlib import Path

_checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints"
_r3d_dir = _checkpoint_dir / "resnet3d"
_medicalnet_dir = _checkpoint_dir / "medicalnet"

class R3d18Config:
    pretrained_classes = 1039               # Kinetics + Motions
    shortcut = 'B'
    ckt = _r3d_dir / "r3d18_KM_200ep.pth"

class R3d50Config:
    pretrained_classes = 1139  # Kinetics, Motions, STAIR
    shortcut = 'B'
    ckt = _r3d_dir / "r3d50_KMS_200ep.pth"

class R3d34Config:
    pretrained_classes = 1039
    shortcut = 'B'
    ckt = _r3d_dir / "r3d34_KM_200ep.pth"

class MedicalNet18Config:
    ckt = _medicalnet_dir / "resnet_18_23dataset.pth"
    shortcut = 'A'

class MedicalNet34Config:
    ckt = _medicalnet_dir / "resnet_34_23dataset.pth"
    shortcut = 'A'

class MedicalNet50Config:
    ckt = _medicalnet_dir / "resnet_50_23dataset.pth"
    shortcut = 'B'