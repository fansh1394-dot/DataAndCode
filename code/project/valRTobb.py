"""RTDETRMM OBB validation entry script."""

from ultralytics import RTDETRMM


if __name__ == "__main__":
    model = RTDETRMM("runs/obb/train/weights/best.pt")
    model.val(data="data.yaml", split="test", device=0)