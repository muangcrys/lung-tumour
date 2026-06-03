from huggingface_hub import snapshot_download
from pathlib import Path
from models.vivit import _model_on_hub, _vivit_dir


def main():
    print(f"Downloading ViViT from {_model_on_hub}.")
    print(f"Save target to {_vivit_dir}.")
    _vivit_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=_model_on_hub,
        local_dir=_vivit_dir,
        local_dir_use_symlinks=False,
    )
    print("OK")

if __name__ == "__main__":
    main()