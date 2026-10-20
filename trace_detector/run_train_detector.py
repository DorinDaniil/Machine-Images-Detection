"""Train TraceDetector on aligned pairs.

    python -m trace_detector.run_train_detector --config trace_detector/configs/train_config_detector.yaml
"""

import argparse

import torch

from .train_detector import build_loaders, build_model, load_config, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--pairs-csv", default=None, help="override data.pairs_csv")
    parser.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pairs_csv:
        cfg["data"]["pairs_csv"] = args.pairs_csv
    torch.manual_seed(cfg.get("seed", 0))

    model = build_model(cfg)
    train_loader, val_loader = build_loaders(cfg, model)
    print(f"train pairs: {len(train_loader.dataset)}  val items: {len(val_loader.dataset)}")
    best = train_model(model, train_loader, val_loader, cfg, resume=args.resume)
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
