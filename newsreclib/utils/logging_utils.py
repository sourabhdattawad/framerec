from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from newsreclib.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters

    Args:
        object_dict: Dictionary of hyperparameters for each module in the pipeline.
    """

    def remove_exp_name(d):
        if isinstance(d, dict):
            d = {k: remove_exp_name(v) for k, v in d.items() if k != "exp_name"}
        elif isinstance(d, list):
            d = [remove_exp_name(i) for i in d]
        return d

    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # sanitize config first
    cfg = remove_exp_name(cfg)

    hparams["model"] = cfg["model"]

    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

