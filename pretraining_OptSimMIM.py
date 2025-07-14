from model.utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy


def extract_dict(config):
    return {k: v for k, v in config.items() if k != "config"}


def get_dict_from_model_config(config):
    dict_1 = extract_dict(config["params"])
    return dict_1


if __name__ == "__main__":
    max_epochs = 1000
    test_at_home = False
    batch_size = 32
    devices = 4
    config_data = "OptSimMIM.yaml"
    config_model = "OptSimMIM.yaml"
    accumulated_grad_batches = 2
    batch_size = int(batch_size / (devices * accumulated_grad_batches))

    apply_data = "Data"
    apply_model = "model"
    log_dir = "OptSimMIM"

    if len(sys.argv) == 1:
        test_at_home = True
        config_data = "work_OptSimMIM.yaml"
        batch_size = 2

    gdm = instantiate_completely(apply_data, config_data, batch_size=batch_size)

    SimMIM = instantiate_completely(apply_model, config_model)

    file = get_yaml(apply_model, config_model)
    config = OmegaConf.load(file)
    config_dict_model = get_dict_from_model_config(config)
    file = get_yaml(apply_data, config_data)
    config = OmegaConf.load(file)

    logger = TensorBoardLogger(save_dir=log_dir, name=config_model[:-5] + "_" + config_data[:-5])
    logger.log_hyperparams({**config_dict_model})
    print("********************************************************************************")
    print("Dataloader file: ", config_data)
    print("Model file: ", config_model)
    print("********************************************************************************")
    target_loss = "val/loss"
    mc = ModelCheckpoint(save_top_k=1, monitor=target_loss, mode="min", filename='{epoch}-{' + target_loss + ':.4f}')
    mc2 = ModelCheckpoint(save_last=True, filename='{epoch}-{' + target_loss + ':.4f}')

    cb = [mc, mc2]
    args = OmegaConf.load(get_yaml(apply_model, config_model))

    if args.get("ckpt") is not None:
        ckpt_optimizers = args["ckpt"]
    else:
        ckpt_optimizers = None

    if test_at_home:
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1, logger=logger,
                             accumulate_grad_batches=accumulated_grad_batches, callbacks=cb, num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=devices, logger=logger,
                             strategy=DDPStrategy(find_unused_parameters=True),
                             accumulate_grad_batches=accumulated_grad_batches, callbacks=cb, sync_batchnorm=True)
    trainer.fit(SimMIM, train_dataloaders=gdm.train_dataloader(), val_dataloaders=gdm.val_dataloader(),
                ckpt_path=ckpt_optimizers)
    logger.finalize("success")
