import os.path

from model.utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from pytorch_lightning.strategies.ddp import DDPStrategy


if __name__ == "__main__":
    batch_size = 128
    devices = 4
    test_at_home = False
    parent_dir_for_validation = "/scratch/dataCaFFeBig/data_raw/"
    config_data = "randomCrop512.yaml"
    config_model = "Tyrion.yaml"
    accumulated_grad_batches = 2

    batch_size = int(batch_size / (devices * accumulated_grad_batches))

    apply_data = "Data"
    apply_model = "model"
    log_dir = "tyrion"

    target_loss = "val/IoU_img"
    problem_type = "max"

    if len(sys.argv) == 1:
        test_at_home = True
        config_data = "work.yaml"
        config_model = "Tyrion.yaml"
        batch_size = 1
        parent_dir_for_validation = "C:\\Users\\nora_admin\\PycharmProjects\\data_raw"

    gdm = instantiate_completely(apply_data, config_data, batch_size=batch_size)

    model = instantiate_completely(apply_model, config_model, parent_dir_for_validation=parent_dir_for_validation)

    args = OmegaConf.load(get_yaml(apply_model, config_model))

    logger = TensorBoardLogger(save_dir=args["log_dir"], name=config_model[:-5] + "_" + config_data[:-5])
    print("********************************************************************************")
    print("Dataloader file: ", config_data)
    print("Model file: ", config_model)
    print("********************************************************************************")

    target_loss2 = "val/mde"
    mc = ModelCheckpoint(save_top_k=1, monitor=target_loss, mode=problem_type,
                         filename='{epoch}-{' + target_loss + ':.4f}')
    mc2 = ModelCheckpoint()
    mc3 = ModelCheckpoint(save_top_k=1, monitor=target_loss2, mode="min", filename='{epoch}-{' + target_loss2 + ':.4f}')
    cb = [mc, mc2, mc3]

    if test_at_home:
        trainer = pl.Trainer(max_epochs=150, accelerator="gpu", devices=1, logger=logger,
                             accumulate_grad_batches=accumulated_grad_batches, callbacks=cb, num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(max_epochs=150, accelerator="gpu", devices=devices, logger=logger,
                             strategy=DDPStrategy(find_unused_parameters=True),
                             accumulate_grad_batches=accumulated_grad_batches, callbacks=cb)

    if args.get("ckpt") is not None and args["resume_training"]:
        ckpt_optimizers = args["ckpt"]
    else:
        ckpt_optimizers = None

    trainer.fit(model, train_dataloaders=gdm.train_dataloader(), val_dataloaders=gdm.val_dataloader(),
                ckpt_path=ckpt_optimizers)
