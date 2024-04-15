import torch
import time
from OriginalTransformer import DataModule, Transformer
import pytorch_lightning as pl
from IPython.utils import io
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

epochs = 100

def run_training_job(random_state):
    torch.manual_seed(random_state)

    dm = DataModule()
    dm.prepare_data()

    transformer_model = Transformer(len(dm.german_vocab), len(dm.english_vocab), dm.return_label_smoothing_criterion())

    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = pl.Trainer(
        logger=False,
        max_epochs=epochs,
        enable_progress_bar=False,
        default_root_dir='logs',
        #accumulate_grad_batches=10,
        #callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    )

    # Train the model and log time ⚡
    start = time.time()
    trainer.fit(transformer_model, dm)
    num_epochs = trainer.current_epoch
    end = time.time()
    train_time = end - start
    print("Training completed in {} epochs.".format(num_epochs))

    # Compute the validation accuracy once and log the score
    with io.capture_output() as captured:
        val_loss = trainer.validate(datamodule=dm)[0]["val_loss"]
    print(f"train time: {train_time}, val loss: {val_loss}")#, num_params: {num_params}")

    return val_loss



if __name__ == "__main__":
    run_training_job(random_state=0)

