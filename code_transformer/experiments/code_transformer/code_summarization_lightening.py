import os
import signal
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.experiments.mixins.code_summarization import CTCodeSummarizationMixin
from code_transformer.experiments.mixins.code_trans_transformer import CodeTransformerDecoderMixin
from code_transformer.experiments.log import ExperimentLogger, TensorboardLogger
from code_transformer.preprocessing.datamanager.base import batch_filter_distances

from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.env import DATA_PATH_STAGE_2
from code_transformer.modeling.constants import PAD_TOKEN, UNKNOWN_TOKEN, NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.preprocessing.dataset.code_summarization import \
    CTCodeSummarizationDatasetNoPunctuation


class CodeTransDecoderExperimentSetup(CodeTransformerDecoderMixin,
                                      CTCodeSummarizationMixin,
                                      ExperimentSetup):
    pass


class CodeTransformerDataModule(LightningDataModule):
    def __init__(self, experiment, **kwargs):
        super().__init__()
        self.experiment = experiment 
        self.get_training_hparams()
        self.experiment._init_metrics(self.metrics)
        #self.test_data_manager = CTBufferedDataManager(
        #    DATA_PATH_STAGE_2,
        #    'python,javascript,ruby,go',
        #    partition='test',
        #    shuffle=False,
        #    filter_language=None
        #    )
        #self.dataset = CTCodeSummarizationDatasetNoPunctuation(
        #    self.test_data_manager,
        #    num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
        #    use_pointer_network=True,
        #    max_num_tokens=1000
        #    )

    @ex.capture(prefix="training")
    def get_training_hparams(
        self, batch_size, simulated_batch_size,
        random_seed, metrics, validate_every=None,
        persistent_snapshot_every=None, simulated_batch_size_valid=None, early_stopping_patience=10,
        max_validation_samples=10000, accumulate_tokens_batch=False):

        self.batch_size = batch_size
        self.simulated_batch_size = simulated_batch_size
        self.random_seed = random_seed
        self.metrics = metrics
        self.validate_every = None
        self.persistent_snapshot_every = None

    def train_dataloader(self):
        dataloader = DataLoader(
            self.experiment.dataset_train,
            batch_size=self.batch_size,
            collate_fn=self.experiment.dataset_train.collate_fn)
        return dataloader 

    def val_dataloader(self):
        dataloader_validation = DataLoader(
            self.experiment.dataset_validation, batch_size=self.batch_size,
            collate_fn=self.experiment.dataset_validation.collate_fn)
        #dataloader_validation = DataLoader(
        #    self.dataset, collate_fn=self.dataset.collate_fn, batch_size=4)
        return dataloader_validation


class CodeTransformerModule(LightningModule):
    def __init__(self, experiment, **kwargs):
        super().__init__()
        self.experiment = experiment 
        self.model = self.experiment.model_lm
    
    @ex.capture(prefix="optimizer")
    def configure_optimizers(self, learning_rate, reg_scale):
        #if self.experiment.scheduler is not None:
        #    return {
        #        "optimizer": self.experiment.optimizer,
        #        "lr_scheduler": self.experiment.scheduler 
        #    }
        #else:
        #    return self.experiment.optimizer
        from transformers import AdamW, get_linear_schedule_with_warmup
        fast_weight = ['alpha', 'gamma']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in fast_weight)],
            'lr': learning_rate,
            'weight_decay': 0.01 
            },
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in fast_weight)], 
            'lr': 0.5 * learning_rate,
            'weight_decay': 0.01 
            }
        ]
        optimizer = AdamW(
            #self.model.parameters(),
            optimizer_grouped_parameters,
            lr=learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=2000, num_training_steps=150000)
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }
        print('build {} with learning rate {}'.format(optimizer, learning_rate))
        return [optimizer], [lr_scheduler] 

    def forward(self, inputs):
        return self.model.forward_batch(inputs)
    
    def training_step(self, batch, batch_idx):
        batch = batch_filter_distances(batch, self.experiment.relative_distances)
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('training_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        validation_batch = batch_filter_distances(batch, self.experiment.relative_distances)
        output = self.forward(batch).cpu()
        evaluation = self.experiment._evaluate_predictions(
            output.logits, batch.labels.cpu(),
            loss=output.loss, partition='valid')
        for k, v in evaluation.items():
            self.log(k, v, on_step=True, on_epoch=True, sync_dist=True)
        return evaluation
    

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


@ex.automain
def main():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    #root_path = 'pl_experiments/rank1_encoder_agb4_normal0.05/'
    #root_path = 'pl_experiments/rank1_encoder_agb4_normal0.5/'
    #root_path = 'pl_experiments/rank1_encoder_agb4_uniform1/'
    #root_path = 'pl_experiments/full_rank1_uniform1/'
    root_path = 'pl_experiments/fullrank1_wpointer_agb2_uniform1_wd0.01/'
    #root_path = 'pl_experiments/fullrank1_wpointer_agb2_normal0.5_wd0.01/'
    #root_path = 'pl_experiments/fullrank1_wpointer_agb2_normal0.5/'
    #root_path = 'pl_experiments/base_agb2/'
    #root_path = 'pl_experiments/rank1_test/'
    tb_logger = pl_loggers.TensorBoardLogger(root_path + "tb_logs/")
    csv_logger = pl_loggers.CSVLogger(root_path + "csv_logs/")
    #experiment.train()
    seed_everything(42)

    experiment = CodeTransDecoderExperimentSetup()
    dm = CodeTransformerDataModule(experiment) 
    model = CodeTransformerModule(experiment) 
    # Ensure graceful shutdown when training is interrupted
    signal.signal(signal.SIGINT, experiment._handle_shutdown)
    trainer = Trainer(
        checkpoint_callback=True,
        default_root_dir=root_path,
        #gpus=1,
        gpus=4,
        strategy='ddp',
        #val_check_interval=0.01,
        #val_check_interval=10,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
        logger=[tb_logger, csv_logger],
        callbacks=[CheckpointEveryNSteps(2500)],
        flush_logs_every_n_steps=100
    )
    trainer.fit(model, dm)


@ex.command(unobserved=True)
def recreate_experiment():
    return CodeTransDecoderExperimentSetup()