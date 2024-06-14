import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,BitsAndBytesConfig


def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class GReaTTrainer(Trainer):
    """GReaT Trainer

    Overwrites the get_train_dataloader methode of the HuggingFace Trainer to not remove the "unused" columns -
    they are needed later!
    """
    bnb_config = BitsAndBytesConfig(load_in_4bit= True,bnb_4bit_quant_type= "nf4",bnb_4bit_compute_dtype= torch.float16,bnb_4bit_use_double_quant= True)
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = (
            self.train_dataset
        )  # self._remove_unused_columns(self.train_dataset, description="training")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )
