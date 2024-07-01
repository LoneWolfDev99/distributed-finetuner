import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, texts, labels):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    dataset = CustomDataset(tokenizer, texts, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # We will use the DataLoader instead
        eval_dataset=None,  # Optionally, set your eval_dataset here
        data_collator=lambda data: {key: torch.stack([f[key] for f in data]) for key in data[0]},
    )

    # Manually set the DataLoader in the Trainer's DataLoader
    trainer.train_dataloader = dataloader

    # Start training
    trainer.train()
    cleanup()

if __name__ == "__main__":
    texts = ["Example sentence 1", "Example sentence 2", "Example sentence 3"]
    labels = [0, 1, 0]

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, texts, labels), nprocs=world_size, join=True)
