import inspect

from tqdm import tqdm
import wandb
from src.pipeline import *
from src.utils import PropertyDict
from src.config import Config
from transformers import EncoderDecoderModel, AutoTokenizer
import torch

CONFIG = Config()

hyperparameters = PropertyDict(
    seed=42,
    checkpoint_name="bert_tiny",
    model_name="bert_tiny",
    model_type="encoder_decoder",
    initialize_cross_attention=True,
    yng_loss_weight=0.6,
    rationale_loss_weight=0.8,
    generative_loss_weight=0.2,
    batch_size=32,
    val_batch_size=64,
    generate_batch_size=32,
    num_workers=2,
    num_epochs=3,
    optimizer_name="AdamW",
    learning_rate=2e-4,
    scheduler="linear",
    warmup_fraction=0.1,
    teacher_force_scheduler="linear",
    tf_start = 1.,
    tf_end = 0.,
    tf_fraction = 0.6,
    accumulation_steps=1,
    gradient_clip=1.0,
    mixed_precision="fp16",
    checkpoint_interval=700,
    log_interval=700,
    cpu=False,
)

with wandb.init(project=CONFIG.wandbConfig.project, config=hyperparameters):
    config = wandb.config

    set_seed(config.seed)

    # Make the model
    tokenizer = make_tokenizer(config)
    model = make_model(config, tokenizer)

    # Make the data
    train_data = get_data("train", config)
    val_data = get_data("validation", config)
    train_dataloader = make_dataloader(train_data, tokenizer, config, split="train")
    val_dataloader = make_dataloader(val_data, tokenizer, config, split="validation")

    # Make the loss, the optimizer and the scheduler
    loss_fn = make_loss(config)
    optimizer = make_optimizer(model, loss_fn, config)
    scheduler = make_scheduler(
        optimizer, steps_per_epoch=len(train_dataloader), config=config
    )
    tf_scheduler = make_teacher_force_scheduler(steps_per_epoch=len(train_dataloader), config=config)

    # model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, metrics = make(config)
    print(model)

    train(
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        scheduler,
        config,
        teacher_force_scheduler=tf_scheduler,
    )

    torch.save(model.state_dict(), "checkpoints/bert_tiny_42.pt")