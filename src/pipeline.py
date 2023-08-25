from collections import defaultdict
import gc
import inspect
import os
import torch
import torch.nn as nn

from tqdm import tqdm


import wandb
import transformers
import datasets
from accelerate import Accelerator

from typing import Dict, Tuple, Union

from .train import DummyScheduler, DynamicPaddingCollatorForSeq2Seq, load_checkpoint, save_checkpoint
from .evaluation import evaluate_model
from .losses import ComputeLoss, EncoderDecoderLoss, EncoderRationaleLoss
from .models import make_encoder_decoder_model, make_qa_encoder
from .utils import (
    AvgValue,
    create_dirs_for_file,
    create_reproducible_dataloader,
    set_seed,
)
from .config import Config

CONFIG: Config = Config()


def pipeline(hyperparameters: dict):
    with wandb.init(**CONFIG.wandbConfig.__dict__, config=hyperparameters):
        config = wandb.config

        set_seed(config.seed)

        (
            model,
            tokenizer,
            train_data,
            val_data,
            train_dataloader,
            val_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            # metrics,
        ) = make(config)
        print(model)

        train(
            model,
            train_dataloader,
            val_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            config,
            # metrics=metrics,
        )

        evaluate(model, tokenizer, train_data, val_data, config)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def make(config):
    # Make the model
    tokenizer, model = make_model(config)

    # Make the data
    train_data, val_data = get_data("train", config), get_data("validation", config)
    train_dataloader = make_dataloader(train_data, tokenizer, config, split="train")
    val_dataloader = make_dataloader(val_data, tokenizer, config, split="validation")

    # Make the loss, the optimizer and the scheduler
    loss_fn = make_loss(config)
    optimizer = make_optimizer(model, loss_fn, config)
    scheduler = make_scheduler(
        optimizer, steps_per_epoch=len(train_dataloader), config=config
    )

    # # Make the evaluation metrics
    # metrics = make_metrics(tokenizer, config)

    return (
        model,
        tokenizer,
        train_data,
        val_data,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        scheduler,
        # metrics,
    )


def make_model(config):
    checkpoint = CONFIG.checkpoints.__dict__[config.checkpoint_name]
    if config.model_type == "encoder_decoder":
        return make_encoder_decoder_model(
            checkpoint=checkpoint,
            decoder_max_length=CONFIG.decoder_max_length,
            generation_kwargs=CONFIG.generation,
            initialize_cross_attention=config.initialize_cross_attention,
        )
    if config.model_type == "encoder":
        return make_qa_encoder(checkpoint=checkpoint)

    raise ValueError(
        "Invalid model_type. Supported values are 'encoder_decoder' and 'encoder'."
    )


def get_data(split: str, config):
    if config.get("history_length", 0) > 0:
        path = CONFIG.dataset.train_with_history(config.checkpoint_name, split=split)
    else:
        path = CONFIG.dataset.train_no_history(config.checkpoint_name, split=split)

    dataset = datasets.load_from_disk(path)
    dataset = dataset.remove_columns(
        ["id", "turn", "offset_mapping", "rationale_start", "rationale_end"]
    )
    return dataset


def make_dataloader(dataset, tokenizer, config, split: str):
    data_collator = DynamicPaddingCollatorForSeq2Seq(tokenizer)
    dataloader = create_reproducible_dataloader(
        dataset,
        batch_size=config.val_batch_size
        if split != "train" and "val_batch_size" in config
        else config.batch_size,
        collate_fn=data_collator,
        num_workers=config.val_num_workers
        if split != "train" and "val_num_workers" in config
        else config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def make_loss(config) -> ComputeLoss:
    if config.model_type == "encoder_decoder":
        # loss = composite_loss(
        #     Criterion(
        #         "rationale_loss", encoder_decoder_rationale_loss, weight=config.rationale_loss_weight
        #     ),
        #     Criterion(
        #         "generative_loss", encoder_decoder_generative_loss, weight=config.generative_loss_weight
        #     ),
        # )
        # loss = UncertaintyLoss(
        #     Criterion(
        #         "rationale_loss",
        #         EncoderDecoderRationaleLoss(
        #             max_rationale_length=CONFIG.rationale_max_length
        #         ),
        #         weight=config.rationale_loss_weight,
        #     ),
        #     Criterion(
        #         "generative_loss",
        #         encoder_decoder_generative_loss,
        #         weight=config.generative_loss_weight,
        #     ),
        # )

        # loss = CompositeLoss(
        #     Criterion(
        #         "rationale_loss",
        #         UncertaintyLoss(
        #             EncoderDecoderRationaleLoss(
        #                 max_rationale_length=CONFIG.rationale_max_length
        #             ),
        #             initial_weight=config.rationale_loss_weight,
        #         ),
        #     ),
        #     Criterion(
        #         "generative_loss",
        #         UncertaintyLoss(
        #             encoder_decoder_generative_loss,
        #             initial_weight=config.generative_loss_weight,
        #         ),
        #     ),
        # )

        loss = EncoderDecoderLoss(
            max_rationale_length=CONFIG.rationale_max_length,
            yng_loss_weight=config.yng_loss_weight,
            rationale_loss_weight=config.rationale_loss_weight,
            generative_loss_weight=config.generative_loss_weight,
        )

    elif config.model_type == "encoder":
        loss = EncoderRationaleLoss(max_rationale_length=CONFIG.rationale_max_length)
    else:
        raise ValueError(
            "Invalid model_type. Supported values are 'encoder_decoder' and 'encoder'."
        )
    return loss


def make_optimizer(model, loss_fn, config):
    optimizer_cls = getattr(torch.optim, config.optimizer_name)
    parameters = [{"params": model.parameters()}]
    if hasattr(loss_fn, "_parameters"):
        loss_params = {"params": loss_fn.parameters()}
        if "loss_learning_rate" in config:
            loss_params["lr"] = config.loss_learning_rate
        parameters.append(loss_params)

    return optimizer_cls(
        parameters,
        lr=config.learning_rate,
        **config.get("optimizer_args", {}),
    )


def make_scheduler(optimizer, steps_per_epoch, config):
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(config.warmup_fraction * total_steps)
    if config.get("scheduler", "none") != "none":
        return transformers.get_scheduler(
            config.scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    return DummyScheduler(optimizer=optimizer)


# def make_metrics(tokenizer, config) -> Dict[str, Metric]:
#     if config.model_type not in ["encoder_decoder", "encoder"]:
#         raise ValueError(
#             "Invalid model_type. Supported values are 'encoder_decoder' and 'encoder'."
#         )

#     metrics = {}
#     metrics["encoder"] = {
#         "yng_accuracy": EncoderYNGAccuracy(),
#         "yng_f1": EncoderYNGF1(),
#         "rationale_accuracy": EncoderRationaleAccuracy(),
#         "rationale_f1": EncoderRationaleF1(),
#     }
#     if config.model_type == "encoder_decoder":
#         metrics["encoder_decoder"] = {
#             "squad_f1": GenerativeSquadF1(tokenizer),
#         }

#     return metrics


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: Union[ComputeLoss, nn.Module],
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    config,
    teacher_force_scheduler = None,
    # metrics: Dict[str, Metric] = {},
):
    watch_list = [model]

    accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.cpu)
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if isinstance(loss_fn, nn.Module):
        watch_list.append(loss_fn)
        loss_fn = accelerator.prepare(loss_fn)

    wandb.watch(watch_list, log="all", log_freq=config.log_interval)

    # Run training and track with wandb
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.num_epochs

    checkpoint_counter = 0
    step = 0
    avg_loss = AvgValue()
    avg_inner_losses = defaultdict(AvgValue)
    model.train()

    forward_signature = set(inspect.signature(model.forward).parameters)
    progress_bar = tqdm(range(total_steps))
    for epoch in range(config.num_epochs):
        for data in train_dataloader:
            inputs = {
                argument: value
                for argument, value in data.items()
                if argument in forward_signature
            }

            loss, inner_losses = train_batch(
                inputs=inputs,
                data=data,
                step=step,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                teacher_force_scheduler=teacher_force_scheduler,
                accelerator=accelerator,
                config=config,
            )
            progress_bar.update(1)

            # Compute statistics
            n_samples = len(next(iter(data.values())))
            step += 1
            avg_loss.update(loss, n_samples)
            for loss_name, loss_value in inner_losses.items():
                avg_inner_losses[f"avg_{loss_name}"].update(loss_value, n_samples)

            wandb.log(
                {
                    "train_loss": loss,
                    **inner_losses,
                    "lr": lr_scheduler.get_last_lr()[0],
                },
                step=step,
            )

            # Evaluate the model and save checkpoints
            if (step % config.log_interval == 0) or (step == total_steps):
                # Evaluate the model
                val_loss, val_inner_losses, val_metrics = train_evaluation(
                    model,
                    val_dataloader,
                    loss_fn,
                    # metrics=metrics,
                )
                model.train()

                train_log(
                    avg_loss,
                    avg_inner_losses,
                    val_loss,
                    val_inner_losses,
                    val_metrics,
                    lr=lr_scheduler.get_last_lr()[0],
                    step=step,
                )
                avg_loss = AvgValue()
                avg_inner_losses = defaultdict(AvgValue)

            if step % config.checkpoint_interval == 0:
                # Saving checkpoint
                save_model_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    step,
                    checkpoint_counter,
                    config,
                )
                wandb.log(
                    {
                        "checkpoint_counter": checkpoint_counter,
                    },
                    step=step,
                )
                checkpoint_counter += 1

        gc.collect()
        torch.cuda.empty_cache()

    wandb.unwatch(watch_list)


def train_batch(
    inputs,
    data,
    step,
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    config,
    teacher_force_scheduler=None,
    accelerator=None,
    device=None,
):
    assert (
        accelerator is not None or device is not None
    ), "One between accelerator and device must be set."

    if accelerator is None:
        data = {key: value.to(device) for key, value in data.items()}

    outputs = model(**inputs, teacher_force=teacher_force_scheduler.get_value(), return_dict=True)

    loss, inner_losses = loss_fn(outputs, data)
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()

    if config.get("gradient_clip", "none") != "none":
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

    if step % config.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    lr_scheduler.step()
    teacher_force_scheduler.step()

    return loss.item(), inner_losses


def train_evaluation(
    model,
    dataloader,
    compute_loss: ComputeLoss = None
    # , metrics: Dict[str, Metric] = {}
) -> Tuple[AvgValue, Dict[str, AvgValue], Dict[str, AvgValue]]:
    model.eval()
    avg_loss = AvgValue()
    avg_inner_losses = defaultdict(AvgValue)
    avg_metrics = defaultdict(AvgValue)

    forward_signature = set(inspect.signature(model.forward).parameters)
    with torch.no_grad():
        for data in dataloader:
            inputs_kwargs = {
                argument: value
                for argument, value in data.items()
                if argument in forward_signature
            }
            n_samples = len(next(iter(data.values())))

            outputs = model(**inputs_kwargs, return_dict=True)
            if compute_loss is not None:
                loss, inner_losses = compute_loss(outputs, data)

                avg_loss.update(loss.item(), n_samples)
                for loss_name, loss_value in inner_losses.items():
                    avg_inner_losses[loss_name].update(loss_value, n_samples)

            # for metric_name, metric in metrics.items():
            #     metric_value = metric(outputs, data)
            #     avg_metrics[metric_name].update(metric_value, n_samples)

    return avg_loss, avg_inner_losses, avg_metrics


def train_log(
    train_loss: AvgValue,
    train_inner_losses: Dict[str, AvgValue],
    val_loss: AvgValue,
    val_inner_losses: Dict[str, AvgValue],
    val_metrics: Dict[str, AvgValue],
    lr,
    step,
):
    train_loss = train_loss.avg()
    train_inner_losses = {
        f"{loss_name}": loss_value.avg()
        for loss_name, loss_value in train_inner_losses.items()
    }

    val_loss = val_loss.avg()
    val_inner_losses = {
        f"val_{loss_name}": loss_value.avg()
        for loss_name, loss_value in val_inner_losses.items()
    }

    val_metrics = {
        f"val_{metric_name}": metric_value.avg()
        for metric_name, metric_value in val_metrics.items()
    }

    wandb.log(
        {
            "avg_train_loss": train_loss,
            **train_inner_losses,
            "val_loss": val_loss,
            **val_inner_losses,
            **val_metrics,
            "lr": lr,
        },
        step=step,
    )
    print(
        f"Iteration: {step:6}",
        f"train loss: {train_loss:.4f}",
        f"val loss: {val_loss:.4f}",
        f"lr: {lr:.6f}",
        sep="\t",
    )


def save_model_checkpoint(
    model, optimizer, lr_scheduler, epoch, step, checkpoint_counter, config
):
    checkpoint_file = os.path.join(
        CONFIG.models.checkpoints_dir(
            config.model_name, config.get("history_length", 0) > 0
        ),
        config.model_name,
        f"checkpoint_{checkpoint_counter}.pt",
    )
    create_dirs_for_file(checkpoint_file)

    save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        epoch,
        step,
        checkpoint_counter,
        checkpoint_path=checkpoint_file,
    )
    wandb.save(f"{config.model_name}_{checkpoint_counter}.pt")


def load_model_checkpoint(
    checkpoint_counter, config, model, optimizer=None, lr_scheduler=None
):
    checkpoint_file = os.path.join(
        CONFIG.models.checkpoints_dir(
            config.model_name, config.get("history_length", 0) > 0
        ),
        config.model_name,
        f"checkpoint_{checkpoint_counter}.pt",
    )
    return load_checkpoint(
        checkpoint_file, model, optimizer=optimizer, scheduler=lr_scheduler
    )


def evaluate(model, tokenizer, train_data: datasets.Dataset, val_data: datasets.Dataset, config):
    datasets = [("train", train_data), ("val", val_data)]
    results = {}
    for dataset_name, dataset in datasets:
        outputs, metrics = evaluate_model(model, tokenizer, dataset, config)
        results[dataset_name] = (outputs, metrics)

        for metric_name, metric_value in metrics.items():
            print(f"{dataset_name}_{metric_name}: {metric_value:.4f}")
            wandb.log({f"evaluation/{dataset_name}_{metric_name}": metric_value})
        
        gc.collect()
        torch.cuda.empty_cache()
    return results


# def evaluate(model, train_dataloader, val_dataloader, metrics, config):
#     accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.cpu)
#     model, train_dataloader, val_dataloader = accelerator.prepare(
#         model, train_dataloader, val_dataloader
#     )

#     datasets = [("train", train_dataloader), ("val", val_dataloader)]
#     for dataset_name, dataloader in datasets:
#         if config.model_type == "encoder_decoder":
#             encoder_results = evaluate_model(
#                 model.encoder,
#                 dataloader,
#                 metrics["encoder"],
#                 config,
#                 get_encoder_outputs,
#             )
#             encoder_decoder_results = evaluate_model(
#                 model,
#                 dataloader,
#                 metrics["encoder_decoder"],
#                 config,
#                 get_encoder_decoder_outputs,
#             )
#             results = {**encoder_results, **encoder_decoder_results}
#         elif config.model_type == "encoder":
#             results = evaluate_model(
#                 model, dataloader, metrics["encoder"], config, get_encoder_outputs
#             )
#         else:
#             raise ValueError(
#                 "Invalid model_type. Supported values are 'encoder_decoder' and 'encoder'."
#             )

#         results = {
#             metric_name: metric_value.avg()
#             for metric_name, metric_value in results.items()
#         }

#         for metric_name, metric_value in results.items():
#             print(f"{dataset_name}_{metric_name}: {metric_value:.4f}")
#             wandb.log({f"evaluation/{dataset_name}_{metric_name}": metric_value})


# def evaluate_model(model, dataloader, metrics, config, get_outputs):
#     model.eval()

#     avg_metrics = defaultdict(AvgValue)
#     forward_signature = set(inspect.signature(model.forward).parameters)
#     with torch.no_grad():
#         for data in tqdm(dataloader):
#             inputs_kwargs = {
#                 arg: value for arg, value in data.items() if arg in forward_signature
#             }
#             n_samples = len(next(iter(data.values())))
#             outputs = get_outputs(model, inputs_kwargs)

#             for metric_name, metric in metrics.items():
#                 metric_value = metric(outputs, data)
#                 avg_metrics[metric_name].update(metric_value, n_samples)
#     return avg_metrics


# def get_encoder_outputs(model, inputs_kwargs):
#     return model(**inputs_kwargs, return_dict=True)


# def get_encoder_decoder_outputs(model, inputs_kwargs):
#     inputs_kwargs.pop("decoder_input_ids", None)
#     return {"output_ids": model.generate(**inputs_kwargs)}