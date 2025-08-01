"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
import time

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

        # <<< ADDED FOR EFFICIENCY PROFILING >>>
        self.profile_efficiency = False  # Control flag
        self.num_warmup_iters = 10
        self.num_measure_iters = 50
        self.training_latencies = []
        self.peak_training_memory_bytes = 0
        self.global_iter_count = 0  # To track iterations across epochs for profiling
        # <<< END ADDED FOR EFFICIENCY PROFILING >>>

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):

        # <<< MODIFIED FOR EFFICIENCY PROFILING >>>
        # Initialize profiling flags if they are not set by a config
        # (You might want to make these configurable via self.cfg)
        if hasattr(self.cfg, 'profile_efficiency') and self.cfg.profile_efficiency:
            self.profile_efficiency = True
            self.num_warmup_iters = getattr(self.cfg, 'profile_warmup_iters', 10)
            self.num_measure_iters = getattr(self.cfg, 'profile_measure_iters', 50)
            self.logger.info(
                f"*** Efficiency Profiling Enabled: Warmup={self.num_warmup_iters}, Measure={self.num_measure_iters} ***")
        # <<< END MODIFIED >>>

        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                    # <<< ADDED FOR EFFICIENCY PROFILING (Iteration counting and early exit) >>>
                    if self.profile_efficiency:
                        self.global_iter_count += 1
                        if self.global_iter_count >= self.num_warmup_iters + self.num_measure_iters:
                            self.logger.info("Efficiency profiling complete. Stopping training early.")
                            self.after_epoch()  # Call after_epoch hooks if stopping mid-epoch
                            self.after_train()  # Call after_train and print results
                            self.logger.info(f"DEBUG: Inside Trainer.train(). Profiling enabled: {self.profile_efficiency}")
                            self.logger.info(f"DEBUG: Warmup iters: {self.num_warmup_iters}, Measure iters: {self.num_measure_iters}")
                            self.logger.info(f"DEBUG: Current global_iter_count at start of train(): {self.global_iter_count}")
                            return  # Exit training early
                    # <<< END ADDED >>>
                # => after epoch
                self.after_epoch()
                # <<< ADDED FOR EFFICIENCY PROFILING (Check after full epoch if not enough iters yet) >>>
                if self.profile_efficiency and self.global_iter_count >= self.num_warmup_iters + self.num_measure_iters:
                    self.logger.info("Efficiency profiling complete (end of epoch).")
                    self.after_train()  # Call after_train and print results
                    return
                # <<< END ADDED >>>
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()
        # <<< ADDED FOR EFFICIENCY PROFILING (Reset peak memory at start of measurement phase) >>>
        if self.profile_efficiency and self.global_iter_count == self.num_warmup_iters:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # Reset for the device model is on
                self.logger.info("Reset peak CUDA memory stats for measurement phase.")
        # <<< END ADDED >>>

    def before_step(self):
        # <<< ADDED FOR EFFICIENCY PROFILING (Latency Start) >>>
        if self.profile_efficiency and self.global_iter_count >= self.num_warmup_iters:
            if torch.cuda.is_available():
                comm.synchronize()  # Synchronize before starting timer
            self.iter_start_time = time.perf_counter()
        # <<< END ADDED >>>
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        # <<< ADDED FOR EFFICIENCY PROFILING (Latency End & Memory Capture) >>>
        if self.profile_efficiency and self.global_iter_count >= self.num_warmup_iters:
            if torch.cuda.is_available():
                comm.synchronize()  # Synchronize before ending timer and capturing memory
            iter_end_time = time.perf_counter()
            self.training_latencies.append((iter_end_time - self.iter_start_time) * 1000)  # ms

            if torch.cuda.is_available():
                current_peak_memory = torch.cuda.max_memory_allocated()  # For current device
                if current_peak_memory > self.peak_training_memory_bytes:
                    self.peak_training_memory_bytes = current_peak_memory
        # <<< END ADDED >>>
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()

        # <<< ADDED FOR EFFICIENCY PROFILING (Result Printing) >>>
        if self.profile_efficiency:
            if comm.is_main_process():  # Only print on main process for DDP
                if self.training_latencies:
                    avg_training_latency = sum(self.training_latencies) / len(self.training_latencies)
                    self.logger.info(f"\n--- Training Efficiency (Batch Size: {self.cfg.batch_size_per_gpu}) ---")
                    self.logger.info(
                        f"Average Training Latency: {avg_training_latency:.0f}ms (over {len(self.training_latencies)} iterations)")
                else:
                    self.logger.info("Not enough iterations performed to measure training latency for profiling.")

                if self.peak_training_memory_bytes > 0 and torch.cuda.is_available():
                    peak_training_memory_gb = self.peak_training_memory_bytes / (1024 ** 3)
                    self.logger.info(f"Peak Training GPU Memory: {peak_training_memory_gb:.1f}G")
                else:
                    self.logger.info(
                        "Not enough iterations performed or CUDA not available to measure training memory for profiling.")
        # <<< END ADDED >>>

        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()

        self.comm_info["iter_per_epoch"] = len(self.train_loader)  # Needed for scheduler if not profiling
        self.max_iter = self.cfg.eval_epoch * len(self.train_loader)  # Total iterations for full training

        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

        

    def train(self):

        # <<< MODIFIED FOR EFFICIENCY PROFILING (Initialize from cfg) >>>
        if hasattr(self.cfg, 'profile_efficiency') and self.cfg.profile_efficiency:
            self.profile_efficiency = True
            self.num_warmup_iters = getattr(self.cfg, 'profile_warmup_iters', 10)
            self.num_measure_iters = getattr(self.cfg, 'profile_measure_iters', 50)
            self.logger.info(
                f"*** Efficiency Profiling Enabled for Trainer: Warmup={self.num_warmup_iters}, Measure={self.num_measure_iters} ***")
            # Optionally, if profiling, cap max_epoch to ensure we don't run too long if iter_per_epoch is small
            # This is a bit crude, a better way is to cap total iterations.
            # self.max_epoch = (self.num_warmup_iters + self.num_measure_iters) // len(self.train_loader) + 1
            # self.logger.info(f"Profiling: Effective max_epoch set to {self.max_epoch} to collect enough iterations.")
        else:
            self.profile_efficiency = False
            self.logger.info(f"DEBUG: Trainer __init__: Profiling Disabled.")
        self.global_iter_count = 0  # Reset for this specific trainer instance
        self.training_latencies = []
        self.peak_training_memory_bytes = 0
        # <<< END MODIFIED >>>

        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()

                    # --- Profiling iteration counting and early exit ---
                    if self.profile_efficiency:
                        self.global_iter_count += 1  # Count after a full step is done
                        if self.comm_info["iter"] % 20 == 0 and comm.is_main_process():  # Log progress
                            self.logger.info(
                                f"Epoch {self.epoch} Iter {self.comm_info['iter']}/{len(self.train_loader)} Global Profiling Iter: {self.global_iter_count}")

                        if self.global_iter_count >= self.num_warmup_iters + self.num_measure_iters:
                            self.logger.info("Efficiency profiling iterations collected. Stopping training early.")
                            # self.after_epoch() # Call after_epoch hooks before final after_train
                            self.after_train()  # This will print results and close writer
                            return  # Exit training
                    # --- End Profiling Iteration Logic ---

                # => after epoch
                self.after_epoch()
                # => after train

                # --- Check if profiling complete at end of epoch (if not enough iters in one epoch) ---
                if self.profile_efficiency and self.global_iter_count >= self.num_warmup_iters + self.num_measure_iters:
                    self.logger.info("Efficiency profiling iterations collected (at end of epoch).")
                    self.after_train()  # This will print results and close writer
                    return  # Exit training
                # --- End Profiling Check ---

            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
                enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
            # find_unused_parameters = True,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        # self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        if not self.profile_efficiency:  # If full training, use all epochs
            self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        else:  # If profiling, scheduler might not run much, but needs a valid total_steps
            # Or you could even skip scheduler updates during pure profiling runs if desired.
            self.cfg.scheduler.total_steps = self.num_warmup_iters + self.num_measure_iters
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
