"""Trainer module."""
import argparse
import dataclasses
import logging
from socket import TIPC_CRITICAL_IMPORTANCE
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from espnet.asr.asr_utils import plot_spectrogram
import os, sys
import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.build_dataclass import build_dataclass

import pyworld as pw
import torchaudio


if torch.distributed.is_available():
    from torch.distributed import ReduceOp
from torchinfo import summary
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


import matplotlib as mpl
import holoviews as hv
import panel as pn
from bokeh.resources import INLINE
hv.extension("bokeh", logo=False)

import torch
import librosa
import torchaudio
import torchaudio.transforms as T

from scipy.io import wavfile
from scipy.signal import spectrogram

import wandb
import matplotlib.pyplot as plt





# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
# # %matplotlib qt
# # from IPython import display
# import matplotlib.cm as cmx
# import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_latent_space(x_batch, y_batch, iteration=None, dim=2):

    model = TSNE(n_components=dim, random_state=0, perplexity=50, learning_rate=500, n_iter=200)
    z_mu = model.fit_transform(mu.eval(feed_dict={X: x_batch}))
    n_classes = len(list(set(np.argmax(y_batch, 1))))
    cmap = get_cmap(n_classes)
    fig = plt.figure(2, figsize=(8,8))

    if dim is 3:
        for i in list(set(np.argmax(y_batch, 1))):
            bx = fig.add_subplot(111, projection='3d')

            index = np.where(np.argmax(y_batch, 1) == i)
            xs = z_mu[index, 0]
            ys = z_mu[index, 1:]
            zs = z_mu[index, 2]
            bx.scatter(xs, ys, zs,c=cmap(i), label=str(i))
    else:
        for i in list(set(np.argmax(y_batch, 1))):
            bx = fig.add_subplot(111)
            index = np.where(np.argmax(y_batch, 1) == i)
            xs = z_mu[index, 0]
            ys = z_mu[index, 1]
            bx.scatter(xs, ys, c=cmap(i), label=str(i))

    bx.set_xlabel('X Label')
    bx.set_ylabel('Y Label')
    bx.legend()
    bx.set_title('Truth')
    if iteration is None:
        plt.savefig('latent_space.png')
    else:
        plt.savefig('latent_space' + str(iteration) + '.png')
    plt.show()



# def generate_melspec(audio_data, sample_rate=48000, power=2.0, n_fft = 1024, win_length = None, hop_length = None, n_mels = 80):
# 	if hop_length is None:
# 		hop_length = n_fft//2

# 	# convert to torch
# 	audio_data = torch.tensor(audio_data, dtype=torch.float32)
# 	mel_spectrogram = T.MelSpectrogram(
# 		sample_rate=sample_rate,
# 		n_fft=n_fft,
# 		win_length=win_length,
# 		hop_length=hop_length,
# 		center=True,
# 		pad_mode="reflect",
# 		power=power,
# 		norm="slaney",
# 		onesided=True,
# 		n_mels=n_mels,
# 		mel_scale="htk",
# 	)

# 	melspec = mel_spectrogram(audio_data).numpy()
# 	mel_db = np.flipud(librosa.power_to_db(melspec))
# 	return mel_db



# def wandb_spectrogram(audio_path:str, audio_data=None, sample_rate=None, specs:str=['all_specs'], layout:str='row', height=170, width=400, channel=0, cmap='blues'):
#     if audio_data.dtype != np.int16:  # Panel Audio widget can't handle floats
# 		audio_data =  np.clip( audio_data.cpu().numpy()*32768 , -32768, 32768).astype('int16')

# 	# Audio widget
# 	audio = pn.pane.Audio(audio_data, sample_rate=sample_rate, name='Audio', throttle=10)

# 	# Add HTML components
# 	line = hv.VLine(0).opts(color='red')
# 	line2 = hv.VLine(0).opts(color='green')
# 	line3 = hv.VLine(0).opts(color='white')

# 	slider = pn.widgets.FloatSlider(end=duration, visible=False, step=0.001)
# 	slider.jslink(audio, value='time', bidirectional=True)
# 	slider.jslink(line, value='glyph.location')
# 	slider.jslink(line2, value='glyph.location')
#  	slider.jslink(line3, value='glyph.location')

# 	# Spectogram plot
# 	if ('spec' in specs) or ('all_specs' in specs) or ('all' in specs):
# 		f, t, sxx = spectrogram(audio_data, sample_rate)
# 		spec_gram_hv = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (hz)"]).opts(width=width, height=height, labelled=[], axiswise=True, color_levels=512, cmap=cmap) * line
# 	else:
# 		spec_gram_hv = None

# 	# Melspectogram plot
# 	if ('melspec' in specs) or ('all_specs' in specs) or ('all' in specs):
# 		# mel_db = generate_melspec(audio_data, sample_rate=sample_rate, power=2.0, n_fft = 1024, n_mels = 128)
# 		melspec_gram_hv = hv.Image(mel_db, bounds=(0, 0, duration, mel_db.max()), kdims=["Time (s)", "Mel Freq"]).opts(width=width, height=height, labelled=[], axiswise=True, color_levels=512, cmap=cmap) * line3
# 	else:
# 		melspec_gram_hv = None


# 	# Waveform plot
# 	# if ('waveform' in specs) or ('all' in specs):
# 	# 	time = np.linspace(0, len(audio_data)/sample_rate, num=len(audio_data))
# 	# 	line_plot_hv = hv.Curve((time, audio_data), ["Time (s)", "amplitude"]).opts(width=width, height=height, axiswise=True) * line2
# 	# else:
# 	# 	line_plot_hv = None


# 	# Create HTML layout
# 	html_file_name = "audio_spec.html"



# 	if layout == 'grid':
# 		logging.warning('doing GRID')
# 		combined = pn.GridBox(audio, spec_gram_hv, line_plot_hv, melspec_gram_hv, slider, ncols=2, nrows=2).save(html_file_name)
# 	else:
# 		combined = pn.Row(audio, line_plot_hv, spec_gram_hv, melspec_gram_hv, slider).save(html_file_name)

# 	# if layout == 'grid':
# 	#   combined = pn.GridBox(audio, spec_gram * line, None, melspec_gram * line3, slider, ncols=2, nrows=2).save(html_file_name)
# 	# else:
# 	#   combined = pn.Row(audio, spec_gram * line, melspec_gram * line3, slider).save(html_file_name)

# 	return wandb.Html(html_file_name)








@dataclasses.dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    vae_annealing_cycle: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    adversarial_list: list
    adv_flag: bool
    save_every_epoch: int
    resume_from_checkpoint: int
    adv_loss_weight: float
    vae_weight_factor: float
    plot_iiter: int

class Trainer:
    """Trainer having a optimizer.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    beta_kl_factor = 0.1

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        pass

    @staticmethod
    def resume(
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        ngpu: int = 0,
    ):
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states["model"])
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.info(f"The training was resumed using {checkpoint}")

    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options,
        distributed_option: DistributedOption,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()

        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True

        if trainer_options.use_amp:
            if V(torch.__version__) < V("1.6.0"):
                raise RuntimeError(
                    "Require torch>=1.6.0 for  Automatic Mixed Precision"
                )
            if trainer_options.sharded_ddp:
                if fairscale is None:
                    raise RuntimeError(
                        "Requiring fairscale. Do 'pip install fairscale'"
                    )
                scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
            else:
                scaler = GradScaler()
        else:
            scaler = None

        if trainer_options.resume :
            if(trainer_options.resume_from_checkpoint < 5):
                loading_path = "{}/checkpoint.pth".format(output_dir)
            else:
                loading_path = "{}/{}_checkpoint.pth".format(output_dir, trainer_options.resume_from_checkpoint)
            logging.warning(">>>>> IMP: Loading resume from CHEKCPOINT :  {} adv_weight {}  ".format(trainer_options.resume_from_checkpoint, trainer_options.adv_loss_weight))
            if(os.path.exists(loading_path)):
                cls.resume(
                    checkpoint= loading_path,
                    model=model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    reporter=reporter,
                    scaler=scaler,
                    ngpu=trainer_options.ngpu,
                )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if distributed_option.distributed:
            if trainer_options.sharded_ddp:
                dp_model = fairscale.nn.data_parallel.ShardedDataParallel(
                    module=model,
                    sharded_optimizer=optimizers,
                )
            else:
                dp_model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=(
                        # Perform multi-Process with multi-GPUs
                        [torch.cuda.current_device()]
                        if distributed_option.ngpu == 1
                        # Perform single-Process with multi-GPUs
                        else None
                    ),
                    output_device=(
                        torch.cuda.current_device()
                        if distributed_option.ngpu == 1
                        else None
                    ),
                    find_unused_parameters=trainer_options.unused_parameters,
                )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter

            train_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "train")
            )
            valid_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "valid")
            )
        else:
            train_summary_writer = None

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            logging.warning(" current epoch {}  max_epoch {} ******".format(iepoch, trainer_options.max_epoch))
            if iepoch != start_epoch:
                logging.warning(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.warning(f"{iepoch}/{trainer_options.max_epoch}epoch started")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)

            # logging.warning(" current epochj {}" .format(iepoch))
            # if ((iepoch % trainer_options.vae_annealing_cycle) == 0):
            #     cls.beta_kl_factor = 0.1
            #     logging.warning('KL annealing restarted')


            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    distributed_option=distributed_option,
                    current_epoch=iepoch,
                )

            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )
            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=train_summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(
                        reporter.get_value(*trainer_options.val_scheduler_criterion)
                    )
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()
            if trainer_options.sharded_ddp:
                for optimizer in optimizers:
                    if isinstance(optimizer, fairscale.optim.oss.OSS):
                        optimizer.consolidate_state_dict()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                if trainer_options.use_matplotlib:
                    reporter.matplotlib_plot(output_dir / "images")
                if train_summary_writer is not None:
                    reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
                    reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                if(iepoch% trainer_options.save_every_epoch == 0 ):
                    # 4.2 Saving every 5th epoch as the checkpoint
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "reporter": reporter.state_dict(),
                            "optimizers": [o.state_dict() for o in optimizers],
                            "schedulers": [
                                s.state_dict() if s is not None else None
                                for s in schedulers
                            ],
                            "scaler": scaler.state_dict() if scaler is not None else None,
                        }, "{}/{}_checkpoint.pth".format( output_dir, iepoch),)

                # 5. Save and log the model and update the link to the best model
                torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in trainer_options.best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.warning(" IMPORTANT: There are no improvements in this epoch")
                else:
                    logging.warning(" IMPORTANT : The best model has been updated: " + ", ".join(_improved))

                log_model = (
                    trainer_options.wandb_model_log_interval > 0
                    and iepoch % trainer_options.wandb_model_log_interval == 0
                )
                if log_model and trainer_options.use_wandb:
                    import wandb

                    logging.warning("Logging Model on this epoch :::::")
                    artifact = wandb.Artifact(
                        name=f"model_{wandb.run.id}",
                        type="model",
                        metadata={"improved": _improved},
                    )
                    artifact.add_file(str(output_dir / f"{iepoch}epoch.pth"))
                    aliases = [
                        f"epoch-{iepoch}",
                        "best" if best_epoch == iepoch else "",
                    ]
                    wandb.log_artifact(artifact, aliases=aliases)

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                        for ph, k, m in trainer_options.best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )

                # Generated n-best averaged model
                if (
                    trainer_options.nbest_averaging_interval > 0
                    and iepoch % trainer_options.nbest_averaging_interval == 0
                ):
                    average_nbest_models(
                        reporter=reporter,
                        output_dir=output_dir,
                        best_model_criterion=trainer_options.best_model_criterion,
                        nbest=keep_nbest_models,
                        suffix=f"till{iepoch}epoch",
                    )

                for e in range(1, iepoch):
                    if( e% trainer_options.save_every_epoch > 0):
                        p = output_dir / f"{e}epoch.pth"
                        if p.exists() and e not in nbests:
                            p.unlink()
                            _removed.append(str(p))
                if len(_removed) != 0:
                    logging.warning("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            # if all_steps_are_invalid:
            #     logging.warning(
            #         f"The gradients at all steps are invalid in this epoch. "
            #         f"Something seems wrong. This training was stopped at {iepoch}epoch"
            #     )
            #     break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                    trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # Generated n-best averaged model
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
            )

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        current_epoch: int,
    ) -> bool:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed






        # for name, param in model.named_parameters():
        #     if (param.requires_grad):
        #         logging.warning(" name {}  ".format(name))
        # summary(model, input_size=(batch_size, 1, 28, 28))


        # "ASRTask"
        adv_mode = options.adversarial_list[current_epoch-1]
        adv_flag = options.adv_flag
        # 'espnet2.asr.espnet_model.ESPnetASRModel'
        if(options.ngpu > 1):
            adv_name = str(type(model.module).__name__)
            # logging.warning(" ------->>>>>>>>>>> ctc weight grad {}  \n ctc bias grad {}".format(  model.module.ctc.ctc_lo.weight.grad,  model.module.ctc.ctc_lo.bias.grad  ) )
        else:
            adv_name = str(type(model).__name__)


        # for name, layer in model.named_modules():
        #     logging.warning( " {} ".format(name))
        # logging.warning( " >>>> adv_name {} ".format(adv_name))
        # logging.warning(" model_vars {} \n\n".format( vars(model)))
        # logging.warning("******************************\n\n")
        # logging.warning(" cls_vars {} \n\n".format( vars(cls)))
        # logging.warning("******************************\n\n")
        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        if (adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'asr'):
            if options.ngpu > 1:
                model.module.freeze_adversarial()
                model.module.unfreeze_encoder()
            else:
                model.freeze_adversarial()
                model.unfreeze_encoder()

        elif (adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'adv'):
            if options.ngpu > 1:
                model.module.freeze_encoder()
                model.module.unfreeze_adversarial()
            else:
                model.freeze_encoder()
                model.unfreeze_adversarial()

        elif(adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'asradv'):
            if (options.ngpu > 1):
                model.module.unfreeze_encoder()
                model.module.unfreeze_adversarial()
            else:
                model.unfreeze_encoder()
                model.unfreeze_adversarial()


        elif(adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'recon'):
            if (options.ngpu > 1):
                model.module.recon_mode()
            else:
                model.recon_mode()

        elif(adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'reinit_adv'):
            if options.ngpu > 1:
                model.module.freeze_encoder()
                model.module.unfreeze_adversarial()
            else:
                model.freeze_encoder()
                model.unfreeze_adversarial()
                model.reinit_adv()
        param_group_length = len(optimizers[0].param_groups)
        current_flr = optimizers[0].param_groups[0]['lr']
        current_llr = optimizers[0].param_groups[-1]['lr']
        logging.warning(" --->>>>>  adv_mode {}  trainer {} adv_name {} current_lr_first_group {:.6f} last_group_lr {:.6f} param_length {} \n".format(adv_mode, options.save_every_epoch, adv_name, float(current_flr), float(current_llr), param_group_length))

        # tmp = float((current_epoch)% options.vae_annealing_cycle)/options.vae_annealing_cycle
        # new_lr = current_flr *0.5*(1+np.cos(tmp * np.pi))
        # for param_group in optimizers[0].param_groups:
        #     param_group['lr'] = new_lr
        # wandb.log( {"new_lr_for_kl_annealing" : new_lr })




        fig = plt.figure(figsize=(10,8), dpi=200 )

        # pca = PCA(n_components=10)
        # tsne = TSNE(n_components=2, perplexity=25, verbose=1, random_state=123)
        # kmeans = KMeans(n_clusters=10)



        if ((current_epoch-1) % options.vae_annealing_cycle) == 0:
            cls.beta_kl_factor = 0.1
            logging.warning(' current epoch {}/{} KL annealing restarted {}'.format(current_epoch, options.vae_annealing_cycle, cls.beta_kl_factor))


        for iiter, (utt_id, batch) in enumerate(reporter.measure_iter_time(iterator, "iter_time"), 1):
            assert isinstance(batch, dict), type(batch)
            cls.beta_kl_factor  = min(1, cls.beta_kl_factor + 1.0/( 5 * len(utt_id)))
            # logging.warning(" cls.beta_kl_factor {} new_lr {} ".format(cls.beta_kl_factor, new_lr))
            # logging.warning(" prinitng iiter {} ")
            # logging.warning( "iiter : {}   utt_id {} utt_idlen {} ".format(iiter, utt_id, len(utt_id)))
            # logging.warning("**************   Batch ************")
            # for keys,values in batch.items():
            #     logging.warning(" {}  >> {} \n".format(keys, values))
            # logging.warning("**************************\n\n")

            # logging.warning(" len_utt_id {} utt_id {} \n".format( len(utt_id), utt_id))



            # ['sp1.1-5703-47212-0024', 'sp1.1-5561-41615-0030', 'sp1.1-3723-171115-0030', 'sp1.1-3214-167607-0035', 'sp1.1-2817-142380-0037', 'sp1.1-2514-149482-0056', 'sp1.1-1624-142933-0016', 'sp0.9-78-369-0052', 'sp0.9-7067-76048-0024', 'sp0.9-5049-25947-0006', 'sp0.9-4481-17499-0016', 'sp0.9-3879-174923-0004', 'sp0.9-211-122442-0003', '441-128982-0006', '3857-182315-0042', '2764-36619-0037', 'sp1.1-1455-138263-0030', 'sp1.1-909-131045-0016', 'sp1.1-8975-270782-0084', 'sp1.1-8238-274553-0024', 'sp1.1-8051-118101-0022', 'sp1.1-7113-86041-0050', 'sp1.1-6081-42010-0021', 'sp1.1-5688-15787-0001', 'sp1.1-4267-287369-0019', 'sp1.1-403-128339-0046', 'sp1.1-332-128985-0040', 'sp1.1-332-128985-0038', 'sp1.1-3240-131232-0004', 'sp1.1-226-131533-0005', 'sp1.1-211-122442-0002', 'sp0.9-4853-27670-0017', 'sp0.9-481-123719-0021', 'sp0.9-445-123857-0021', 'sp0.9-4340-15220-0086', '911-130578-0007', '6836-61803-0037', '6385-34669-0018', '4267-287369-0008']
            # **************************
            # speech  >> tensor([[-1.3733e-03, -7.9346e-04,  3.3569e-04,  ..., -1.1292e-03,
            #         -1.6174e-03, -2.2278e-03],
            #     [ 1.6174e-03,  2.9297e-03,  2.1973e-03,  ..., -3.3569e-04,
            #         -4.5776e-04, -7.0190e-04],
            #     [-1.2207e-04,  6.1035e-04,  1.1902e-03,  ..., -4.5776e-03,
            #         -4.7913e-03, -2.5635e-03],
            #     ...,
            #     [-3.9673e-04, -3.6621e-04, -3.0518e-04,  ...,  0.0000e+00,
            #         0.0000e+00,  0.0000e+00],
            #     [ 3.9673e-04,  7.3242e-04,  1.2512e-03,  ...,  0.0000e+00,
            #         0.0000e+00,  0.0000e+00],
            #     [ 6.1035e-05,  6.1035e-05,  1.2207e-04,  ...,  0.0000e+00,
            #         0.0000e+00,  0.0000e+00]])
            # speech_lengths  >> tensor([180218, 180218, 180218, 180218, 180218, 180218, 180218, 180178, 180178,
            #     180178, 180178, 180178, 180178, 180160, 180160, 180160, 180146, 180145,
            #     180145, 180145, 180145, 180145, 180145, 180145, 180145, 180145, 180145,
            #     180145, 180145, 180145, 180145, 180089, 180089, 180089, 180089, 180080,
            #     180080, 180080, 180080])
            # text  >> tensor([[  11,   12,  425,  ...,  321,   -1,   -1],
            #     [  10,   70,   26,  ...,   20,   89,   -1],
            #     [   4,  497,   11,  ...,   -1,   -1,   -1],
            #     ...,
            #     [ 607,   86,    2,  ...,   -1,   -1,   -1],
            #     [   2,  121, 4357,  ...,   -1,   -1,   -1],
            #     [  13,   18,   66,  ...,   -1,   -1,   -1]])
            # text_lengths  >> tensor([47, 48, 35, 34, 43, 42, 37, 37, 38, 39, 37, 36, 34, 40, 46, 34, 39, 41,
            #     44, 47, 43, 38, 29, 49, 31, 39, 48, 45, 44, 33, 38, 27, 42, 31, 38, 38,
            #     36, 38, 28])


            # logging.warning(" encoder weight  {}  ".format(  model.encoder.encoders[1].feed_forward.w_1.weight  ) )
            # logging.warning(" reconstruction weight {}  ".format(  model.reconstruction_decoder.decoders[0].feed_forward.w_1.weight  ) )


            # logging.warning(" encoder weight grad {}  ".format(  model.encoder.encoders[1].feed_forward.w_1.weight.grad  ) )
            # logging.warning(" reconstruction weight grad {}  ".format(  model.reconstruction_decoder.decoders[0].feed_forward.w_1.weight.grad  ) )


            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            # for key, value in batch.items() :
            #     logging.warning(" key  {} ".format(key ) )

            with autocast(scaler is not None):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)

                    # Note(kamo):
                    # Supporting two patterns for the returned value from the model
                    #   a. dict type
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        loss_adversarial = retval.get( "loss_adversarial", 0 )
                        reconstruction_loss = retval.get("reconstruction_loss", 0)
                        kld_loss = retval.get("reconstruction_kld_loss", 0)
                        # vae_loss = retval.get("vae_loss",0)
                        # logging.warning(" retval : loss_without {}  weight {} loss_adversarial {} \n".format(loss, weight, loss_adversarial))
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()

                    #   b. tuple or list type
                    else:
                        loss, stats, weight = retval
                        optim_idx = None


                ###################################################################################
                ###################################################################################
                ###################################################################################



                ###################################################################################
                ###################################################################################
                ###################################################################################

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                # loss /= accum_grad

            reporter.register(stats, weight)


            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    if (adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'asr'):
                        decay = cls.beta_kl_factor
                        vae_loss = reconstruction_loss + (decay * kld_loss)
                        total_loss = (1 - options.vae_weight_factor) * loss + options.vae_weight_factor  *  vae_loss
                        total_loss /= accum_grad
                        scaler.scale(total_loss).backward()
                    elif (adv_flag == True and adv_name == "ESPnetASRModel" and  adv_mode == 'adv'):
                        loss_adversarial /= accum_grad
                        # loss_adversarial.requires_grad = True
                        scaler.scale(loss_adversarial).backward()
                    elif(adv_flag == True and adv_name == "ESPnetASRModel" and adv_mode == 'asradv'):
                        # loss_adversarial.requires_grad = True
                        decay = cls.beta_kl_factor
                        vae_loss = reconstruction_loss + (decay * kld_loss)
                        total_loss =  (1-options.vae_weight_factor) * loss + options.vae_weight_factor * vae_loss + options.adv_loss_weight * loss_adversarial
                        total_loss /= accum_grad
                        scaler.scale(total_loss).backward()
                    elif (adv_flag == True and adv_name == "ESPnetASRModel" and  adv_mode == 'reinit_adv'):
                        loss_adversarial /= accum_grad
                        # loss_adversarial.requires_grad = True
                        scaler.scale(loss_adversarial).backward()
                    elif (adv_flag == True and adv_name == "ESPnetASRModel" and  adv_mode == 'recon'):
                        # vae_loss = reconstruction_loss + kld_loss
                        # regularized vae_loss=kl_loss*decay + recon_loss
                        decay = cls.beta_kl_factor
                        vae_loss = reconstruction_loss + (decay * kld_loss)
                        # vae_loss /= accum_grad
                        wandb.log({ "beta_kl_factor" : decay } )
                        wandb.log({ "vae_loss" : vae_loss.detach() } )
                        scaler.scale(vae_loss).backward()

                    else:
                        vae_loss = reconstruction_loss + (decay * kld_loss)
                        total_loss = (1 - options.vae_weight_factor) * loss + options.vae_weight_factor  *  vae_loss
                        total_loss /= accum_grad
                        scaler.scale(total_loss).backward()
                        # scaler.scale(loss_adversarial).backward()
                        # Scales loss.  Calls backward() on scaled loss
                        # to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose
                        # for corresponding forward ops.
                else:
                    if (adv_flag == True and adv_name == "ESPnetASRModel" and  adv_mode == 'recon'):
                        decay = cls.beta_kl_factor
                        vae_loss = reconstruction_loss + (decay * kld_loss)
                        # vae_loss /= accum_grad
                        wandb.log({ "beta_kl_factor" : decay } )
                        wandb.log({ "vae_loss" : vae_loss.detach() } )
                        # logging.warning("vae_loss {} ".format(vae_loss))
                        vae_loss.backward()
                    else:
                        loss /= accum_grad
                        loss.backward()

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                ###################################################################################
                ###################################################################################
                ###################################################################################

                if((iiter % options.plot_iiter) == 0):
                    feats_plot = retval["feats_plot"]
                    recons_feats_plot = retval["recons_feats_plot"]
                    mu_logvar_combined = retval["mu_logvar_combined"]
                    feats_lengths = retval["feats_lengths"]
                    html_file_name = "./with_working_audio_30_march.png"

                    logging.warning("recons {}  mu_logvar_combined {} ".format(recons_feats_plot.shape, mu_logvar_combined.shape))
                    ax1 = plt.subplot(3, 1, 1)
                    ax1.set_title('Original feats linear')
                    plot_spectrogram(ax1, feats_plot.T, fs=16000, mode='linear', frame_shift=10, bottom=False, labelbottom=False)

                    ax2 = plt.subplot(3, 1, 2)
                    ax2.set_title('Reconstructed feats linear')
                    plot_spectrogram(ax2, recons_feats_plot.T, fs=16000, mode='linear', frame_shift=10, bottom=True, labelbottom=True)


                    ax3 = plt.subplot(3, 1, 3)

                    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
                    # processor = bundle.get_text_processor()
                    # tacotron2 = bundle.get_tacotron2().to("cuda")
                    vocoder = bundle.get_vocoder().to("cuda")

                    with torch.inference_mode():
                        # processed, lengths = processor( batch["text"][0] )
                        # processed = processed.to("cuda")
                        # lengths = lengths.to("cuda")
                        # spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                        # logging.warning(" >> just before vocoder  spec {} feats_lengths {} ".format(recons_feats_plot.shape, feats_lengths ))
                        spec = torch.Tensor(np.expand_dims(recons_feats_plot, axis=0).transpose(0, 2, 1)).to("cuda")
                        spec_lengths = torch.Tensor(feats_lengths).to("cuda")
                        waveforms, lengths = vocoder(spec, spec_lengths)

                    ax3.plot(waveforms[0].cpu().detach())
                    wandb.log({"recon_utt": wandb.Audio(waveforms, caption="reconstructed_utt", sample_rate=16)})
                    # IPython.display.Audio(waveforms[0:1].cpu(), rate=vocoder.sample_rate)


                    fig.subplots_adjust(hspace=0.15, bottom=0.00, wspace=0)
                    plt.tight_layout()
                    plt.savefig( '{}'.format(html_file_name), bbox_inches='tight' )
                    wandb.log({f"spectrogram plot": wandb.Image(plt)})
                    fig.clf()




                if( (iiter % 200) == 0):
                    logging.warning(" MODE: {} adv_loss_weight {} iiter {} current_epoch {} adv_flag {}  >>   asr_loss {} grad_norm {} ".format( adv_mode, options.adv_loss_weight, iiter, current_epoch, adv_flag,  stats["loss"].detach(), grad_norm ))
                    # logging.warning( " vae_loss {}  ctc_att_loss {} ").format(stats["vae_loss"], stats["ctc_att_loss"])
                    if(adv_flag == True and adv_name == "ESPnetASRModel"):
                        logging.warning(" adversarial_loss : {}   accuracy_adversarial {} \n".format( stats["loss_adversarial"].detach(), stats["accuracy_adversarial"] ))
                    if(iiter == 200):
                        # logging.warning(model)
                        if(options.ngpu > 1):
                            # logging.warning(" ctc weight grad {}  \n ctc bias grad {}".format(  model.module.ctc.ctc_lo.weight.grad,  model.module.ctc.ctc_lo.bias.grad  ) )
                            # logging.warning(" encoder weight grad {}  \n encoder bias grad {}".format(  model.module.encoder.encoders[2].feed_forward.w_1.weight.grad, model.module.encoder.encoders[2].feed_forward.w_1.bias.grad   ) )
                            if(adv_flag == True and adv_name == "ESPnetASRModel"):
                                logging.warning(" adversarial weight grad {}  \n adversarial bias grad {}".format( model.module.adversarial_branch.output.weight.grad, model.module.adversarial_branch.output.bias.grad   ) )

                        elif(options.ngpu == 1):

                            if(model.ctc.ctc_lo.weight.grad is not None):
                                logging.warning(" ctc weight grad {}   shape {}  ctc bias grad {}".format(  torch.count_nonzero(model.ctc.ctc_lo.weight.grad), model.ctc.ctc_lo.weight.grad.shape ,  torch.count_nonzero(model.ctc.ctc_lo.bias.grad)  ) )
                                logging.warning(" encoder weight grad {}  shape {}  encoder bias grad {}".format(  torch.count_nonzero(model.encoder.encoders[2].feed_forward.w_1.weight.grad), model.encoder.encoders[2].feed_forward.w_1.weight.grad.shape, torch.count_nonzero(model.encoder.encoders[2].feed_forward.w_1.bias.grad)   ) )
                            if(adv_flag == True and adv_name == "ESPnetASRModel"):
                                if(model.adversarial_branch.output.weight.grad is not None):
                                    logging.warning(" adversarial weight grad {} shape {}  adversarial bias grad {}".format( torch.count_nonzero(model.adversarial_branch.output.weight.grad), model.adversarial_branch.output.weight.grad.shape, torch.count_nonzero(model.adversarial_branch.output.bias.grad)   ) )

                        logging.warning("******************************")


                            # if(self.encoder_weight_layer is not None):
                            #     tmpx =  model.encoder.encoders[2].feed_forward.w_1.weight - self.encoder_weight_layer
                            #     tmpy =  model.ctc.ctc_lo.weight - self.ctc_weight_layer
                            #     if(self.adversarial_weight_layer is not None):
                            #         tmpz = model.adversarial_branch.output.weight - self.adversarial_weight_layer
                            #         logging.warning(" adversarial {}   ".format(torch.count_nonzero(tmpz)))
                            #     logging.warning(" encoder : {} ctc {}  ".format(torch.count_nonzero(tmpx), torch.count_nonzero(tmpy)))

                            # self.encoder_weight_layer = model.encoder.encoders[2].feed_forward.w_1.weight
                            # self.ctc_weight_layer = model.ctc.ctc_lo.weight
                            # if(adv_mode == "adv" or adv_mode == "asradv"):
                            #     self.adversarial_weight_layer = model.adversarial_branch.output.weight






                # logging.info("\n ***** Grad norm : {} and loss :{} \n".format(grad_norm, loss))
                # print("/*** train/trainer.py grad norm {} and loss {} \n".format(grad_norm, loss))
                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()


                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad(set_to_none=True)

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.warning(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        adv_flag = options.adv_flag

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (utt_id, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break


            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            retval = model(**batch)
            if isinstance(retval, dict):
                # stats = retval["stats"]
                # weight = retval["weight"]
                stats = retval.get("stats")
                weight = retval.get("weight")
            else:
                if(adv_flag):
                    _, stats, weight, __ = retval
                else:
                    _, stats, weight = retval




            # feats_plot = retval["feats_plot"]
            # recons_feats_plot = retval["recons_feats_plot"]
            # mu_logvar_combined = retval["mu_logvar_combined"]
            # feats_lengths = retval["feats_lengths"] 
            # html_file_name = "./with_working_audio_30_march.png"

            # logging.warning("recons {}  mu_logvar_combined {} ".format(recons_feats_plot.shape, mu_logvar_combined.shape))
            # ax1 = plt.subplot(3, 1, 1)
            # plt.title('Original feats linear')
            # plot_spectrogram(ax1, feats_plot.T, fs=16000, mode='linear', frame_shift=10, bottom=False, labelbottom=False)

            # ax2 = plt.subplot(3, 1, 2)
            # # plt.title('Reconstructed feats linear')
            # plot_spectrogram(ax2, recons_feats_plot.T, fs=16000, mode='linear', frame_shift=10, bottom=True, labelbottom=True)


            # ax3 = plt.subplot(3, 1, 3)

            # bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
            # # processor = bundle.get_text_processor()
            # # tacotron2 = bundle.get_tacotron2().to("cuda")
            # vocoder = bundle.get_vocoder()
            # vocoder= to_device(vocoder, "cuda" if ngpu > 0 else "cpu")

            # with torch.inference_mode():
            #     # processed, lengths = processor( batch["text"][0] )
            #     # processed = processed.to("cuda")
            #     # lengths = lengths.to("cuda")
            #     # spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
            #     # logging.warning(" >> just before vocoder  spec {} feats_lengths {} ".format(recons_feats_plot.shape, feats_lengths ))
            #     spec = torch.Tensor(np.expand_dims(recons_feats_plot, axis=0).transpose(0, 2, 1)).to("cuda")
            #     spec_lengths = torch.Tensor(feats_lengths).to("cuda")
            #     waveforms, lengths = vocoder(spec, spec.lengths)


            # ax3.plot(waveforms[0].cpu().detach())
            # wandb.log({"recon_utt": wandb.Audio(waveforms, caption="reconstructed_utt", sample_rate=16)})
            # # IPython.display.Audio(waveforms[0:1].cpu(), rate=vocoder.sample_rate)


            # fig.subplots_adjust(hspace=0.15, bottom=0.00, wspace=0)
            # plt.tight_layout()
            # plt.savefig( '{}'.format(html_file_name), bbox_inches='tight' )
            # wandb.log({f"spectrogram plot": wandb.Image(plt)})
            # fig.clf()





            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Optional[Path],
        summary_writer,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        import matplotlib

        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )

            batch["utt_id"] = ids

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):

                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim > 3 or att_w.ndim == 1:
                        raise RuntimeError(f"Must be 2 or 3 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    if output_dir is not None:
                        p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(p)

                    if summary_writer is not None:
                        summary_writer.add_figure(
                            f"{k}_{id_}", fig, reporter.get_epoch()
                        )

                    if options.use_wandb:
                        import wandb
                        wandb.log({f"attention plot/{k}_{id_}": wandb.Image(fig)})
            reporter.next()
