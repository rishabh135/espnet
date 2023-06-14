#!/usr/bin/env python

from distutils import text_file
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch, sys
from torch.nn import functional as F
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

from espnet.asr.asr_utils import plot_spectrogram

from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from espnet2.asr.adversarial_branch import SpeakerAdv
from espnet2.asr.adversarial_branch import ReverseLayerF

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield



from torchinfo import summary
from transformers import AutoModelForSequenceClassification

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav




import numpy as np

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

from transformers import Wav2Vec2ConformerConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2ForPreTraining, Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CTCTokenizer
from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForPreTraining, Wav2Vec2ConformerModel, Wav2Vec2Processor
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    _compute_mask_indices,
    _sample_negative_indices,
)


def draw_mfcc(mfcc_feats, ax, fig, ylabel=True):
    mfcc_data= np.swapaxes(mfcc_feats, 0 ,1)

    #mfcc_x = [0.025]
    mfcc_x = [0.0]
    for i in range(1, mfcc_data.shape[1]):
        mfcc_x.append(0.025+i*0.01)
    mfcc_y = np.arange(0, 80)

    cax = ax.pcolormesh(mfcc_x, mfcc_y, mfcc_data, cmap='afmhot')
    #ax.set_ylim([0, spectrogram.ymax])

    #mfcc_data= mfcc_feats
    #cax = ax.imshow(mfcc_data, cmap='afmhot', origin='lower', interpolation='none')
    #ax.set_aspect('auto')
    ax.grid(False)
    #ax.set_ylim([0, 13])
    # if ylabel:
    # 	ax.set_ylabel("Mel scale", size=5)
    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("bottom", size="5%", pad=0.2)
    cb = fig.colorbar(cax, ax=ax, orientation='horizontal', cax=colorbar_axes)
    # cb.ax.tick_params(labelsize=5)

    #ax.set_xlabel("Time [s]", size=AX_LABEL)
    # ax.tick_params(axis='x', labelsize=2)
    ax.tick_params(axis='y', labelsize=2)







class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        adv_flag,
        grlalpha,
        latent_dim,
        embed_input_size: int,
        reconstruction_decoder: Optional,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        adversarial_branch: Optional[SpeakerAdv],
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.reconstruction_decoder = reconstruction_decoder
        self.adversarial_branch = adversarial_branch
        self.grlalpha = grlalpha
        self.adv_flag = adv_flag
        self.embed_input_size = embed_input_size


        self.adv_flag = adv_flag
        self.encoder_frozen_flag = False
        self.adversarial_frozen_flag = False
        self.reinit_adv_flag = False
        self.recon_mode_flag = False




        # encoder_out_size fixed unless you change the encoder params()
        self.final_encoder_dim = 256
        self.latent_dim = latent_dim
        # self.spk_embed_dim is present determined on how we calculated the xvector, dont change it
        self.spk_embed_dim = 512

        self.fc_mu = torch.nn.Linear(self.final_encoder_dim , self.latent_dim)
        self.fc_var = torch.nn.Linear(self.final_encoder_dim , self.latent_dim)
        self.fc_spemb = torch.nn.Linear(self.spk_embed_dim , self.latent_dim)
        self.decoder_input_projection = torch.nn.Linear(self.latent_dim, self.latent_dim//2 )



        self.disentangling_mapping_network = torch.nn.Sequential(torch.nn.Linear(768, self.final_encoder_dim), torch.nn.ReLU(), torch.nn.Linear(self.final_encoder_dim, self.final_encoder_dim), torch.nn.ReLU(), torch.nn.Linear(self.final_encoder_dim, self.final_encoder_dim))
        self.pretrained_embed = Conv2dSubsampling2(self.embed_input_size, 768, dropout_rate=0.0)


        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats


    #################################################################################################################################################################################################################################
    #################################################################################################################################################################################################################################
    #################################################################################################################################################################################################################################
    ### adding a adversarial branch  as done here https://github.com/espnet/espnet/blob/876c75f40a159fb41f82267b64a5bd7d55de1a96/espnet/nets/e2e_asr_th.py#L457
    # For training the adverarial branch, encoder must be frozen
    # self.enc_frozen = False




    def reinit_adv(self,):
        if(self.reinit_adv_flag == False):
            self.adversarial_branch.reinit_adv()
            self.reinit_adv_flag = True

    def print_flags(self,):
        logging.warning(" encoder frozen : {} adversarial_frozen : {} self.recon_frozen {} ".format(self.encoder_frozen_flag, self.adversarial_frozen_flag, self.recon_mode_flag ))




    def recon_mode(self):
        if(self.recon_mode_flag == False):
            logging.warning("  RECON MODE flag changing ")
            for param in self.decoder.parameters():
                param.requires_grad = False
                param.grad = None
            for param in self.ctc.ctc_lo.parameters():
                param.requires_grad = False
                param.grad = None
            for param in self.adversarial_branch.parameters():
                param.requires_grad = False
                param.grad = None
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.reconstruction_decoder.parameters():
                param.requires_grad = True

            self.recon_mode_flag = True

        return








    def freeze_encoder(self):
        if not self.encoder_frozen_flag:
            for param in self.encoder.parameters():
                param.requires_grad = False
                param.grad = None
                # if param.grad is not None:
                #     param.grad.zero_()
            for param in self.decoder.parameters():
                param.requires_grad = False
                param.grad = None
                # if param.grad is not None:
                #     param.grad.zero_()

            for param in self.ctc.ctc_lo.parameters():
                param.requires_grad = False
                param.grad = None
                # if param.grad is not None:
                #     param.grad.zero_()
            for param in self.reconstruction_decoder.parameters():
                param.requires_grad = False
                param.grad = None
            # for param in self.criterion_att.parameters():
            #     param.requires_grad = False
            #     if param.grad is not None:
            #         param.grad.zero_()

            self.encoder_frozen_flag = True
            self.recon_mode_flag = True
        self.print_flags()





    def unfreeze_encoder(self):
        if self.encoder_frozen_flag:
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.ctc.ctc_lo.parameters():
                param.requires_grad = True
            for param in self.reconstruction_decoder.parameters():
                param.requires_grad = True
            # for param in self.criterion_att.parameters():
            #     param.requires_grad = True

            self.encoder_frozen_flag = False
            self.recon_mode_flag = False


    def freeze_adversarial(self):
        if not self.adversarial_frozen_flag:
            for param in self.adversarial_branch.parameters():
                param.requires_grad = False
                param.grad = None
                # if param.grad is not None:
                #     # p.grad.detach_()
                #     param.grad.zero_()
            self.adversarial_frozen_flag = True
        self.print_flags()


    def unfreeze_adversarial(self):
        if self.adversarial_frozen_flag:
            for param in self.adversarial_branch.parameters():
                param.requires_grad = True
            self.adversarial_frozen_flag = False







    #################################################################################################################################################################################################################################
    #################################################################################################################################################################################################################################
    #################################################################################################################################################################################################################################



    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spkid: torch.Tensor,
        spkid_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            spkid: The shape is (Batch, 1)
            You can refer spkid for loss computation.
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

          # for data-parallel
        text = text[:, : text_lengths.max()]
        if(spembs is not None):
            # logging.warning("before_projection spembs {} ".format(spembs.shape))
            spembs = self.fc_spemb(spembs)
            # logging.warning("after_projection spembs {} ".format(spembs.shape))

        # 1. Encoder
        # https://huggingface.co/transformers/v4.4.2/model_doc/wav2vec2.html#wav2vec2featureextractor
        # https://huggingface.co/transformers/v4.4.2/model_doc/wav2vec2.html

        # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        # self.torchaudio_model = bundle.get_model( dl_kwargs={"model_dir":"/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/"} ).to(text.device)
        # logging.warning(f"{self.extract_feats_model.__class__}")


        # self.pretrained_features = transformers.Wav2Vec2FeatureExtractor(attention_mask = )



        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################

        configuration = Wav2Vec2Config(hidden_size=768)
        # # processor = .from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.wav2_feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" )
        self.wav2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/")
        self.wav2_pretrained_model =  Wav2Vec2Model(configuration).from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/").to(text.device)

        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################


        # "facebook/wav2vec2-conformer-rel-pos-large"
        # Initializing a Wav2Vec2Conformer facebook/wav2vec2-conformer-rel-pos-large style configuration
        # self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding=False, do_normalize=False, return_attention_mask=True)
        # # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" )
        # # processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer = tokenizer)
        # # processor =  Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" )
        # configuration = Wav2Vec2ConformerConfig()
        # self.wav2_pretrained_model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" ).to(text.device)

        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################
        ########################################################################################################################################################################################################################################################################





        # Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" )


        # .to(text.device)

        # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        # input_speech = self._load_datasamples(2)


        # Initializing a Wav2Vec2Conformer facebook/wav2vec2-conformer-rel-pos-large style configuration
        # configuration = Wav2Vec2ConformerConfig()
        # Initializing a model (with random weights) from the facebook/wav2vec2-conformer-rel-pos-large style configuration
        # self.wav2_model = Wav2Vec2ConformerModel(configuration)

        # Accessing the model configuration
        # configuration = model.config
        # logging.warning(f" {configuration} ")
        # using a pretrained wav2vec2 model

        # self.wav2_pretrained_model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft", cache_dir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/" ).to(text.device)



        # AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large", savedir="/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/rgupta/pretrained_vocoder/wav2vec2conf/", run_opts={"device":text.device })
        # model = Wav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")




        encoder_out, encoder_out_lens, feats, feats_lengths, aug_feats, aug_feats_lengths = self.encode(speech, speech_lengths)
        # logging.warning(" encoder_out {} feats {} ".format( encoder_out.shape, feats.shape))
        # original_feats = feats
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(encoder_out)
        log_var = self.fc_var(encoder_out)
        z = self.reparameterize(mu, log_var)
        bayesian_latent = z


        ################################################################################################################################################################################################
        ################################################################################################################################################################################################
        ################################################################################################################################################################################################
        ################################################################################################################################################################################################
        # https://github.com/espnet/espnet/blob/695c9954e20800875b22d985e9c0b0a70e8e2082/espnet2/tts/transformer/transformer.py


        # if(spembs is not None):
        #     zeros_spembs = torch.zeros_like(spembs)

        # logging.warning(f" >>> bayesian_latent: {bayesian_latent.shape}, {encoder_out_lens} {feats.shape} {feats_lengths} ")

        recons_feats = self.reconstruction_decoder( text=bayesian_latent, text_lengths=encoder_out_lens, feats=feats, feats_lengths=feats_lengths, spembs = spembs )
        # out_sum = summary(self.reconstruction_decoder, input_data=[bayesian_latent, encoder_out_lens, feats, feats_lengths, spembs], mode="train", col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], depth=1)
        # logging.warning(" reconstruction_decoder_summary: {} \n".format(out_sum))

        # logging.warning(" original_feats {} recons_feats shape {} ".format(feats.shape, recons_feats.shape))
        # # sum_recon_kl_loss =  reconstruction_loss + kld_loss

        reconstruction_loss , kld_loss = self.vae_loss_function(recons_feats, feats, mu, log_var)

        # logging.warning(" count how different feats are {} ".format( torch.nonzero(original_feats-feats) ))
        ################################################################################################################################################################################################
        ################################################################################################################################################################################################
        ################################################################################################################################################################################################









        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()



        # 1. CTC branch
        # encoder_out =
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc



        #################################################################################################################################################################################################################################
        #################################################################################################################################################################################################################################
        #################################################################################################################################################################################################################################
        ### adding a adversarial branch  as done here https://github.com/espnet/espnet/blob/876c75f40a159fb41f82267b64a5bd7d55de1a96/espnet/nets/e2e_asr_th.py#L457

        #################################################################################################################################################################################################################################
        #################################################################################################################################################################################################################################



        retval = {}
        if (self.adv_flag):
            # logging.info("Computing adversarial loss and flag inside {}  \n".format(self.adv_flag))
            rev_hs_pad = ReverseLayerF.apply(encoder_out, self.grlalpha)
            # print("\n\n rev hs pad : {} \n  encoder: out {}  \n text len {}  \n\n\n".format(rev_hs_pad.shape, encoder_out_lens.shape, text.shape ))
            loss_adv, acc_adv = self.adversarial_branch(rev_hs_pad, encoder_out_lens, spkid)

            # print("espnet_model.py adversarial_loss {} and accuracy {} \n".format(loss_adv, acc_adv))
            stats["loss_adversarial"] = loss_adv.detach() if loss_adv is not None else None
            retval["loss_adversarial"]= loss_adv if loss_adv is not None else None

            stats["accuracy_adversarial"]= acc_adv * 100 if acc_adv is not None else None
            retval["accuracy_adversarial"]= acc_adv if acc_adv is not None else None



        retval["reconstruction_loss"] = reconstruction_loss
        stats["reconstuction_loss"] = reconstruction_loss.detach()

        retval["reconstruction_kld_loss"] = kld_loss
        stats["KLD_loss"] = kld_loss.detach()

        # retval["vae_loss"]= sum_recon_kl_loss
        # stats["sum_kl_recon_loss"] = sum_recon_kl_loss.detach()


        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + reconstruction_loss + kld_loss

            # Collect Attn branch stats
            stats["loss_attention"] = loss_att.detach() if loss_att is not None else None
            stats["accuracy_asr"] = acc_att * 100
            stats["cer_attention"] = cer_att
            stats["wer_attention"] = wer_att


        # Collect total loss stats
        stats["loss"] = loss.detach()
        # stats["recons_loss"] = reconstruction_loss.detach()
        # stats["recons_kld_loss"] = kld_loss.detach()


        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        retval["loss"] = loss
        retval["stats"] = stats
        retval["weight"] = weight
        retval["loss_ctc"] = loss_ctc
        retval["loss_att"] = loss_att

        # retval["feats_plot"] = feats[0].detach().cpu().numpy()
        # retval["recons_feats_plot"] = recons_feats[0].detach().cpu().numpy()
        # retval["aug_feats_plot"] = aug_feats[0].detach().cpu().numpy()


        return retval





    def vae_loss_function(self, recon_decoder_output, ground_truths, mu, log_var ):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # recons = args[0]
        # input = args[1]
        # mu = args[2]
        # log_var = args[3]
        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recon_decoder_output, ground_truths)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        # loss = recons_loss + kld_weight * kld_loss
        return recons_loss, kld_loss

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor ) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu



    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            aug_feats = feats
            aug_feats_lengths = feats_lengths
            # 2. Data augmentation
            if self.specaug is not None and self.training:
                # logging.warning(f" aug_feats type {type(aug_feats)} ")
                aug_feats, aug_feats_lengths = self.specaug(aug_feats, aug_feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                aug_feats, aug_feats_lengths = self.normalize(aug_feats, aug_feats_lengths)



        # Pre-encoder, e.g. used for raw input data generally deactive
        # if self.preencoder is not None:
        #     logging.warning(" preencoder active \n\n")
        #     feats, feats_lengths = self.preencoder(feats, feats_lengths)



        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder( feats, feats_lengths, ctc=self.ctc )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(aug_feats, aug_feats_lengths)
            logging.warning(f" Original  {encoder_out_lens.shape} ")





            ############################################################################################################################################
            ############################################################################################################################################
            ############################################################################################################################################


            input_speech = self.wav2_processor(speech, return_tensors="pt", padding=False).input_values.squeeze().to(feats.device)
            with torch.no_grad():
                
            #     # last_hidden_state, extract_features, hidden_states
            #     # logging.warning(f" >>   {dir(last_hidden_state)}")
            #     # logging.warning(f" >>    {dir(extract_features)}")
            #     # logging.warning(f" >>  {dir(hidden_states)} ")  attention_mask=olens.squeeze()


                masks = (~make_pad_mask(feats_lengths)[:, None, :])
                xs_pad, olens = self.pretrained_embed(feats, masks)
                # logging.warning(f" input_speech {input_speech.shape} olens {olens.shape}  ")
                tmp_extract_feats = self.wav2_pretrained_model(input_speech, output_hidden_states=True, output_attentions=True, return_dict=True)
                extract_output_lengths = self.wav2_pretrained_model._get_feat_extract_output_lengths(speech_lengths)
                
                
                
                
                
                
                # logging.warning(f" >> feats_length: {feats_lengths} speech_lengths {speech_lengths}  extract_output_lengths {extract_output_lengths} ")
                
                
                # out_sum = summary(self.wav2_pretrained_model, input_data=[speech], mode="eval", col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], depth=1)
                # logging.warning(f" summary_wav2 {out_sum} ")


                
                # logging.warning(f" >>> {dir(tmp_extract_feats)}")
                # for key, value in tmp_extract_feats.items() :
                #     logging.warning(f">> Key: {key}   length: {len(value)} ")
                #     for idx in range(len(value)):
                #         logging.warning(f" {value[idx].shape}")




            # https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig
            # indexing 7th layer hidden state https://huggingface.co/transformers/v4.5.1/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
            encoder_out_tmp = tmp_extract_feats.hidden_states[6]
            encoder_out = self.disentangling_mapping_network(encoder_out_tmp)
            encoder_out_lens =  olens.squeeze(1).sum(1).to(feats.device)
            
            
            # logging.warning(f" >>>> encoder_out_pre_training {encoder_out.shape} encoder_out_trainable {encoder_out.requires_grad}    self.disentangling_mapping_network_requires_grad: {self.disentangling_mapping_network[0].weight.requires_grad} ")
            # last_hidden_states = tmp_extract_feats.last_hidden_state
            # out_sum = summary(self.encoder, input_data=[feats, feats_lengths],mode="train", col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], depth=1)
        
        ############################################################################################################################################
        ############################################################################################################################################
        ############################################################################################################################################











        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        # assert encoder_out.size(1) <= encoder_out_lens.max(), (
        #     encoder_out.size(),
        #     encoder_out_lens.max(),
        # )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens, feats, feats_lengths, aug_feats, aug_feats_lengths

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths



    def recon_decoder_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.reconstruction_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll




    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

