import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, epoch):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
                # tambahan coder untuk tampilkan epoch         
            self.add_scalar("training.loss", reduced_loss, epoch)
            self.add_scalar("grad.norm", grad_norm, epoch)
            self.add_scalar("learning.rate", learning_rate, epoch)
            self.add_scalar("duration", duration, epoch)             

    def log_validation(self, reduced_loss, model, y, y_pred, iteration, epoch):
        self.add_scalar("validation.loss.iteration", reduced_loss, iteration)
        self.add_scalar("validation.loss.epoch", reduced_loss, epoch)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
            self.add_histogram(tag, value.data.cpu().numpy(), epoch)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment iteration",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target iteration",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted iteration",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate iteration",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
         self.add_image(
            "alignment epoch",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            epoch, dataformats='HWC')
         self.add_image(
            "mel_target epoch",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            epoch, dataformats='HWC')
        self.add_image(
            "mel_predicted epoch",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            epoch, dataformats='HWC')
        self.add_image(
            "gate epoch",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            epoch, dataformats='HWC')
