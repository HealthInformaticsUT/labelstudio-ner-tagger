import csv
from os.path import isdir

import matplotlib.pyplot as plt
from transformers import TrainerCallback
import time


class TsvLogWriterCallback(TrainerCallback):
    """
    writes the training logs to a .tsv file. It is done each log (after every training_arguments.logging_steps).
     The file will be located in *training_arguments.logging_dir* directory. If the directory is not found, then it does nothing.
    """
    def __init__(self):
        self.init_time = time.time()
        self.keys = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if not isdir(args.logging_dir):
            return
        if state.is_local_process_zero:
            with open(args.logging_dir + "/" + str(self.init_time) + ".tsv", "a", newline='', encoding='utf-8') as file:
                tsv_writer = csv.writer(file, delimiter="\t")
                if state.global_step == args.logging_steps:
                    self.keys = logs.keys()
                    tsv_writer.writerow(self.keys)

                if self.keys == logs.keys():
                    tsv_writer.writerow(logs.values())


class NotebookPreTrainingLogPlotCallback(TrainerCallback):
    """
    displays nsp and mlm loss plots on each log. Make sure that you use "%matplotlib notebook".
     When using it in the jupyter notebook. Otherwise it wont work.
    """

    def __init__(self):
        self.steps = []
        self.nsp_loss = []
        self.mlm_loss = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(2)
        self.fig.set_size_inches(9, 9, forward=True)
        self.fig.suptitle('Pretraining loss plots')

        self.ax1.plot([], [])
        # self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('NSP Loss')

        self.ax2.plot([], [])
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('MLM Loss')

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_local_process_zero:
            self.steps.append(logs['step'])
            self.nsp_loss.append(logs['nsp_loss'])
            self.mlm_loss.append(logs['mlm_loss'])

            for line in self.ax1.lines:
                line.set_xdata(self.steps)
                line.set_ydata(self.nsp_loss)

            for line in self.ax2.lines:
                line.set_xdata(self.steps)
                line.set_ydata(self.mlm_loss)

            self.ax1.set_xlim(0, self.steps[-1] * 1.1)
            self.ax1.set_ylim(-0.5, max(self.nsp_loss) * 1.1)

            self.ax2.set_xlim(0, self.steps[-1] * 1.1)
            self.ax2.set_ylim(-0.5, max(self.mlm_loss) * 1.1)

            self.fig.canvas.draw()