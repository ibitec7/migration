import logging
import matplotlib.pyplot as plt
import numpy as np
import os

global PLOTS_DIR
PLOTS_DIR = './data/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def setup_logger(log_file, log_level=logging.INFO, write_console=True, write_file=True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    if write_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if write_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def save_figure(fig, name, tight_layout=True, logger=None):
    """Save figure to plots directory with consistent naming"""
    if tight_layout:
        fig.tight_layout()
    filepath = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {os.path.basename(filepath)}")
    plt.close(fig)


def add_title_and_save(fig, title, subtitle, filename, logger=None):
    """Add title and save figure"""
    fig.suptitle(title, fontsize=15, y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=11, style='italic', color='gray')
    save_figure(fig, filename, logger=logger)


def annotate_max_point(ax, x_data, y_data, **kwargs):
    """Annotate maximum point on a line plot"""
    max_idx = np.argmax(y_data)
    ax.annotate('Peak',
                xy=(x_data[max_idx], y_data[max_idx]),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=9)