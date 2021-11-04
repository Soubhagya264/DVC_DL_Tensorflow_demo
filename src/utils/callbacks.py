import os
import time
import tensorflow as tf
import joblib
import logging
from utils.all_utils import get_time_stamp



def create_and_save_checkpoint_callback(Callbacks_dir,tensorboard_log_dir):
    pass
def create_and_save_tensorboard_callback(Callbacks_dir,tensorboard_log_dir):
    unique_name=get_time_stamp("tb_logs")
    tb_running_log_dir=os.path.join(tensorboard_log_dir,unique_name)
    tb_callbacks=tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    tb_callbacks_filepath=os.path.join(Callbacks_dir,"tb_cb.cb")
    joblib.dump(tb_callbacks,tb_callbacks_filepath)
    logging.info(f"tensorboard callback is being saved at {tb_callbacks_filepath}")


  