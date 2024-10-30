"""Training and evaluation for score-based generative models. """

from dataclasses import dataclass
import gc
import io
import os
import time

import numpy as np
# import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ncsnpp, ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
# import evaluation
import sde_lib
from absl import flags
import torch
from torch import nn
# from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils.utils import *
import utils.datasets as datasets
from utils.datasets import PSNR

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    # tf.io.gfile.makedirs(sample_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    # tf.io.gfile.makedirs(tb_dir)
    # writer = tensorboard.SummaryWriter(tb_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    print(score_model)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # tf.io.gfile.makedirs(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Resume training when intermediate checkpoints are detected
    # state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, 'training')
    num_data = len(train_dl.dataset)

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'petsde':
        sde = sde_lib.PETSDE(config)
        sampling_eps = 1e-5  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(config, sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    # eval_step_fn = losses.get_step_fn(config, sde, train=False, optimize_fn=optimize_fn,
    #                                   reduce_mean=reduce_mean, continuous=continuous,
    #                                   likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        # sampling_shape = (config.training.batch_size, config.data.num_channels,
        #                   config.data.image_size, config.data.image_size)
        # sampling_fn = sampling.get_sampling_fn(
        #     config, sde, sampling_shape, inverse_scaler, sampling_eps)
        pass

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            ###########################################
            pet, mri = batch
            # print('pet=',pet)
            # # TODO: mask condition
            # label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x320x320
            # label = c2r(label).type(torch.FloatTensor).to(config.device)
            # label = scaler(label)
            # loss = train_step_fn(state, label)
            ###########################################
            pet = scaler(pet).to(config.device)
            mri = scaler(mri).to(config.device)
            # save_mat('.', batch, 'label', 0, False)

            # Execute one training step
            loss = train_step_fn(state, pet, mri)
            loss_sum += loss

            param_num = sum(param.numel()
                            for param in state["model"].parameters())
            if step % 10 == 0:
                print('Epoch', epoch + 1, '/', config.training.epochs, 'Step', step,
                        'loss = ', loss.cpu().data.numpy(),
                        'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                        'time', time.time() - t0, 'param_num', param_num)

            if step % config.training.log_freq == 0:
                # logging.info("step: %d, training_loss: %.5e" %
                #              (step, loss.item()))
                # global_step = num_data * epoch + step
                # writer.add_scalar(
                #     "training_loss", scalar_value=loss, global_step=global_step)
                pass

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0:
                pass

        # Save a checkpoint for every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth'), state)

        # Generate and save samples for every epoch
        if config.training.snapshot_sampling and (epoch + 1) % config.training.snapshot_freq == 0:
            # config.sampling.ckpt = epoch + 1
            # sample_dir = ""
            pass


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    pet_score_model = mutils.create_model(config)
    pet_optimizer = losses.get_optimizer(config, pet_score_model.parameters())
    pet_ema = ExponentialMovingAverage(pet_score_model.parameters(), decay=config.model.ema_rate)
    pet_state = dict(optimizer=pet_optimizer, model=pet_score_model, ema=pet_ema, step=0)
    
    # FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.config.sampling.folder)
    pet_checkpoint_dir = os.path.join(workdir, FLAGS.config.sampling.folder, "checkpoints")
    pet_ckpt_path = os.path.join(pet_checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
    pet_state = restore_checkpoint(pet_ckpt_path, pet_state, device=config.device)
    print("load pet weights:", pet_ckpt_path)
    
    if not config.training.joint:
        mri_score_model = mutils.create_model(config)
        mri_optimizer = losses.get_optimizer(config, pet_score_model.parameters())
        mri_ema = ExponentialMovingAverage(mri_score_model.parameters(), decay=config.model.ema_rate)
        mri_state = dict(optimizer=mri_optimizer, model=mri_score_model, ema=mri_ema, step=0)
        mri_checkpoint_dir = os.path.join(workdir, FLAGS.config.sampling.mri_folder, "checkpoints")
        mri_ckpt_path = os.path.join(mri_checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
        mri_state = restore_checkpoint(mri_ckpt_path, mri_state, device=config.device)
        print("load mri weights:", mri_ckpt_path)
    else:
        mri_score_model = None

    SAMPLING_FOLDER_ID = '_'.join(['ckpt', str(config.sampling.ckpt),
                        FLAGS.config.sampling.predictor, 
                        FLAGS.config.sampling.corrector,
                        str(config.sampling.snr),
                        'predictor_mse', str(FLAGS.config.sampling.mse),
                        'corrector_mse', str(FLAGS.config.sampling.corrector_mse),
                        str(FLAGS.config.model.beta_max)])
    # Build data pipeline
    test_dl = datasets.get_dataset(config, 'sample') # mode=test:90多张图，modex=sample:一张图，第十张

    psnr = PSNR()
    FLAGS.config.sampling.folder = os.path.join(workdir, FLAGS.config.sampling.folder, SAMPLING_FOLDER_ID)
    # tf.io.gfile.makedirs(FLAGS.config.sampling.folder)
    if not os.path.exists(FLAGS.config.sampling.folder):
        os.makedirs(FLAGS.config.sampling.folder)

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'petsde':
        sde = sde_lib.PETSDE(config)
        sampling_eps = 1e-5  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")


    # Build the sampling function when sampling is enabled
    sampling_shape = (config.sampling.batch_size, 1,
                                config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                inverse_scaler, sampling_eps)

    best_psnr = 0.0
    for index, point in enumerate(test_dl):
        print('---------------------------------------------')
        print('---------------- point:', index, '------------------')
        print('---------------------------------------------')
        pet, mri,file_name = point
        
        
        # # TODO: mask condition
        # label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x320x320
        # label = c2r(label).type(torch.FloatTensor).to(config.device)
        # label = scaler(label)
        # loss = train_step_fn(state, label)
        ###########################################
        pet = scaler(pet).to(config.device)
        mri = scaler(mri).to(config.device)

        save_mat_label('./label_mat', pet, 'pet', file_name, 0, False)
        save_mat_label('./label_mat', mri, 'mri', file_name, 0, False)

        # print('pet=',pet)
        # print('mri=',mri)
        recon = sampling_fn(pet_score_model, mri_score_model, mri)

        test_psnr = psnr(pet, recon)

        if test_psnr > best_psnr:
            best_psnr = test_psnr
        print('[filename] test_psnr: %.5f' % test_psnr)
        print('[filename] best_psnr: %.5f' % best_psnr)

        save_mat_label('./label_mat', recon, 'recon', file_name, index, normalize=False)