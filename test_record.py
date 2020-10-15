#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.algos import amplitude_mask, misi, bregmisi
from helpers.data_io import load_src, record_src
from helpers.stft import my_stft
from open_unmx.estim_spectro import estim_spectro_from_mix


def test_record(params):

    # Load the optimal step sizes from validation
    gd_step_opt = np.load('outputs/val_gd_step.npz')['gd_step']
    ib = 5
    b = 1.25
    my_step = gd_step_opt[ib, 1, 1, :]

    # Loop over iSNRs, mixtures and parameters
    for jsnr, isnr in enumerate(params['input_SNR_list']):
        for n in range(params['n_mix']):
            print('iSNR ' + str(jsnr + 1) + ' / ' + str(len(params['input_SNR_list'])) +
                  ' -- Mix ' + str(n + 1) + ' / ' + str(params['n_mix']))
            # Start from mixture number 50 (since the first 50 are for validation)
            i_n = n + params['n_mix']
            # Load data and get estimated spectrograms
            audio_path = 'data/SNR_' + str(isnr) + '/' + str(i_n) + '/'
            src_ref, mix = load_src(audio_path, params['sample_rate'])
            mix_stft = my_stft(mix, n_fft=params['n_fft'], hop_length=params['hop_length'],
                               win_length=params['win_length'], win_type=params['win_type'])[:, :, 0]

            # Estimate the magnitude spectrograms
            spectro_mag = estim_spectro_from_mix(mix)

            # Amplitude mask
            src_est_am = amplitude_mask(spectro_mag, mix_stft, win_length=params['win_length'],
                                        hop_length=params['hop_length'], win_type=params['win_type'])
            record_src(audio_path + 'am_', src_est_am, params['sample_rate'])

            # MISI
            src_est_misi = misi(mix_stft, spectro_mag, win_length=params['win_length'], hop_length=params['hop_length'],
                                max_iter=params['max_iter'])[0]
            record_src(audio_path + 'misi_', src_est_misi, params['sample_rate'])

            # Gradient descent (beta=1.25 d=2, left)
            spectro_pow = np.power(spectro_mag, 2)
            src_est_gd = bregmisi(mix_stft, spectro_pow, win_length=params['win_length'],
                                  hop_length=params['hop_length'], beta=b, d=2, grad_step=my_step[jsnr], direc='left',
                                  max_iter=params['max_iter'])[0]
            record_src(audio_path + 'gd_', src_est_gd, params['sample_rate'])

    return


if __name__ == '__main__':

    # Parameters
    params = {'sample_rate': 16000,
              'win_length': 1024,
              'hop_length': 256,
              'n_fft': 1024,
              'win_type': 'hann',
              'max_iter': 5,
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              'beta_range': np.linspace(0, 2, 9)
              }

    test_record(params)

# EOF

