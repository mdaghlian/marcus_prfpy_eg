import numpy as np
import os
opj = os.path.join
import scipy.io as sio
import scipy.interpolate as interpolate

# Just copied from the matlab files
# inside <mr_vista_dir_for_a_sub>/Stimuli/params.mat
path_to_mrvista_dot_mat = ''
mrv_dm = sio.loadmat(
    path_to_mrvista_dot_mat,
    squeeze_me=True,
    struct_as_record=False,
    )

prf_images = mrv_dm['stimulus'].images           # 1080 x 1080 x 801, different frames/textures for PRF bar
prf_seq = mrv_dm['stimulus'].seq                 # order of the frames presented (each is an index for prf_images)
total_n_frames = prf_seq.shape[-1]
relevant_seq_id = np.arange(0,total_n_frames, 30)
relevant_frames_id = prf_seq[relevant_seq_id] - 1 # *** -1 because the seq was written in matlab
hires_tr_images = prf_images[:,:,relevant_frames_id]
hires_binary_dm = (hires_tr_images!=128)*1.0 #keep as float (BINARIZE)
n_pix = 100 # NUMBER OF PIXELS FOR DOWNSAMPLED DM

prf_design_matrix = resample2d(hires_binary_dm, n_pix) # Downsample

