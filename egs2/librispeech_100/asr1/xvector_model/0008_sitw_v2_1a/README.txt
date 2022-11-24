
 UPLOADER        David Snyder 
 DATE            2018-05-30
 KALDI VERSION   447e964

 This directory contains files generated from the recipe in egs/sitw/v2/.
 It's contents should be placed in a similar directory, with symbolic links to
 utils/, sid/, steps/, etc.  This was created when Kaldi's master branch was
 at git log 447e96498e865d1c7e17e702ab3eb3dffee21bb3.

 This recipe replaces i-vectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "x-vectors."  The recipe in local/nnet3/xvector/tuning/run_xvector_1a.sh is
 closesly based on "X-vectors: Robust DNN Embeddings for Speaker Recognition."


 I. Files list
 ------------------------------------------------------------------------------
 
 ./
     README.txt               This file
     run.sh                   A copy of the egs/sitw/v2/run.sh
                              at the time of uploading this file.  Look at this
                              to see examples of computing features and 
                              extracting x-vectors.

 local/nnet3/xvector/tuning/
     run_xvector_1a.sh        This is the default recipe, at the time of
                              uploading this resource.  The script generates
                              the configs, egs, and trains the model.

 conf/
     vad.conf                 The energy-based VAD configuration
     mfcc.conf                A wideband MFCC configuration

 exp/xvector_nnet_1a/ 
     final.raw                The pretrained DNN model
     nnet.config              The nnet3 config file that was used when the
                              DNN model was first instantiated.
     extract.config           Another nnet3 config file that modifies the DNN
                              final.raw to extract x-vectors.  It should be
                              automatically handled by the script 
                              extract_xvectors.sh.
     min_chunk_size           Min chunk size used (see extract_xvectors.sh)
     max_chunk_size           Max chunk size used (see extract_xvectors.sh)
     srand                    The RNG seed used when creating the DNN

 exp/xvector_nnet_1a/xvectors_train_combined_200k/
     mean.vec                 Vector for centering
     transform.mat            Whitening matrix
     plda                     PLDA model

 II. Citation
 ------------------------------------------------------------------------------

 If you wish to use this system in a publication, please cite
 "X-vectors: Robust DNN Embeddings for Speaker Recognition."  The
 recipe is closely based on that paper.  The main difference is that
 here we use exclusively wideband training data.  The bibtex is as follows:

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 III. Corpora
 ------------------------------------------------------------------------------

 The pretrained model used the following datasets for training.  Note that
 there are 60 speakers which overlap between VoxCeleb 1 and our evaluation
 dataset, Speakers in the Wild.  Those 60 speakers were removed from
 VoxCeleb 1 prior to training.

 Evaluation
     
     Speakers in the Wild    http://www.speech.sri.com/projects/sitw

 System Development
     
     VoxCeleb 1              http://www.robots.ox.ac.uk/~vgg/data/voxceleb
     VoxCeleb 2              http://www.robots.ox.ac.uk/~vgg/data/voxceleb2
     MUSAN                   http://www.openslr.org/17
     RIR_NOISES              http://www.openslr.org/28
