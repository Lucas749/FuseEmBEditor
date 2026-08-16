#!/usr/bin/env python3
#  Copyright 2025 Jan Gorodkin, Sun Ying, Christian Anthon, Giulia Corsi
# 
#  Use of this software is governed by the Business Source License
#  included in the file licenses/BSL.txt.
# 
#  As of the Change Date specified in that file, in accordance with
#  the Business Source License, use of this software will be governed
#  by the Apache License, Version 2.0, included in the file
#  licenses/APL.txt.

import os
import pickle
import pandas as pd
import argparse as ap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, Input, utils, callbacks

#***********************************************************************************************************************

LEN_SEQ=30
LEN_GRNA=20
NTS=['Aa','TtUu','Gg','Cc']
WINDOW = {'ABE':[6,13], #3-10 in 1-based gRNA sequence
          'CBE':[6,13]} #3-10 in 1-based gRNA sequence

#***********************************************************************************************************************

def get_dataset(args):
    print('Loading train-test data')
    tx_raw = pickle.load(open(args.tx, 'rb'))
    ty_raw = pickle.load(open(args.ty, 'rb'))
    vx_raw = pickle.load(open(args.vx, 'rb'))
    vy_raw = pickle.load(open(args.vy, 'rb'))
    vdf_raw = pd.read_csv(args.vdf, sep = '\t', header = 0, compression = 'gzip')
    vdf_raw = vdf_raw[['refs', 'outcomes', 'editingeff', 'outcomefreq']]
    t_labels = pickle.load(open(args.tl, 'rb'))
    v_labels = pickle.load(open(args.vl, 'rb'))
    #filtering by label
    tx = [i[t_labels] for i in tx_raw]
    ty = ty_raw[t_labels]
    vx = [i[v_labels] for i in vx_raw]
    vy = vy_raw[v_labels]
    vdf = vdf_raw[v_labels]
    print('Training set')
    for array in tx:
        print(array.shape)
    print('target:', ty.shape)
    print('Validation set')
    for array in vx:
        print(array.shape)
    print('target:', vy.shape)
    return tx, ty, vx, vy, vdf

def get_pooling(args, d):
    if args.pt == 'avg' and d == 1:
        pooling = layers.AveragePooling1D(pool_size=args.pfs, strides=args.pfs, padding='same')
    elif args.pt == 'avg' and d == 2:
        pooling = layers.AveragePooling2D(pool_size=(args.pfs,args.pfs), strides=args.pfs, padding='same')
    elif args.pt == 'max' and d == 1:
        pooling = layers.MaxPooling1D(pool_size=args.pfs, strides=args.pfs, padding='same')
    elif args.pt == 'max' and d == 2:
        pooling = layers.MaxPooling2D(pool_size=(args.pfs,args.pfs), strides=args.pfs, padding='same')
    else: raise Exception('Pooling strategy must be avg or max, %s given' % args.pt)
    return pooling

#***********************************************************************************************************************

if __name__ == '__main__':
    parser=ap.ArgumentParser()
    parser.add_argument('-e', help = 'Name of the base editor', choices=['ABE','CBE'], required=True, type=str)
    parser.add_argument('-m', help = 'Name of the model', required=True, type=str)
    parser.add_argument('-tx', help = 'Path to training set features (pkl)', required=True, type=str)
    parser.add_argument('-ty', help = 'Path to training set target values (pkl)', required=True, type=str)
    parser.add_argument('-tl', help = 'Path to training set labels (pkl)', required=True, type=str)
    parser.add_argument('-vx', help = 'Path to validation set features (pkl)', required=True, type=str)
    parser.add_argument('-vy', help = 'Path to validation set target values (pkl)', required=True, type=str)
    parser.add_argument('-vl', help = 'Path to validation set labels (pkl)', required=True, type=str)
    parser.add_argument('-vdf', help = 'Path to validation set dataframe', required=True, type=str)
    parser.add_argument('-d', help= 'Num dataset', required=True, type=int)
    parser.add_argument('-pt', help = 'Pooling strategy', choices=['avg','max'], required=True, type=str)
    parser.add_argument('-pfs', help= 'Pooling filters and stride size', required=True, type=int)
    parser.add_argument('-act', help = 'Activation function type', choices=['relu','sigmoid','softmax'], required=True, type=str)
    parser.add_argument('-opt', help = 'Optimizer', choices=['adam','rmsprop','sgd'], required=True, type=str)
    parser.add_argument('-bs', help='Size batches', required=True, type=int)
    parser.add_argument('-ne', help='Number epochs', required=True, type=int)
    parser.add_argument('-c1n', help = 'Num filters convolution 1', required=True, type = int)
    parser.add_argument('-c2n', help = 'Num filters convolution 2', required=True, type = int)
    parser.add_argument('-c3n', help = 'Num filters convolution 3', required=True, type = int)
    parser.add_argument('-c1s', help='Size filters convolution 1', required=True, type=int)
    parser.add_argument('-c2s', help='Size filters convolution 2', required=True, type=int)
    parser.add_argument('-c3s', help='Size filters convolution 3', required=True, type=int)
    parser.add_argument('-c1d', help = 'Dropout convolution 1', required=True, type = float)
    parser.add_argument('-c2d', help = 'Dropout convolution 2', required=True, type = float)
    parser.add_argument('-c3d', help = 'Dropout convolution 3', required=True, type = float)
    parser.add_argument('-d1', help = 'Size dense layer 1', required=True, type=int)
    parser.add_argument('-d1d', help = 'Dropout convolution 1', required=True, type = float)
    parser.add_argument('-d2', help = 'Size dense layer 2', required=True, type=int)
    parser.add_argument('-d2d', help = 'Dropout convolution 2', required=True, type = float)
    parser.add_argument('-d3', help = 'Size dense layer 3', required=True, type=int)
    parser.add_argument('-d3d', help = 'Dropout convolution 3', required=True, type = float)
    parser.add_argument('-lr', help = 'learning rate', required=True, type=float)
    parser.add_argument('-l', help = 'loss', choices=['mae', 'mse'], required=True, type=str)
    parser.add_argument('-rs', help = 'random seed', required=True, type = int)
    parser.add_argument('-plot', help = 'draw plot', action='store_true')

    args = parser.parse_args()

    tx, ty, vx, vy, vdf = get_dataset(args)

    tf.random.set_seed(args.rs)
    print(f'seed:{args.rs}')
    ews, ewe = WINDOW[args.e][0], WINDOW[args.e][1] + 1

    print('Creating model')
    ins = [Input(shape = (LEN_SEQ, len(NTS), ), name = 'one_hot', dtype = 'float32'),
           Input(shape = (ewe-ews, ), name='outcome_properties', dtype='float32'),
           Input(shape = (1, ), name='energy_properties', dtype='float32'),
           Input(shape = (1, ), name='cas9', dtype='float32'),
           Input(shape = (args.d, ), name='dataset', dtype='float32')]

    conv1out = layers.Conv1D(args.c1n, args.c1s, activation=args.act, name='Conv1_1D')(ins[0])
    conv1out = layers.SpatialDropout1D(args.c1d)(conv1out)
    conv1out = get_pooling(args, 1)(conv1out)
    conv1out = layers.Flatten(name='Flat1')(conv1out)
    conv2out = layers.Conv1D(args.c2n, args.c2s, activation=args.act, name='Conv2_1D')(ins[0])
    conv2out = layers.SpatialDropout1D(args.c2d)(conv2out)
    conv2out = get_pooling(args, 1)(conv2out)
    conv2out = layers.Flatten(name='Flat2')(conv2out)
    conv3out = layers.Conv1D(args.c3n, args.c3s, activation=args.act, name='Conv3_1D')(ins[0])
    conv3out = layers.SpatialDropout1D(args.c3d)(conv3out)
    conv3out = get_pooling(args, 1)(conv3out)
    conv3out = layers.Flatten(name='Flat3')(conv3out)

    concat = layers.concatenate([conv1out, conv2out, conv3out])

    dense1out = layers.Dense(args.d1, activation=args.act, name='Dense1')(concat)
    dense1out = layers.Dropout(args.d1d, name='Dropout1')(dense1out)
    dense1out_concat = layers.concatenate([dense1out, ins[1], ins[2], ins[3], ins[4]])
    dense2out = layers.Dense(args.d2, activation=args.act, name = 'Dense2')(dense1out_concat)
    dense2out = layers.Dropout(args.d2d, name = 'Dropout2')(dense2out)
    dense3out = layers.Dense(args.d3, activation=args.act, name = 'Dense3')(dense2out)
    dense3out = layers.Dropout(args.d3d, name='Dropout3')(dense3out)
    outsout = layers.Dense(2, name = 'Output')(dense3out)

    print('Compiling model')
    model = Model(inputs=ins, outputs=outsout)
    model.summary()
    if args.opt == 'adam': optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    elif args.opt == 'rmsprop': optimizer = keras.optimizers.RMSprop()
    elif args.opt == 'sgd': optimizer = keras.optimizers.SGD()
    else: raise Exception('Optimizers can be adam, SGD, or rmsprop, %s given' % args.opt)
    model.compile(loss = args.l, optimizer = optimizer)
    if args.plot:
        print('draw_plot')
        utils.plot_model(model, to_file=args.m+'.png', show_shapes=True, dpi=600)

    print('Training model')
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights = True)
    mc = callbacks.ModelCheckpoint(args.m, verbose=1, save_best_only=True)
    history = model.fit(tx, ty, validation_data = (vx, vy), sample_weight = None, batch_size = args.bs, epochs = args.ne, use_multiprocessing = True, workers = 32, verbose = 2, callbacks = [es,mc])

    print('Predict validation set')
    test_pred = model.predict(vx)
    vdf.insert(vdf.shape[1], 'predicted_editingeff', test_pred.T[0])
    vdf.insert(vdf.shape[1], 'predicted_outcomefreq', test_pred.T[1])
    output = 'test_result.tsv.gz'
    vdf.to_csv(output, sep = '\t', header = True, index = False, compression = 'gzip')
    print('Done!')
