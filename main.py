#!/usr/bin/env python
# encoding: utf-8
import torch
from torch.utils.data import DataLoader

import os
import pickle
import argparse
import logging as log

import models
import importlib
from dataset import Dataset
from dataset import Dataset_empty
import numpy as np
import random
import math

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

##################################################################

parser = argparse.ArgumentParser(description='Hidden Graph Model')
parser.add_argument('--debug',          action='store_true',        help='log debug messages or not')
parser.add_argument('--run_exist',      action='store_true',        help='run dir exists ok or not')
parser.add_argument('--run_dir',        type=str,   default='None', help='dir to save log and models')
parser.add_argument('--data_dir',       type=str,   default='data/4time_mini_09/')

parser.add_argument('--log_every',      type=int,   default=0,      help='number of steps to log loss, do not log if 0')
parser.add_argument('--eval_every',     type=int,   default=0,      help='number of steps to evaluate, only evaluate after each epoch if 0')
parser.add_argument('--save_every',     type=int,   default=5,      help='number of steps to save model')
parser.add_argument('--device',         type=int,   default=-1,      help='gpu device id, cpu if -1')

parser.add_argument('--n_layer',type=int,   default=1,      help='number of mlp hidden layers in decoder')
parser.add_argument('--dim',type=int,   default=64,     help='hidden size for nodes')

parser.add_argument('--n_epochs',       type=int,   default=200,   help='number of epochs to train')

parser.add_argument('--batch_size',     type=int,   default=32,      help='number of instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-3,   help='learning rate')
parser.add_argument('--gen_lr',             type=float, default=1e-3,   help='learning rate of generator')
parser.add_argument('--disc_lr',             type=float, default=5e-3,   help='learning rate of discriminator')
parser.add_argument('--dropout',        type=float, default=0.0,   help='dropout') 
parser.add_argument('--seq_len',       type=int, default=200,   help='the length of the sequence') 
parser.add_argument('--gamma',        type=float, default=0.93,   help='graph_type') #
parser.add_argument('--lamb',        type=float, default=1e-4,   help='graph_type') #
parser.add_argument('--cog_levels',        type=int, default=10,   help='the response action space for cognition estimation')
parser.add_argument('--acq_levels',        type=int, default=10,   help='the response action space for  sensitivity estimation')


parser.add_argument('--g',        type=float, default=1,   help='weights on generator')
parser.add_argument('--e',        type=float, default=1,   help='weights on expert learner hidden state gap')
parser.add_argument('--r',        type=float, default=1,   help='weights on rl loss')
parser.add_argument('--k',        type=float, default=1,   help='weights on kt loss')

parser.add_argument('--plan',        type=str, default='2', help='the training plan') 
parser.add_argument('--data_gen',          type=str,   default='ques_seq_gen',   help='run model')
parser.add_argument('--model',          type=str,   default='ques_seq_gen',   help='run model')
parser.add_argument('--discriminator',  type=str,   default='disc_aktcl_1',   help='run model') 
parser.add_argument('--generator',  type=str,   default='gen38_m3_21',   help='run model')

parser.add_argument('--s_alpha',  type=float,   default=1,   help='the number of attention heads')
parser.add_argument('--alpha',  type=float,   default=0.05,   help='hyperparameter of emb constrain')

parser.add_argument('--multi_len',  type=int,   default=30,   help='the max length for testing continuous prediction') 
parser.add_argument('--neg_reward',  type=int,   default=-10,   help='the negtive reward when the continuous is cutted down (<=0)')
parser.add_argument('--attention_heads',  type=int,   default=1,   help='the number of attention heads') 
parser.add_argument('--train_en_epoch',  type=int,   default=1,   help='the epoch of training the encoder before train the gail') 
parser.add_argument('--dis_train_every',        type=int, default=2,   help='training discriminator every n epoch')
parser.add_argument('--dis_train_epoch',        type=int, default=1,   help='training epoch of discriminator')
parser.add_argument('--gen_para',        type=float, default=2,   help='the parameter of gen_dis rl loss')
parser.add_argument('--action',type=str,  default= 'train', choices=['continue', 'sta', 'train', 'test', 'oracle_gen', 'uniform_gen', 'visualize'],    help='the task to complite')
parser.add_argument('--checkpoint_path',type=str,  default= 'run/1/params_0.pt',   help='the path of checkpoint')
parser.add_argument('--reverse_ratio',type=float,  default= '0.05',   help='the ratio of reverse operate')
parser.add_argument('--pair_ratio',type=float,  default= '1e4',   help='the ratio of pair loss')
parser.add_argument('--disc_thres',type=list, default= [0.56, 0.45],   help='stop crition of disc for real acc and fake acc') 
parser.add_argument('--gen_thres',type=list,  default= [0.85, 0.65],   help='stop crition of gen for real acc and fake acc') 
parser.add_argument('--disc_stop',type=float, default= 0.05,   help='stop crition of disc for real acc and fake acc') 
parser.add_argument('--gen_start',type=list,  default= 0.45,   help='stop crition of gen for real acc and fake acc') 
parser.add_argument('--op_type_num',type=int,  default= 20,   help='extend the number type of operation ') 
parser.add_argument('--train_sample',  type=int,   default=6000,   help='the number of samples used for training')

parser.add_argument('--mask_prob',        type=float, default=0.6,   help='hyper parameter for loss')
parser.add_argument('--replace_prob',  type=float, default=0.3,   help='the head of attention') 
parser.add_argument('--crop_ratio',        type=float, default=0.6,   help='hyper parameter for loss')
parser.add_argument('--perm_ratio',  type=float, default=0.6,   help='the head of attention') 
parser.add_argument('--contrast_num',  type=int, default=200,   help='the head of attention') 



args = parser.parse_args() 

if args.debug:
    args.run_exist = True
    args.run_dir = 'debug'


dataset = ['ass12', 'slepemapy', 'ednet', 'junyi']
for dtname in dataset:
    if dtname in args.data_dir:
        break
run_path = 'run/'+ dtname + '/different_sample_num/train_500/plan' + \
            args.plan + '_' + args.discriminator + '_'+ args.generator + '/'
if args.run_dir == 'None':
    args.run_dir  = run_path
os.makedirs(args.run_dir, exist_ok=args.run_exist)
print(args.run_dir)


log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.DEBUG if args.debug else log.INFO)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, 'log.txt'), mode='w'))
log.info('args: %s' % str(args))
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

def preprocess():
    datasets = {}

    if args.action in ['train', 'sta', 'continue']:

        splits = ['train', 'valid', 'test']
        with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, concept_number, max_concept_of_problem = pickle.load(fp)
        setattr(args, 'max_concepts', max_concept_of_problem)
        setattr(args, 'concept_num', concept_number)
        setattr(args, 'problem_number', problem_number)

        for split in splits:
            file_name = os.path.join(args.data_dir, 'dataset_%s.pkl' % split)
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    # x = pickle.load(f)
                    datasets[split] = pickle.load(f)
                log.info('Dataset split %s loaded' % split)
            else:
                datasets[split] = Dataset(args.problem_number, args.concept_num, args.train_sample, root_dir=args.data_dir, split=split)
                with open(file_name, 'wb') as f:
                    pickle.dump(datasets[split], f)
                log.info('Dataset split %s created and dumpped' % split)

        loaders = {}
        for split in splits:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=args.batch_size,
                collate_fn=datasets[split].collate,
                shuffle=True if split == 'train' else False
            )

        return loaders
    elif args.action == 'test':
        with open(args.test_data_path, 'rb') as f:
            datasets = pickle.load(f)
            log.info('Dataset  loaded: %s' % args.test_data_path)
        
        loaders = DataLoader(
            datasets,
            batch_size=args.batch_size,
            collate_fn=datasets.collate,
            shuffle=False
        )
        return loaders
    
    elif args.action == 'oracle_gen':
        
        oracle_train = Dataset_empty(args.problem_number, args.concept_num)
        oracle_valid = Dataset_empty(args.problem_number, args.concept_num)
        oracle_test = Dataset_empty(args.problem_number, args.concept_num)

        return [oracle_train, oracle_valid, oracle_test]

    elif args.action == 'uniform_gen':
        
        with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, concept_number, max_concept_of_problem = pickle.load(fp)
        setattr(args, 'max_concepts', max_concept_of_problem)
        setattr(args, 'concept_num', concept_number)
        setattr(args, 'problem_number', problem_number)


        file_name = args.data_dir + 'dataset_uniform_raw.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                datasets = pickle.load(f)
            log.info('Uniform raw data loaded')
        else:
            datasets = Dataset(args.problem_number, args.concept_num, args.train_sample, root_dir=args.data_dir, split='uniform_raw')
            with open(file_name, 'wb') as f:
                pickle.dump(datasets, f)
            log.info('Uniform raw data  created and dumpped')

        raw_loader = DataLoader(
            datasets,
            batch_size=args.batch_size,
            collate_fn=datasets.collate,
            shuffle=False
        )

        empty_loader = Dataset_empty(args.problem_number, args.concept_num)
        return raw_loader, empty_loader
        # return raw_loader

if __name__ == '__main__':
    if args.action == 'test':
        if args.checkpoint_path == 'None':
            raise IOError
        print('obtain graph')
        '''test'''
        model, generater = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
        
        model.device = args.device
        generater.device = args.device
        plan = generater.plan
        setattr(args, 'max_concepts', model.max_concepts)
        setattr(args, 'concept_num', model.concept_num)
        setattr(args, 'problem_number', model.ques_num)
        setattr(args, 'batch_size', generater.batch_size)
        log.info(str(vars(args)))
        # setattr(args, 'prob_dim', int(math.log(problem_number,2)) + 1)

        eval_func = model.forward
        if args.test_type == 'multi':
            eval_func = model.forward_gen
        loader = preprocess()
        train_module = importlib.import_module('train_' + plan)
        acc, auc, _ = train_module.evaluate(model, loader, args, eval_func)
        log.info('test_acc: {:.7f}, test_auc: {:.7f}'.format(acc, auc))
    
    elif args.action == 'continue':
        if args.checkpoint_path == 'None':
            raise IOError
        print('continue train...')
        '''test'''
        generater, disc = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
        
        disc.device = args.device
        generater.device = args.device
        # plan = generater.plan
        setattr(args, 'max_concepts', generater.max_concept)
        setattr(args, 'concept_num', generater.concept_num)
        setattr(args, 'problem_number', generater.problem_num)
        setattr(args, 'batch_size', generater.batch_size)
        log.info(str(vars(args)))
        
        loaders = preprocess()
        train_module = importlib.import_module('train_' + plan)
        train_module.train(model, loaders, args)

    elif args.action == 'oracle_gen':
        if args.checkpoint_path == 'None':
            raise IOError
        model, generater = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
        model.device = args.device
        generater.device = args.device

        plan = generater.plan
        setattr(args, 'max_concepts', model.max_concepts)
        setattr(args, 'concept_num', model.concept_num)
        setattr(args, 'problem_number', model.ques_num)
        setattr(args, 'batch_size', generater.batch_size)

        args.data_dir = generater.data_dir
        args.run_dir = args.checkpoint_path.split('params_')[0]
        args.model = generater.model

        log.info(str(vars(args)))

        loader = preprocess()
        train_module = importlib.import_module('train_' + args.plan)
        train_module.generate_oracle(model, generater, args, loader)
    
    elif args.action == 'uniform_gen':
        if args.checkpoint_path == 'None':
            raise IOError
        try:
            generater, discriminator = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
        except:
            _, generater, discriminator = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
        discriminator.device = args.device
        generater.device = args.device

        log.info(str(vars(args)))

        uniform_loader, empty_loader = preprocess()
        train_module = importlib.import_module('train_' + args.plan)
        train_module.generate_uniform(discriminator, generater, args, uniform_loader, empty_loader)

    elif args.action  == 'train':
        loaders = preprocess()
        Model = getattr(models, args.model) 
        model = Model(args).to(args.device)
        log.info(str(vars(args)))
        train_module = importlib.import_module('train_' + args.plan)
        train_module.train(model, loaders, args)

    elif args.action == 'sta': # count the accuracy of dataset
        loaders = preprocess()
        train_module = importlib.import_module('train_' + args.plan)
        train_module.count_acc_from_y(loaders, args)


    elif args.action == 'visualize':
        with open(args.visualize_data, 'rb') as fp:
            x, label = pickle.load(fp)
        visual_path = args.visualize_data.split('.')[0] + '_revisual.pdf'
        scatter.draw_scatter(x, label, visual_path, dim_=2)


    
