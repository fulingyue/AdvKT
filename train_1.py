from logging.handlers import RotatingFileHandler
import os
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import accuracy_score
# import pytorch_lightning.metrics.functional as light_func
import logging as log
import numpy
import tqdm
import pickle
from utils import batch_data_to_device
import models
import shutil
import random
from torch.autograd import Variable
import torch.autograd as autograd
cos_simi = torch.nn.CosineSimilarity(dim = 2, eps=1e-6)
sigmoid = torch.nn.Sigmoid()

def gradient_penalty(discriminator, real_vec_org, fake_vec_org, device):
    # alpha = tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    sample_num = min(real_vec_org.size()[0], fake_vec_org.size()[0])
    real_vec = real_vec_org[0: sample_num]
    fake_vec = fake_vec_org[0: sample_num]

    alpha = torch.rand(sample_num).to(device).unsqueeze(-1)


    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_vec + ((1 - alpha) * fake_vec)).requires_grad_(True)
    d_interpolates = sigmoid(
                            discriminator.predictor(
                                interpolates))
    fake = Variable(torch.ones(sample_num, 1).to(device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=fake,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(model, loaders, args):
    log.info("training...")
    one = torch.tensor(1.0).to(args.device)
    zero = torch.tensor(0.0).to(args.device)
    test_batch = None

    Disc = getattr(models, args.discriminator)
    discriminator = Disc(args).to(args.device)

    Gen = getattr(models, args.generator)
    generator = Gen(args).to(args.device)

    D_Gen = getattr(models, args.data_gen)
    data_gen = D_Gen(args).to(args.device)

    BCELosswithlog = torch.nn.BCEWithLogitsLoss()
    BCEloss = torch.nn.BCELoss()
    kl_loss = torch.nn.KLDivLoss()
    l1_loss = torch.nn.SmoothL1Loss()
    mse_loss = torch.nn.MSELoss()
    

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.gen_lr)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)
    train_sigmoid = torch.nn.Sigmoid()
    train_len = len(loaders['train'].dataset)
    '''freeze discri'''
    train_disc = 1
    for epoch in range(args.n_epochs):
        loss_all = 0
        dis_loss = 0.0
        
        for step, data in enumerate(loaders['train']):
            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            
            if step % args.dis_train_every == 0:
                dis_gen_label_sum, real_label_sum, gen_ques_label_sum, count = 0, 0, 0, 0
                discriminator.train()
                generator.eval()

                pos_score_list = []
                r_score, ques_loss, resp_loss, real_mlp_vec = discriminator.obtain_score(x)
                
                constrain_loss = ques_loss + resp_loss
                
                pos_score_list.append(r_score)
                pos_aug_data = discriminator.data_aug(x)
                
                for pos_x in pos_aug_data:
                    this_pos_score, _, _, _ = discriminator.obtain_score(pos_x)
                    pos_score_list.append(this_pos_score)
                
                
                gen_batch_size = max(int(len(y) / 40), 1)
                total_record_num = len(y)
                gen_batch = data_gen.gen(gen_batch_size, total_record_num)[0]
                gen_x = generator.gen_seq(gen_batch, take_index = 1)
                gen_score, _, _, gen_mlp_vec = discriminator.obtain_score(gen_x)

                gen_real_ques_score, gen_ques_part, gen_ques_pos_one, gen_ques_mlp_vec = discriminator.obtain_score_reverse(x)
                
                real_score = torch.mean(r_score).detach()
                fake_score = torch.mean(gen_score).detach()
                gen_ques_score = torch.mean(gen_real_ques_score).detach()
                # fake_score = torch.mean(gen_real_ques_score).detach()
                '''gradient penalty'''
                real_gen_gp = gradient_penalty(discriminator, real_mlp_vec, gen_mlp_vec, args.device)
                real_genques_gp = gradient_penalty(discriminator, real_mlp_vec, gen_ques_mlp_vec, args.device)
                all_gp = 10 * (real_gen_gp + real_genques_gp)

                '''ranking loss'''
                pos_score = torch.cat(pos_score_list, dim = 0).squeeze(-1)
                fake_score = torch.cat([gen_real_ques_score, gen_score.squeeze(-1)], dim = 0)
                pos_fake_rank = torch.mean(pos_score.unsqueeze(1) - fake_score.unsqueeze(0))
                real_posgen_rank = torch.mean(pos_score_list[0] - \
                                            torch.cat(pos_score_list[1:], dim = 0).transpose(1, 0))

                short_score = discriminator.obtain_score_short()
                
                disc_bce = - torch.mean(short_score) + 10
                
                dis_loss = - pos_fake_rank - real_posgen_rank + disc_bce + constrain_loss + all_gp

                optimizer_dis.zero_grad() 
                dis_loss.backward()
                optimizer_dis.step()

                real_label_sum += real_score
                dis_gen_label_sum += fake_score
                gen_ques_label_sum += gen_ques_score
 
            dis_gen_label_sum, real_label_sum, count = 0, 0, 0

            generator.train()
            discriminator.eval()

            '''real performance on gen'''
            real_logits, real_rl_loss, real_ques_loss, real_qas = generator.supervised(x, take_index  = 1)
            real_score = torch.mean(discriminator.obtain_score(x)[0].detach())   

            real_loss = BCEloss(train_sigmoid(real_logits), y) + real_ques_loss

            _, _, rrr = generator.forward_gen(x, discriminator, take_index = 0)
            rrr_score = torch.mean(rrr.detach())

            '''train kt & generator'''
            '''generate data'''

            # gen_batch_size = int(len(y) / 40)
            gen_batch_size = max(int(len(y) / 40), 1)
            total_record_num = len(y)
            gen_batch = data_gen.gen(gen_batch_size, total_record_num)[0] 
            
            gen_logits, gen_rl_loss, gen_score = generator.forward_gen(gen_batch, discriminator)
            fake_score = torch.mean(gen_score.detach())
            gen_loss = args.pair_ratio * (gen_rl_loss) + real_loss

            optimizer_gen.zero_grad() 
            gen_loss.backward()
            optimizer_gen.step()
            step += 1
            with torch.no_grad():
                loss_all += gen_loss.detach().item()

            count += 1
            dis_gen_label_sum += fake_score
            # real_label_sum += real_score
            real_label_sum += rrr_score



        fake_pos_ratio = dis_gen_label_sum / count
        real_pos_ratio = real_label_sum / count
        if fake_pos_ratio > args.gen_thres[1]:
            train_disc = 1
        show_loss = loss_all / train_len
        generator.eval()
        vmacc, vmauc, _ = evaluate(model, loaders['valid'], args, generator.evaluate, discriminator)
        tmacc, tmauc, _ = evaluate(model, loaders['test'], args, generator.evaluate, discriminator)
        tmcacc, tmcauc, _ = evaluate(model, loaders['test'], args, generator.evaluate_cut, discriminator)
        log.info('Epoch: {:03d}, pos_real: {:.4f}, pos_gen: {:.4f}, G_Loss: {:.7f}, D_loss: {:.7f}, v_macc: {:.7f}, v_mauc: {:.7f},t_macc: {:.7f}, t_mauc: {:.7f}, t_mcacc: {:.7f}, t_mcauc: {:.7f}'.format(
                    epoch, real_pos_ratio, fake_pos_ratio, show_loss, dis_loss, vmacc, vmauc, tmacc, tmauc, tmcacc, tmcauc))
            
     
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save([generator, discriminator], os.path.join(args.run_dir, 'params_%i.pt' % epoch))

def evaluate(model, loader, args, eval_funcs, discriminator):
    model.eval()
    rre_list = []
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, hat_y_list, label_list = [], [], []
    with torch.no_grad():
        for data in loader:
            # x, y = data
            x, y = batch_data_to_device(data, args.device)
       
            hat_y_prob, labels = eval_funcs(x, discriminator)
            y_list.append(y)
            hat_y_list.append(eval_sigmoid(hat_y_prob))
            label_list.append(labels)
    y_tensor = torch.cat(y_list, dim = 0).int()
    hat_y_prob_tensor = torch.cat(hat_y_list, dim = 0)
    # acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.2).int().cpu().numpy())
    label_tensor = torch.cat(label_list, dim = 0).squeeze(-1)
    acc = accuracy_score(y_tensor.cpu().numpy(), label_tensor.int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    auroc = 0
    
    return acc, auc, auroc




