from functools import cmp_to_key
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
import modules
import numpy as np
import pickle
import random

class ques_seq_gen(nn.Module): #
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.device = args.device
        self.max_concept = args.max_concepts
        self.max_len = 200
        self.min_len = 150
        self.seq_len = args.seq_len
        self.ques_num = args.problem_number
        self.ques_concept_relation = None

        self.ones = torch.tensor(1)
        self.zeros = torch.tensor(0)

        '''load ques_concept relation'''
        with open(args.data_dir + 'problem_skills_relation.pkl', 'rb') as fp:
            ques_concept_relation = pickle.load(fp)
        ques_concept_relation_list = [[0] * self.max_concept]
        for i in range(1, len(ques_concept_relation) + 1):
            this_append = ques_concept_relation[i]
            while(len(this_append) < self.max_concept):
                this_append.append(0)
            ques_concept_relation_list.append(this_append)
        self.ques_concept_relation = torch.tensor(ques_concept_relation_list).to(args.device)

        with open(args.data_dir + 'next_question_set.pkl', 'rb') as fp:
            next_question_set = pickle.load(fp)
        self.next_question_set = torch.tensor(next_question_set)

    def gen(self, this_batch_size, total_num):
        batch_num = int(total_num / (this_batch_size * ((self.max_len + self.min_len) / 2)))
        if batch_num == 0:
            batch_num = 1

        all_batches = []
        for i in range(batch_num):

            this_len = np.random.randint(self.min_len, self.max_len)
            this_batch = [torch.tensor([this_len] * this_batch_size).to(self.device), []]
            ques_id = torch.randint(1, self.ques_num, (this_batch_size,)).to(self.device)
            for i in range(0, this_len):
                related_concepts = self.ques_concept_relation[ques_id].to(self.device)
                response = torch.randint(0, 2, (this_batch_size,)).to(self.device).float()
                interval_time = torch.zeros(this_batch_size).to(self.device).long()
                concept_interval_time = torch.zeros(this_batch_size).to(self.device).long()
                elapsed_time = torch.zeros(this_batch_size).to(self.device).long()

                this_step = [ques_id, related_concepts, interval_time, concept_interval_time, elapsed_time, response.unsqueeze(-1)]
                this_batch[1].append(this_step)
                
                '''the questions in next step'''
                next_index_in_nextset = torch.randint(1, self.ques_num, (this_batch_size,)).to(self.device)
                ques_id = self.next_question_set[ques_id, next_index_in_nextset]
                ques_id = ques_id.to(self.device)

            for j in range(this_len, self.seq_len):
                ques_id = torch.zeros(this_batch_size).to(self.device).long()
                related_concepts = torch.zeros(this_batch_size, self.max_concept).to(self.device).long()
                response = torch.zeros(this_batch_size).to(self.device).float()
                interval_time = torch.zeros(this_batch_size).to(self.device).long()
                concept_interval_time = torch.zeros(this_batch_size).to(self.device).long()
                elapsed_time = torch.zeros(this_batch_size).to(self.device).long()

                this_step = [ques_id, related_concepts, interval_time, concept_interval_time, elapsed_time, response.unsqueeze(-1)]
                this_batch[1].append(this_step)
            all_batches.append(this_batch)
        return all_batches

class discriminator(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.attention_heads = args.attention_heads
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.seq_len = args.seq_len
        self.mask_prob = args.mask_prob
        self.replace_prob = args.replace_prob
        self.crop_ratio = args.crop_ratio
        self.perm_ratio = args.perm_ratio
        self.contrast_num = args.contrast_num
        self.ques_predictor = modules.funcs(args.n_layer, args.dim * 2, args.problem_number + 1, args.dropout)
        self.resp_predictor = modules.funcs(args.n_layer, args.dim * 2, 1, args.dropout) 
        self.predictor = modules.funcs(args.n_layer, args.dim * 3, 1, args.dropout) #args.dim +
        self.pos_emb = nn.Parameter(torch.randn(args.seq_len,  args.dim).to(args.device), requires_grad=True)

        self.ex_map = modules.funcs(0, args.dim * 2, args.dim, args.dropout) 
        self.response_emb = nn.Parameter(torch.randn((args.problem_number + 1) * 2,  args.dim).to(args.device), requires_grad=True)
        self.prob_emb = nn.Parameter(torch.randn((args.problem_number + 1), args.dim).to(args.device), requires_grad=True)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num + 1, args.dim).to(args.device), requires_grad=True)

        self.ques_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim *2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.ques_k = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.ques_v = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.ques_o = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)

        self.resp_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.resp_k = nn.ParameterList([nn.Parameter(torch.randn(args.dim, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.resp_v = nn.ParameterList([nn.Parameter(torch.randn(args.dim, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.resp_o = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)

        self.final_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.final_k = nn.ParameterList([nn.Parameter(torch.randn(args.dim, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.final_v = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 3, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
        self.final_o = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)

        self.ques_ffn = modules.funcs(1, args.dim, args.dim, args.dropout)
        self.resp_ffn = modules.funcs(1, args.dim, args.dim, args.dropout)

        self.final_ffn = modules.funcs(1, args.dim, args.dim, args.dropout)
        showi0 = []
        for i in range(0, 50000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.norm= nn.LayerNorm(args.dim)
        self.x_list = []
        self.y_list = []
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)
        self.cos_simi = nn.CosineSimilarity(dim = 2, eps=1e-6)
        self.ques_num = args.problem_number
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce=False)
        self.bce_loss = torch.nn.BCELoss(reduce=False)
        self.short_gen(args.data_dir)

        ques_concept_relation = []
        with open(args.data_dir + 'problem_skills_relation.pkl', 'rb') as fp:
            raw_realtion = pickle.load(fp)
        raw_realtion[0] = [0] * self.max_concept
        for i in range(0, self.ques_num):
            related_concept = raw_realtion[i]
            while len(related_concept) < self.max_concept:
                related_concept.append(0)
            ques_concept_relation.append(related_concept)
        self.ques_concept_relation = torch.tensor(ques_concept_relation).to(self.device)

        with open(args.data_dir + 'cl4kt_permute_correct_candidate.pkl', 'rb') as fp:
            raw_candidate = pickle.load(fp)
        self.ques_correct_candidate = torch.tensor(raw_candidate)

        with open(args.data_dir + 'cl4kt_permute_wrong_candidate.pkl', 'rb') as fp:
            raw_candidate = pickle.load(fp)
        self.ques_wrong_candidate = torch.tensor(raw_candidate)
        
    def get_ques_representation(self, prob_ids, related_concept_index):

        filter0 = torch.where(related_concept_index == 0, self.ones, self.zeros).float()
        data_len = prob_ids.size()[0]
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0).unsqueeze(0).repeat(data_len, 1, 1)
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        related_concepts = concepts_cat[r_index, related_concept_index,:]
        filter_sum = torch.sum(filter0, dim = 1)

        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.node_dim)
        
        concept_level_rep = torch.sum(related_concepts, dim = 1) / div
        
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = 1)
        return v

    def compute_attention(self,  key, query, value):
        relevance_score = torch.matmul(
                                        query,
                                        key.transpose(2, 1)
                                        ) / math.sqrt(self.node_dim)
        soft_score = F.softmax(relevance_score, dim = -1)
        attened_representation = torch.matmul(soft_score, value)
        return attened_representation
    
    def multi_head_attention(self, k_resp, q_resp, v_resp, k_func, q_func, v_func,
                            o_func, ffn):
        representation_list = []
        for i in range(0, self.attention_heads):
            this_representation = self.compute_attention(
                            torch.matmul(k_resp, k_func[i]),
                            torch.matmul(q_resp, q_func[i]), 
                            torch.matmul(v_resp, v_func[i]))

            representation_list.append(this_representation)
        representation_raw = torch.matmul(torch.stack(representation_list, dim = 3), 
                                        o_func).squeeze(-1)

        representation_raw = ffn(representation_raw) + representation_raw
        representation = self.norm(representation_raw)
        return representation

    def core(self, all_ques_representation, all_response):

        all_response_representation = self.response_emb[all_response.long()]
        all_ques_resp = torch.cat([all_ques_representation, all_response_representation], dim = -1)
        ques_self_attention_raw = self.multi_head_attention(all_ques_representation,
                                                        all_ques_representation,
                                                        all_ques_representation,
                                                        self.ques_k, self.ques_q,
                                                        self.ques_v, self.ques_o,
                                                        self.ques_ffn)

        resp_self_attention = self.multi_head_attention(all_response_representation, 
                                                        all_response_representation,
                                                        all_response_representation,
                                                        self.resp_k, self.resp_q,
                                                        self.resp_v, self.resp_o,
                                                        self.resp_ffn)

        decoder_output_org = self.multi_head_attention(ques_self_attention_raw, 
                                                resp_self_attention,
                                                all_ques_resp,
                                                self.final_k, self.final_q,
                                                self.final_v, self.final_o, 
                                                self.final_ffn)

        data_len = all_response.size()[0]
        cat_zero = torch.zeros(data_len, self.node_dim).unsqueeze(1).to(self.device)

        ques_predict_x = torch.cat([cat_zero,
                                    ques_self_attention_raw.split([1, self.seq_len - 1], dim = -2)[1]], dim = -2)
        response_predict_x = torch.cat([cat_zero, 
                                    resp_self_attention.split([1, self.seq_len - 1], dim = -2)[1]], dim = -2)                            
        
        predict_questions = self.ques_predictor(torch.cat([ques_predict_x, resp_self_attention], dim = -1))
        predict_response = self.resp_predictor(torch.cat([ques_self_attention_raw, response_predict_x], dim = -1))

        predict_x = torch.cat([decoder_output_org, all_ques_representation], dim = -1)
        logits = self.predictor(predict_x)

        return self.sigmoid(logits), self.sigmoid(predict_questions), \
            self.sigmoid(predict_response).squeeze(-1), predict_x
    
    def integrate_ques_resp(self, inputs):
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = inputs
        this_ques = self.get_ques_representation(prob_ids, related_concept_index)
        this_resp = prob_ids * 2 + operate.squeeze(-1)
        return this_ques, this_resp

    def obtain_score(self, x, change_label = 0):
        data_len = len(x[0])
        seq_num = x[0]
        predict_list = []
        ques_list, resp_list = [],  []
        prob_id_list, operate_list = [], []
        
        for i in range(0, self.seq_len):
            this_ques, this_resp = self.integrate_ques_resp(x[1][i])
            ques_list.append(this_ques)
            resp_list.append(this_resp)

            prob_id_list.append(x[1][i][0])
            operate_list.append(x[1][i][-1].squeeze(-1))



        all_ques = torch.stack(ques_list, dim = 1)
        all_resp = torch.stack(resp_list, dim = 1)

        prob_id_tensor = torch.stack(prob_id_list, dim = 1)
        operate_tensor = torch.stack(operate_list, dim = 1)


        score, predict_ques, predict_resp, score_vec  = self.core(all_ques, all_resp)
        
        final_score, final_score_vec, ques_loss_list, resp_loss_list = [], [], [], []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_score = score[i][0: this_seq_len ]
            final_score.append(this_score)

            this_ques = prob_id_tensor[i][0: this_seq_len]
            this_predict_ques = predict_ques[i][0: this_seq_len]
            this_ques_loss = self.ce_loss(this_predict_ques, this_ques)
            ques_loss_list.append(this_ques_loss)

            this_resp = operate_tensor[i][0: this_seq_len]
            this_predict_resp = predict_resp[i][0: this_seq_len]
            this_resp_loss = self.bce_loss(this_predict_resp, this_resp.float())
            resp_loss_list.append(this_resp_loss)
            # resp_loss += this_resp_loss

            this_vec = score_vec[i][0: this_seq_len]
            final_score_vec.append(this_vec)

        return torch.cat(final_score, dim = 0), \
                torch.mean(torch.cat(ques_loss_list, dim = 0)), \
                torch.mean(torch.cat(resp_loss_list, dim = 0)), \
                torch.cat(final_score_vec, dim = 0)
    
    def obtain_reward(self, ques_cons_list, all_resp):
        ques_resp_list = []
        for step_info in ques_cons_list:
            prob_ids, related_concept_index = step_info
            this_ques = self.get_ques_representation(prob_ids, related_concept_index)
            ques_resp_list.append(this_ques)
        all_ques = torch.stack(ques_resp_list, dim = 1)
        score, _, _, state = self.core(all_ques, all_resp)
        return score.squeeze(-1), state 

    def core_each(self, ques_representation, response, x_tensor, y_tensor):
        ques_resp = torch.cat([ques_representation, response], dim = -1)
        ques_self_attention_raw = self.multi_head_attention(
                                                        x_tensor,
                                                        ques_representation.unsqueeze(1),
                                                        x_tensor,
                                                        self.ques_k, self.ques_q,
                                                        self.ques_v, self.ques_o,
                                                        self.ques_ffn)

        resp_self_attention = self.multi_head_attention(
                                                        y_tensor,
                                                        response.unsqueeze(1), 
                                                        y_tensor,
                                                        self.resp_k, self.resp_q,
                                                        self.resp_v, self.resp_o,
                                                        self.resp_ffn)

        decoder_output = self.multi_head_attention(ques_self_attention_raw, 
                                                resp_self_attention,
                                                ques_resp.unsqueeze(1),
                                                self.final_k, self.final_q,
                                                self.final_v, self.final_o, 
                                                self.final_ffn)
        # logits = self.predictor(decoder_output)
        predict_vec = torch.cat([decoder_output.squeeze(1), ques_representation], dim = -1)
        logits = self.predictor(predict_vec)

        return self.sigmoid(logits), predict_vec

    def short_gen(self, path):
        with open(path + 'problem_skills_relation.pkl', 'rb') as fp:
            ques_con_relation = pickle.load(fp)
        '''scan the list, compasate 0'''
        max_con_len = 0
        for k in ques_con_relation.keys():
            if len(ques_con_relation[k]) > max_con_len:
                max_con_len = len(ques_con_relation[k])
        for k in ques_con_relation.keys():
            while len(ques_con_relation[k]) < max_con_len:
                ques_con_relation[k].append(0)
        '''scan end'''
        batch_data = []
        batch_size = 1024
        batch_num = int(self.ques_num / batch_size) + 1
        self.batches = []
        for i in range(0, batch_num):
            start = max(i * batch_size, 1)
            end = min(batch_size * (i+1), self.ques_num)
            this_ques, this_concepts= [], [] #[ques, concepts], [resp]
            for ques_id in range(start, end):
                related_concepts = ques_con_relation[ques_id]
                this_ques.append(ques_id)
                this_concepts.append(related_concepts)
            self.batches.append([torch.tensor(this_ques).to(self.device), \
                                torch.tensor(this_concepts).to(self.device)])

    def obtain_score_short(self):
        scores_list = []
        
        for ques_con in self.batches:
            ques, cons = ques_con
            ques_resp = self.get_ques_representation(ques, cons)
            data_len = ques.size()[0]
            x_tensor = torch.zeros(data_len, 1, 2 * self.node_dim).to(self.device)
            y_tensor = torch.zeros(data_len, 1, self.node_dim).to(self.device)
            for i in range(0, 2):
                response = self.response_emb[i + ques * 2]
                this_score = self.core_each(ques_resp, response, x_tensor, y_tensor)[0]
                scores_list.append(this_score.squeeze(-1))
       
        return torch.cat(scores_list, dim = 0)

    def obtain_score_reverse(self, x):
        data_len = len(x[0])
        seq_num = x[0]
        score_list, vec_list, x_list, y_list = [], [], \
            [torch.zeros(data_len, self.node_dim * 2).to(self.device)], \
            [torch.zeros(data_len, self.node_dim).to(self.device)]
        
        for i in range(0, self.seq_len):
            prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = x[1][i]
            ques_resp = self.get_ques_representation(prob_ids, related_concept_index)
            reverse_response = self.response_emb[(1 - operate.long().squeeze(-1)) + prob_ids * 2]
            x_tensor = torch.stack(x_list, dim = 1)
            y_tensor = torch.stack(y_list, dim = 1)

            score, score_vec = self.core_each(ques_resp, reverse_response, x_tensor, y_tensor)
            x_list.append(ques_resp)
            real_response = self.response_emb[operate.long().squeeze(-1) + prob_ids * 2]
            y_list.append(real_response)
            score_list.append(score)
            vec_list.append(score_vec)

        score = torch.stack(score_list, dim = 1).squeeze(-1)
        vec = torch.stack(vec_list, dim = 1)
           
        final_score, final_vec = [], []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_score = score[i][0: this_seq_len]
            final_score.append(this_score)

            this_vec = vec[i][0: this_seq_len]
            final_vec.append(this_vec)
        return torch.cat(final_score, dim = 0), \
                None, None, torch.cat(final_vec, dim = 0)

    def data_aug(self, inputs): 
        '''positive sample gen: crop, mask, permute, replace'''
        
        data_len = len(inputs[0])
        masked_data, crop_data, \
            permute_data, replace_data = [inputs[0], []], [inputs[0], []],\
                                        [inputs[0], []], [inputs[0], []]
        '''here is mask, and replace: args.mask_prob, args.replace_prob'''
        for i in range(0, self.seq_len):
            prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate_org = inputs[1][i]
            mask_rand = torch.tensor(np.random.random(data_len)).to(self.device)
            mask_ques_id = torch.where(mask_rand < self.mask_prob, self.ques_num, prob_ids)
            masked_concepts =  torch.zeros_like(related_concept_index) + self.concept_num
            mask_concept_id_list = []
            for j in range(0, data_len):
                if mask_rand[j] < self.mask_prob:
                    mask_concept_id_list.append(masked_concepts[j])
                else:
                    mask_concept_id_list.append(related_concept_index[j])
            mask_concept_id = torch.stack(mask_concept_id_list, dim = 0)
            masked_data[1].append([mask_ques_id, mask_concept_id, interval_time, concept_interval_time, elapsed_time, operate_org])

            replace_rand = torch.tensor(np.random.random(data_len)).to(self.device)
            replace_correct_cadidate = self.ques_correct_candidate[prob_ids, torch.randint(0, self.ques_num, (data_len,))] #TODO
            replace_correct_cadidate = replace_correct_cadidate.to(self.device)
            replace_wrong_candidate = self.ques_wrong_candidate[prob_ids, torch.randint(0, self.ques_num, (data_len,))]
            replace_wrong_candidate = replace_wrong_candidate.to(self.device)
            replace_wrong_correct_cat = torch.stack([replace_wrong_candidate, replace_correct_cadidate], dim = -1)
            used_to_replace_quesid = replace_wrong_correct_cat[self.show_index[0: data_len], operate_org.long().squeeze(-1)]
            used_to_replaced_concepts =  self.ques_concept_relation[used_to_replace_quesid] #TODO
            replace_ques_id = torch.where(replace_rand < self.replace_prob, used_to_replace_quesid, prob_ids)
            replace_concept_id_list = []
            for j in range(0, data_len):
                if replace_rand[j] < self.replace_prob:
                    replace_concept_id_list.append(used_to_replaced_concepts[j])
                else:
                    replace_concept_id_list.append(related_concept_index[j])
            replaced_concepts = torch.stack(replace_concept_id_list, dim = 0)

            # replaced_concepts = torch.where(replace_rand < self.replace_prob, used_to_replaced_concepts, related_concept_index)
            replace_data[1].append([replace_ques_id, replaced_concepts, interval_time, concept_interval_time, elapsed_time, operate_org])

        '''here is the crop'''
        s_list = [np.random.randint(0, inputs[0][i].cpu().numpy() - 1) for i in range(0, data_len)]
        start_pos = torch.tensor(s_list).to(self.device).long()
        crop_length = (inputs[0] * self.crop_ratio).long()#).to(self.device).long()
        
        end_pos = start_pos + crop_length
        end_pos = torch.where(end_pos > inputs[0], inputs[0], end_pos)
        crop_data[0] = end_pos - start_pos
        # end_pos = torch.where(end_pos >= self.seq_length, self.seq_length, end_pos)
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate_org = inputs[1][0]
        fill_data = [torch.zeros_like(prob_ids[0]).long(), torch.zeros_like(related_concept_index[0]).long(), \
                        torch.zeros_like(interval_time[0]), torch.zeros_like(concept_interval_time[0]).long(), \
                        torch.zeros_like(elapsed_time[0]).long(), torch.zeros_like(operate_org[0]).long()]
        attributes_num = len(inputs[1][0])
        for i in range(0, self.seq_len):
            this_step_data = []
            for j in range(0, attributes_num):
                this_attribute_list = []
                for k in range(0, data_len):
                    if i + start_pos[k] < end_pos[k]:
                    # if i >= start_pos[k] and i < end_pos[k]:
                        this_attribute_list.append(inputs[1][i + start_pos[k]][j][k])
                    else:
                        this_attribute_list.append(fill_data[j])
                this_attribute = torch.stack(this_attribute_list, dim = 0)
                this_step_data.append(this_attribute)
            crop_data[1].append(this_step_data)
        
        '''here is the permute'''
        
        shuffled_index_number, start_pos, end_pos = [], [], []
        for i in range(0, data_len):
            this_start_pos = np.random.randint(0, inputs[0][i].cpu().numpy() - 1)
            this_end_pos = int(inputs[0][i] * self.perm_ratio) + this_start_pos 
            this_end_pos = this_end_pos if this_end_pos < inputs[0][i] else inputs[0][i] - 1
            start_pos.append(this_start_pos)
            end_pos.append(this_end_pos)

            this_index = [j for j in range(this_start_pos, this_end_pos)]
            np.random.shuffle(this_index)
            shuffled_index_number.append(this_index)

        for i in range(0, self.seq_len):
            this_step_data = []
            for j in range(0, attributes_num):
                attribute_list = []
                for k in range(0, data_len):
                    if start_pos[k] <= i and end_pos[k] > i:
                        this_shuffled_index = shuffled_index_number[k][i - start_pos[k]]
                        this_info = inputs[1][this_shuffled_index][j][k]
                        attribute_list.append(this_info)
                    else:
                        attribute_list.append(inputs[1][i][j][k])
                this_attribute = torch.stack(attribute_list, dim = 0)
                this_step_data.append(this_attribute)
            permute_data[1].append(this_step_data)

        return masked_data, crop_data, permute_data, replace_data

class generator(nn.Module): 
    def __init__(self, args):
        super().__init__()
        
        self.node_dim = args.dim
        self.n_layer = args.n_layer
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.seq_len = args.seq_len
        self.gamma = args.gamma
        self.lamb = args.lamb 
        self.multi_len = args.multi_len
        self.action_predictor = modules.funcs(self.n_layer, self.node_dim * 3, 1, args.dropout)
        self.ques_predictor = modules.funcs(self.n_layer, self.node_dim * 2, args.problem_number, args.dropout)
        self.v_map = modules.funcs(self.n_layer, self.node_dim * 2, self.node_dim, args.dropout)
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, self.node_dim).to(args.device), requires_grad=True)
        self.gru_h = modules.mygru(0, self.node_dim * 4, self.node_dim)
        self.value_est = modules.funcs(self.n_layer, args.dim * 3, 1, args.dropout)
        showi0 = []
        for i in range(0, 1000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, self.node_dim).to(args.device), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce = False)
        self.cos_simi = nn.CosineSimilarity(dim = 2, eps=1e-6)

    def get_ques_representation(self, prob_ids, related_concept_index):

        filter0 = torch.where(related_concept_index == 0, self.ones, self.zeros).float()
        data_len = prob_ids.size()[0]
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0).unsqueeze(0).repeat(data_len, 1, 1)
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        related_concepts = concepts_cat[r_index, related_concept_index,:]
        filter_sum = torch.sum(filter0, dim = 1)

        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.node_dim)
        
        concept_level_rep = torch.sum(related_concepts, dim = 1) / div
        
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = 1)
        return self.v_map(v)
        
    def cell(self, inputs, h, stack_h, stack_ques, this_pos, take = 0): 
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = inputs
        ques_resp = self.get_ques_representation(prob_ids, related_concept_index)

        previous_ques_cat = stack_ques
        my_ques_cat = ques_resp

        weighted_h = torch.matmul(F.softmax(torch.matmul(my_ques_cat.unsqueeze(1), 
                                                        previous_ques_cat.transpose(2, 1)
                                            ) / math.sqrt(2 * self.node_dim), 
                                            dim  = -1), # attention score
                                    stack_h).squeeze(1)


        
        logits = self.action_predictor(
                                        torch.cat([ques_resp, h, weighted_h], dim = -1))
        prob = self.sigmoid(logits)
        prob_s = torch.cat([1 - prob, prob],dim = 1)
        out_operate_logits = Categorical(prob_s).sample().unsqueeze(-1)
        out_thres = torch.where(prob > 0.5, self.ones, self.zeros)
        take_label = [operate, out_operate_logits, out_thres][take]

        ques_pre_in = torch.cat([h.mul(take_label.float()), 
                                 h.mul(1 - take_label.float())], dim = -1)
        ques_probs = self.ques_predictor(ques_pre_in)
        this_ce_loss = self.ce_loss(self.sigmoid(ques_probs), prob_ids)

        # vec_ques_action_state = torch.cat([ques_resp, op_emb, h], dim = -1)
        vec_ques_action_state = None
        weight_ques_cat = torch.cat([weighted_h, ques_resp], dim = -1)
        update_in = torch.cat([weight_ques_cat.mul(take_label.float()), 
                                weight_ques_cat.mul(1 - take_label.float())], 
                                dim = -1)
        h = self.gru_h(update_in, h)

        return h, logits, take_label, ques_resp, this_ce_loss, vec_ques_action_state, [prob_ids, related_concept_index]

    def forward_gen(self, x, discriminator, take_index = 1):  
        data_len = len(x[0])
        seq_num = x[0]
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        p_action_list, pre_state_list, emb_action_list, predict_list, label_prob_list = [], [], [], [], []

        h_list, ques_list, info_disc = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)], []
        all_response, all_ques_cons = [], []
        for seqi in range(0, self.seq_len):
            
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, out_operate, ques_resp, ques_ce_loss, vec_qas, ques_cons_id = self.cell(x[1][seqi], h, stack_h, stack_ques, seqi, take = take_index)
            h_list.append(h)
            ques_list.append(ques_resp)

            label_prob_all = torch.cat([1 - self.sigmoid(logits), self.sigmoid(logits)], dim = -1)
            label_prob_list.append(label_prob_all.gather(1, out_operate.long()).squeeze(-1))
            predict_list.append(logits.squeeze(1))
            
            all_response.append(self.sigmoid(logits).squeeze(-1))
            all_ques_cons.append(ques_cons_id)

        reward_tensor, states = discriminator.obtain_reward(all_ques_cons,
                                                torch.stack(all_response, dim = 1))
        reward_tensor = reward_tensor.detach()
        states = states.detach()
        label_prob_tensor = torch.stack(label_prob_list, dim = 1)
        logits_tensor = torch.stack(predict_list, dim = 1)
        loss, tracat_logits, emb_list = [], [], []
        reward_list = []
        
        for i in range(0, data_len):
            
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
            
            after_state = states[i][0: this_seq_len]
            before_state = torch.cat([torch.zeros_like(after_state[0].unsqueeze(0)).to(self.device),
                                    after_state[0: -1]], dim = 0)
            this_reward =  -torch.log(1 - reward_tensor[i][0: this_seq_len] + 1e-10)
            td_target = this_reward + 0.98 * self.value_est(after_state).squeeze(-1)
            delta = td_target - self.value_est(before_state).squeeze(-1)
            delta = delta.detach().cpu().numpy()
            
            reward_list.append(this_reward_list[0: this_seq_len])
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
            
            '''predicted loss'''
            this_label_prob = label_prob_tensor[i][0: this_seq_len]
           
            smooth_loss = F.smooth_l1_loss(self.value_est(before_state).squeeze(-1) , td_target.detach(), reduce = False)
            predict_loss = -torch.log(this_label_prob) * advantage.squeeze(-1) + smooth_loss
            loss.append(torch.sum(predict_loss))

            this_prob = logits_tensor[i][0: this_seq_len]
            tracat_logits.append(this_prob)

        label_len = torch.cat(tracat_logits, dim = 0).size()[0]
        loss_l = sum(loss)
        loss = self.lamb * (loss_l / label_len) 
        return torch.cat(tracat_logits, dim = 0), loss, torch.cat(reward_list, dim = 0)

    def supervised(self, x, take_index = 1):  
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        predict_list, label_prob_list, reward_list = [], [], []

        consec_right = torch.zeros(data_len).to(self.device)
        h_list, ques_list = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)]
        ques_ce_loss_list = [] 
        for seqi in range(0, self.seq_len):
            
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, out_operate, ques_resp, ques_ce_loss, vec_qas, ques_cons_id  = self.cell(x[1][seqi], h, stack_h, stack_ques, seqi, take = take_index)
            h_list.append(h)
            ques_list.append(ques_resp)

            label_prob_all = torch.cat([1 - self.sigmoid(logits), self.sigmoid(logits)], dim = -1)
            label_prob_list.append(label_prob_all.gather(1, out_operate.long()).squeeze(-1))
            predict_list.append(logits.squeeze(1))
            
            ground_truth = x[1][seqi][-1]
            original_reward = torch.where(ground_truth.squeeze(-1).float() == out_operate.squeeze(-1),
                            torch.tensor(1.0).to(self.device), 
                            torch.tensor(0.0).to(self.device)) #original reward
            consec_right = consec_right.mul(original_reward) + original_reward
            this_reward = consec_right
            reward_list.append(this_reward)
            ques_ce_loss_list.append(ques_ce_loss)

            

        reward_tensor = torch.stack(reward_list, dim = 1).float() 
        seq_num = x[0]
        label_prob_tensor = torch.stack(label_prob_list, dim = 1)
        logits_tensor = torch.stack(predict_list, dim = 1)
        ques_ce_loss_tensor = torch.stack(ques_ce_loss_list, dim = 1)
        loss, tracat_logits, ques_ce_loss_list, qas = [], [], [], []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
        
            td_target = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta = td_target.detach().cpu().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
            
            '''predicted loss'''
            this_label_prob = label_prob_tensor[i][0: this_seq_len]
            predict_loss = -torch.log(this_label_prob) * advantage.squeeze(-1)
            loss.append(torch.sum(predict_loss))

            this_prob = logits_tensor[i][0: this_seq_len]
            tracat_logits.append(this_prob)

            this_ques_ce_loss = ques_ce_loss_tensor[i][0: this_seq_len]
            ques_ce_loss_list.append(this_ques_ce_loss)

        label_len = torch.cat(tracat_logits, dim = 0).size()[0]
        loss_l = sum(loss)
        loss = self.lamb * (loss_l / label_len) 
        ques_final_ce_loss = torch.mean(torch.cat(ques_ce_loss_list, dim = 0))
        return torch.cat(tracat_logits, dim = 0), loss, ques_final_ce_loss, None

    def gen_seq_new(self, x, discriminator, take_index = 1):  
        new_data = [x[0], []]
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        h_list, ques_list = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)] 
        for seqi in range(0, self.seq_len):
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, out_operate, ques_resp, ques_ce_loss, vec_qas, ques_cons_id = self.cell_d(x[1][seqi], h, stack_h, stack_ques, seqi, discriminator,take = take_index)
            h_list.append(h.detach())
            ques_list.append(ques_resp.detach())
            out_operate = out_operate.detach()

            prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, _ = x[1][seqi]
            this_labeled_data = [prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, out_operate]
            new_data[1].append(this_labeled_data)

        return new_data

    def gen_seq(self, x, take_index = 1):  
        # useless now
        new_data = [x[0], []]
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        h_list, ques_list = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)] 
        for seqi in range(0, self.seq_len):
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, out_operate, ques_resp, ques_ce_loss, vec_qas, ques_cons_id = self.cell(x[1][seqi], h, stack_h, stack_ques, seqi, take = take_index)
            h_list.append(h.detach())
            ques_list.append(ques_resp.detach())
            out_operate = out_operate.detach()

            prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, _ = x[1][seqi]
            this_labeled_data = [prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, out_operate]
            new_data[1].append(this_labeled_data)

        return new_data

    def cell_d(self, inputs, h, stack_h, stack_ques, this_pos, discriminator, take = 0): 
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = inputs
        data_len = prob_ids.size()[0]
        ques_resp = self.get_ques_representation(prob_ids, related_concept_index)

        previous_ques_cat = stack_ques
        my_ques_cat = ques_resp

        weighted_h = torch.matmul(F.softmax(torch.matmul(my_ques_cat.unsqueeze(1), 
                                                        previous_ques_cat.transpose(2, 1)
                                            ) / math.sqrt(2 * self.node_dim), 
                                            dim  = -1), # attention score
                                    stack_h).squeeze(1)


        
        logits = self.action_predictor(
                                        torch.cat([ques_resp, h, weighted_h], dim = -1))
        prob = self.sigmoid(logits)
        '''random prob id and related_concept_index'''
        take_label_1 = torch.where(prob > 0.5, self.ones, self.zeros).long()
        
        take_label = take_label_1

        ques_pre_in = torch.cat([h.mul(take_label.float()), 
                                 h.mul(1 - take_label.float())], dim = -1)
        ques_probs = self.ques_predictor(ques_pre_in)
        this_ce_loss = self.ce_loss(self.sigmoid(ques_probs), prob_ids)

        vec_ques_action_state = None

        weight_ques_cat = torch.cat([weighted_h, ques_resp], dim = -1)
        update_in = torch.cat([weight_ques_cat.mul(take_label.float()), 
                                weight_ques_cat.mul(1 - take_label.float())], 
                                dim = -1)
        h = self.gru_h(update_in, h)

        return h, logits, take_label, ques_resp, this_ce_loss, vec_ques_action_state, [prob_ids, related_concept_index]

    def f_gen(self, x, discriminator,  take_index = 0, mod_len = 300):
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        # predict_x_list = []
        predict_list, label_list = [], []

        h_list, ques_list = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)]
        discriminator.reset_xy(data_len)
        all_response, all_ques_cons = [], []

        for seqi in range(0, self.seq_len):
            
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, labels, ques_resp, ques_ce_loss, vec_qas, ques_cons_id  = self.cell_d(x[1][seqi], h, stack_h, stack_ques, seqi, discriminator, take = take_index)
            h_list.append(h)
            ques_list.append(ques_resp)
            
            predict_list.append(logits.squeeze(1))
            label_list.append(labels.squeeze(1))

            # all_response.append(ques_cons_id[0] * 2 + labels.squeeze(-1))
            all_response.append(labels.squeeze(-1))
            all_ques_cons.append(ques_cons_id)

        seq_num = x[0]
        reward_tensor = discriminator.obtain_reward(all_ques_cons,
                                                torch.stack(all_response, dim = 1)).detach()
        logits_tensor = torch.stack(predict_list, dim = 1)
        label_tensor = torch.stack(label_list, dim = 1)
        tracat_logits, tracat_labels_list, reward_list = [], [], []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_prob = logits_tensor[i][0: this_seq_len]
            tracat_logits.append(this_prob)

            this_labels = label_tensor[i][0: this_seq_len]
            tracat_labels_list.append(this_labels) 

            this_reward_list = reward_tensor[i]
            reward_list.append(this_reward_list[0: this_seq_len])
        return torch.cat(tracat_logits, dim = 0), torch.cat(tracat_labels_list, dim = 0), torch.cat(reward_list, dim = 0)

    def eval_core(self, x, discriminator,  take_index = 0, mod_len = 300):
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        # predict_x_list = []
        predict_list, label_list = [], []

        h_list, ques_list = [h], [torch.zeros(data_len, self.node_dim ).to(self.device)]

        for seqi in range(0, self.seq_len):
            
            stack_h = torch.stack(h_list, dim = 1)
            stack_ques = torch.stack(ques_list, dim = 1)
            h, logits, labels, ques_resp, ques_ce_loss, vec_qas, ques_cons_id  = self.cell_d(x[1][seqi], h, stack_h, stack_ques, seqi, discriminator, take = take_index)
            h_list.append(h.detach())
            ques_list.append(ques_resp.detach())
            
            predict_list.append(logits.detach().squeeze(1))
            label_list.append(labels.squeeze(1))

            if seqi % mod_len == 0 and seqi > 0:
                h_mul = 0.0
            else:
                h_mul = 1.0
            h = h.mul(h_mul)

        seq_num = x[0]
        logits_tensor = torch.stack(predict_list, dim = 1)
        label_tensor = torch.stack(label_list, dim = 1)
        tracat_logits, tracat_labels_list = [], []
        # reward_list = []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_prob = logits_tensor[i][0: this_seq_len]
            tracat_logits.append(this_prob)

            this_labels = label_tensor[i][0: this_seq_len]
            tracat_labels_list.append(this_labels) 

            

        return torch.cat(tracat_logits, dim = 0), torch.cat(tracat_labels_list, dim = 0)#, torch.cat(predict_x_trac, dim = 0), 0

    def evaluate(self, inputs, discriminator):
        probs, labels = self.eval_core(inputs, discriminator, take_index = 1, mod_len = 300)
        return probs, labels

    def evaluate_cut(self, inputs, discriminator):
        probs, labels = self.eval_core(inputs, discriminator,  take_index = 1, mod_len = self.multi_len)
        return probs, labels
