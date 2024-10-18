from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from get_data import get_data_OPPORTUNITY
from get_data import get_data_realWorld
from get_data import get_data_SBHAR
from model_opp import FeatureExtracter
from model_opp import Discriminator
from model_opp import ActivityClassifier
from model_opp import ClassAwareWeightNetwork
from model_opp import ReverseLayerF
from tqdm import tqdm
from sklearn.cluster import KMeans
import higher

class Solver(object):
    def __init__(self, args):
        self.seed = args.seed
        self.N_steps = args.N_steps
        self.N_steps_eval = args.N_steps_eval
        self.N_eval = int(args.N_steps/args.N_steps_eval)
        self.test_user = args.test_user

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.confidence_rate = args.confidence_rate
        self.w_c_T = args.w_c_T

        self.dataset = args.dataset
        self.tag = args.dataset + '_SWL-Adapt_user' + str(args.test_user)

        if args.dataset == 'SBHAR':
            self.N_channels = args.N_channels_S
            self.N_classes = args.N_classes_S
        if args.dataset == 'OPPORTUNITY':
            self.N_channels = args.N_channels_O
            self.N_classes = args.N_classes_O
        if args.dataset == 'realWorld':
            self.N_channels = args.N_channels_R
            self.N_classes = args.N_classes_R
        
        if args.dataset == 'SBHAR':
            self.train_loader_S, self.train_loader_T, self.test_loader = get_data_SBHAR(self.batch_size, args.test_user, args)
        elif args.dataset == 'OPPORTUNITY':
            self.train_loader_S, self.train_loader_T, self.test_loader = get_data_OPPORTUNITY(self.batch_size, args.test_user, args)
        elif args.dataset == 'realWorld':
            self.train_loader_S, self.train_loader_T, self.test_loader = get_data_realWorld(self.batch_size, args.test_user, args)

        self.FE = FeatureExtracter(self.N_channels)
        self.D = Discriminator()
        self.AC = ActivityClassifier(self.N_classes)
        self.CS = ClassAwareWeightNetwork(1, 100, 100, 1, 3)
        
        self.FE.cuda()
        self.D.cuda()
        self.AC.cuda()
        self.CS.cuda()
        
        self.opt_fe = optim.Adam(self.FE.parameters(), lr=self.lr)
        self.opt_d = optim.Adam(self.D.parameters(), lr=self.lr)
        self.opt_ac = optim.Adam(self.AC.parameters(), lr=self.lr)
        self.opt_cs = optim.Adam(self.CS.parameters(), lr=args.WA_lr)

        self.scheduler_fe = optim.lr_scheduler.CosineAnnealingLR(self.opt_fe, self.N_eval)
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(self.opt_d, self.N_eval)
        self.scheduler_ac = optim.lr_scheduler.CosineAnnealingLR(self.opt_ac, self.N_eval)
        self.scheduler_cs = optim.lr_scheduler.CosineAnnealingLR(self.opt_cs, self.N_eval)

    def reset_grad(self):
        self.opt_fe.zero_grad()
        self.opt_d.zero_grad()
        self.opt_ac.zero_grad()
        self.opt_cs.zero_grad()

    def forward_pass(self, inputs, out_type=None):
        fused_feature = self.FE(inputs)
        disc = None
        activity_clsf = None
        if out_type != 'C':
            reverse_feature = ReverseLayerF.apply(fused_feature, 1)
            disc = self.D(reverse_feature)
        if out_type != 'D':
            activity_clsf = self.AC(fused_feature)
        return disc, activity_clsf

    def train(self):
        print('\n>>> Start Training ...')
        test_acc, test_f1 = 0, 0
        
        criterion_c = nn.CrossEntropyLoss(reduction='none').cuda()
        criterion_d = nn.BCEWithLogitsLoss(reduction='none').cuda()

        train_c_acc_S = torchmetrics.Accuracy(task='multiclass', num_classes=self.N_classes).cuda()  
        train_c_f1_S = torchmetrics.F1Score(task='multiclass', num_classes=self.N_classes, average='macro').cuda()
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        step = 0
        self.train_loader_S_iter = iter(self.train_loader_S)

        a = [0 for _ in range(self.N_classes)]  
        for batch in tqdm(self.train_loader_S, desc="Counting samples"):
            labels = batch[1]
            labels = torch.tensor(labels)
            for label in labels.view(-1):  
                label_idx = label.item()
                if 0 <= label_idx < self.N_classes:
                    a[label_idx] += 1  
                else:
                    print(f"Invalid label index: {label_idx}")
        
   
        a = [[count] for count in a] 
        es = KMeans(3)
        es.fit(a)
        c = es.labels_
        #print('c:', c.tolist())
        
        # w = [[],[],[]]
        # for i in range(3):
        #     for k, j in enumerate(c):
        #         if i == j:
        #             w[i].append(a[k][0])
        #print(w)

        self.train_loader_T_iter = iter(self.train_loader_T)
        for n_eval in range(self.N_eval):

            self.FE.train()
            self.D.train()
            self.AC.train()
            self.CS.train()

            Loss_c = 0
            Loss_d = 0
            for batch_idx in range(self.N_steps_eval):
                step += 1

                x_T, y_T = None, None
                try:
                    x_T, y_T = next(self.train_loader_T_iter)
                except StopIteration:
                    self.train_loader_T_iter = iter(self.train_loader_T)
                    x_T, y_T = next(self.train_loader_T_iter)

                x_S, y_S = None, None
                try:
                    x_S, y_S = next(self.train_loader_S_iter)
                except StopIteration:
                    self.train_loader_S_iter = iter(self.train_loader_S)
                    x_S, y_S = next(self.train_loader_S_iter)

                x_S = Variable(x_S.cuda())
                y_S = Variable(y_S.long().cuda())
                x_T = Variable(x_T.cuda())
                yd_S = torch.zeros(self.batch_size)
                yd_T = torch.ones(self.batch_size)                
                yd_S = Variable(yd_S.cuda())
                yd_T = Variable(yd_T.cuda())

                self.reset_grad()
                
                """ step 1: update feature extractor and classifier"""
                _, logits_ac_S = self.forward_pass(x_S, 'C')  
                loss_c_S = criterion_c(logits_ac_S, y_S)
                loss_c = torch.mean(loss_c_S)
                
                loss_c.backward()
                self.opt_fe.step()
                self.opt_ac.step()
                self.reset_grad()

                # track training losses and metrics
                Loss_c += loss_c.item()
                train_c_acc_S(logits_ac_S.softmax(dim=-1), y_S)
                train_c_f1_S(logits_ac_S.softmax(dim=-1), y_S)

                """ step 2: update class-aware weight network"""
                torch.cuda.empty_cache()  
                
                _, logits_ac_T = self.forward_pass(x_T)

                with torch.no_grad():
                    pseudo_y_T = logits_ac_T.max(1)[1]
                    certainty_y_T = logits_ac_T.softmax(dim=1).max(1)[0]
                    mask_C = (certainty_y_T > self.confidence_rate).float()
                if mask_C.sum() > 0:
                    loss_c_T = criterion_c(logits_ac_T, pseudo_y_T) * mask_C
                    sample_cost_T = torch.reshape(loss_c_T, (len(loss_c_T), 1)) 
                    v_lambda_T = self.CS(sample_cost_T, pseudo_y_T, c).squeeze(1)
                    meta_loss = torch.mean(v_lambda_T * loss_c_T)
                    meta_loss.backward()     
                    self.opt_cs.step()
                    self.reset_grad()
                else:
                    v_lambda_T = 0
                    loss_c_T = 0
                
                _, logits_ac_S = self.forward_pass(x_S, 'C')  
                loss_c_S = criterion_c(logits_ac_S, y_S)
                sample_cost_S = torch.reshape(loss_c_S, (len(loss_c_S), 1)) 
                v_lambda_S = self.CS(sample_cost_S, y_S, c).squeeze(1)
                cl_loss = torch.mean(v_lambda_S * loss_c_S)

                cl_loss.backward()
                self.opt_ac.step()
                self.reset_grad()

                """ step 3: update feature extractor and domain discriminator"""
                logits_d_S, logits_ac_S = self.forward_pass(x_S)
                loss_d_S = criterion_d(logits_d_S, yd_S)
                loss_d_S = torch.mean(loss_d_S)

                logits_d_T, logits_ac_T = self.forward_pass(x_T)
                loss_d_T = criterion_d(logits_d_T, yd_T)
                loss_d_T = torch.mean(loss_d_T)
               
                loss_d = loss_d_S + loss_d_T

                loss_d.backward()
                self.opt_fe.step()
                self.opt_d.step()
                self.reset_grad()            

                # track training losses and metrics after optimization
                Loss_d += loss_d.item()

            test_Loss_c, test_c_acc_T, test_c_f1_T = self.eval()

            print('Train Eval {}: Train: c_acc_S:{:.6f} c_f1_S:{:.6f} Loss_c:{:.6f} Loss_d:{:.6f}'.format(
                n_eval, train_c_acc_S.compute().item(), train_c_f1_S.compute().item(), Loss_c, Loss_d))
            print('               Test:  c_acc_T:{:.6f} c_f1_T:{:.6f} Loss_c_T:{:.6f}'.format(
                test_c_acc_T, test_c_f1_T, test_Loss_c))  

            if n_eval == self.N_eval-1:
                test_acc = test_c_acc_T
                test_f1 = test_c_f1_T              

            self.scheduler_fe.step()
            self.scheduler_d.step()
            self.scheduler_ac.step()
            #self.scheduler_wa.step()
            self.scheduler_cs.step()

            train_c_acc_S.reset()
            train_c_f1_S.reset()             

        print('>>> Training Finished!')
        return test_acc, test_f1
        
    def eval(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        criterion_c = nn.CrossEntropyLoss().cuda()

        test_c_acc_T = torchmetrics.Accuracy(task='multiclass', num_classes=self.N_classes).cuda()  
        test_c_f1_T = torchmetrics.F1Score(task='multiclass', num_classes=self.N_classes, average='macro').cuda()

        self.FE.eval()
        self.AC.eval()

        Loss_c = 0
        with torch.no_grad():
            for _, (x_T, y_T) in enumerate(self.test_loader):

                x_T = Variable(x_T.cuda())
                y_T = Variable(y_T.long().cuda())
                        
                _, logits_ac_T = self.forward_pass(x_T, 'C')
                loss_c = criterion_c(logits_ac_T, y_T)

                # track training losses and metrics 
                Loss_c += loss_c.item()
                test_c_acc_T(logits_ac_T.softmax(dim=-1), y_T)
                test_c_f1_T(logits_ac_T.softmax(dim=-1), y_T)
        
        self.FE.train()
        self.AC.train()

        return Loss_c, test_c_acc_T.compute().item(), test_c_f1_T.compute().item()
