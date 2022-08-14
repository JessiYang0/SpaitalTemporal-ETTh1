from pyexpat import features
import matplotlib
import pandas
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from IPython import embed

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from IPython import embed

import os
import time

import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib.pyplot as plt

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # return model_optim

        param_sgd = dict(self.model.encoder.attn_layers.named_parameters())
        param_adam = dict(self.model.named_parameters())

        param_sgd_name = param_sgd.keys()
        param_sgd_name = ['encoder.attn_layers.' + name for name in param_sgd_name]
        [param_adam.pop(name) for name in param_sgd_name]

        model_optim1 = optim.Adam(param_adam.values(), lr=self.args.learning_rate)
        model_optim2 = optim.SGD(param_sgd.values(), lr=10 * self.args.learning_rate)
        return model_optim1 , model_optim2
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        model_optim1, model_optim2 = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                #batch_x : (batch, seq_len,feature(12個)) 
                #batch_x_mark : (batch, seq_len, time_feature(年月日時假日))
                #batch_y : (batch,pred+label_len, feature)
                #batch_y_mark : (batch,pred+label_len, time_feature)

                # model_optim.zero_grad()
                model_optim1.zero_grad()
                model_optim2.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                #print(f'x={pred}')

                loss = criterion(pred, true) #pred:(batch,pred_len,1)
                
                train_loss.append(loss.item())
                

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    #scaler.step(model_optim)
                    scaler.step(model_optim1)
                    scaler.step(model_optim2)
                    scaler.update()
                else:
                    loss.backward()
                    # model_optim.step()
                    model_optim1.step()
                    model_optim2.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch+1, self.args) #原始
            adjust_learning_rate(model_optim1, epoch+1, self.args)
            adjust_learning_rate(model_optim2, epoch+1, self.args) 
            
        ## best_model_path = path+'/'+'checkpoint.pth'
        ## self.model.load_state_dict(torch.load(best_model_path))
        
        
        ###############for shap#####################
        # model2 = self.model.cpu().eval()

        # batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))  #取batch
        # batch_x = batch_x.float() #(batch,seq_len,feature12)
        # batch_y = batch_y.float() 
        # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float() #(batch,seq_len,feature12)
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float()

        # batch_x_mark = batch_x_mark.float()
        # batch_y_mark = batch_y_mark.float()
        
        # e = shap.DeepExplainer(model2, [batch_x, batch_x_mark, dec_inp, batch_y_mark]) ##init

        # batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        # batch_x = batch_x.float()  #(32,168,12)
        # batch_y = batch_y.float() #(32,121,12)
        # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float()
        # batch_x_mark = batch_x_mark.float() #(32,168,5)
        # batch_y_mark = batch_y_mark.float() #(32,121,5)
        # shap_values = e.shap_values([batch_x, batch_x_mark, dec_inp, batch_y_mark]) #Return approximate SHAP values for the model applied to the data given by X.
        # # returns a tensor of SHAP values with the same shape as X
        # shap_values = shap_values[0][0][0] #第一個seq len 的第一個batch_x的第一個batch #(168,12)
        # e.expected_value


        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        
        # plt.rcParams["figure.figsize"] = (20, 16)
        # plt.rcParams["axes.labelsize"] = (30)
        # plt.rcParams['xtick.labelsize']= 18
        # plt.rcParams['ytick.labelsize']= 50 #30

        # #plt.figure(figsize = (22,20))
        # shap.summary_plot(shap_values,batch_x[0],feature_names=train_data.colsname, show=False) 
        # #取第一個batch所有timestep跟feature  #batch_x[0]=(168,12)
        # fig=plt.gcf()
        # fig.set_figheight(10) #16
        # fig.set_figwidth(20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=15)
        # plt.xlabel('SHAP value(impact on model output)', fontsize=20)
        # plt.savefig('img/summaryplot.jpg')
        # plt.close()


        # for i, shap_value in enumerate(shap_values): #sequence len個i ＃一個output對應sequence len的feature
        #     shap.bar_plot(shap_value, show=False, feature_names=train_data.colsname)
        #     plt.savefig('img2'+'/'f'tmp{i}.jpg')
        #     plt.close()
        #     shap.force_plot(e.expected_value[0], shap_value, matplotlib=True, show=False, feature_names=train_data.colsname) #expected value=base value, prediction len個expect value
        #     plt.savefig('img2' + '/' f'e{i}.jpg')
        #     plt.close()


        self.model.cuda()
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mae:{}, mspe:{}'.format(rmse, mae, mspe))  #mse, mae

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device) #(batch,seq_len,feature)
        batch_y = batch_y.float() #(batch,pred_len+label_len,feature)
    
        batch_x_mark = batch_x_mark.float().to(self.device) #(batch,seq_len,time_feature)
        batch_y_mark = batch_y_mark.float().to(self.device) #(batch,pred_len+label_len,time_feature)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float() #(batch,pred_len,feature)
    
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
    
        dec_inp  = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        #(batch,pred_len+label_len,feature)  #batch_y pred_len的部分mask掉了

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #(batch,pred_len,1)
            
                ##dec_inp為pred_len的部分馬掉，只輸入label_len
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device) ##(batch,pred_len,1)

        return outputs, batch_y

    
