import model as ST_model
import data_loader_with_ST as data_loader 
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.nn.functional import relu
import numpy as np
import random 

#random.seed(10)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total: ' + str(total_num) + '; trainable: ' + str(trainable_num))



torch.cuda.set_device(0)
data = np.load('./src_mask_final_4.npz')
adjcent = data['arr_0']
region_mask = torch.BoolTensor(200,200).cuda()
for i in range(200):
    for j in range(200):
        if adjcent[i][j] == 0:
            region_mask[i][j] = False
        else:
            region_mask[i][j] = True
region_mask.cuda()




def evaluate(y, pred_y, threshold):
    mask = y >= threshold
    if torch.sum(mask) != 0:
        rmse = torch.sqrt(torch.mean( (y[mask] - pred_y[mask]) * (y[mask] - pred_y[mask])))
        #mape = torch.mean(torch.abs(y[mask] - pred_y[mask]) / y[mask])
        #mse = torch.mean( (y[mask] - pred_y[mask]) * (y[mask] - pred_y[mask]))
        return rmse

    else:
        return -1

def eval(y, pred_y, threshold):
    mask = y >= threshold
    if torch.sum(mask) != 0:
        rmse = torch.sqrt(torch.mean( (y[mask] - pred_y[mask]) * (y[mask] - pred_y[mask])))
        mape = torch.mean(torch.abs(y[mask] - pred_y[mask]) / y[mask])
        #rmse = torch.mean((y[mask] - pred_y[mask]) * (y[mask] - pred_y[mask]))
        #mape = torch.mean(torch.abs(y[mask] - pred_y[mask]) / y[mask])
        return rmse, mape
    else:
        return -1, -1


def testing(model, loader, threshoad,region_mask):
    loss = 0
    in_count = 0
    out_count = 0
    result = 0
    in_rmse_total = 0
    in_mape_total = 0
    out_rmse_total = 0
    out_mape_total = 0
    model.eval()
    for step, (batch_x, batch_y) in enumerate(loader):

        input_data = batch_x.transpose(0, 1).cuda()
        spatial_embedding = input_data[:,:,0:200]
        temporal_embedding = input_data[:,:,200:248]
        src = input_data[:,:,248:input_data.shape[2]]
        target = batch_y.transpose(0, 1).cuda()
        # target = batch_y.transpose(0,1)
        output = model(src,spatial_embedding,temporal_embedding,region_mask)
        # print(output.shape)
        output.cuda()
        in_rmse, in_mape = eval(target[:, :, 0], output[:, :, 0], threshoad)
        out_rmse, out_mape = eval(target[:, :, 1], output[:, :, 1], threshoad)
        if in_rmse != -1:
            in_count += 1
            in_rmse_total += in_rmse.item()
            in_mape_total += in_mape.item()
        if out_rmse != -1:
            out_count += 1
            out_rmse_total += out_rmse.item()
            out_mape_total += out_mape.item()
        # evaluate_test(target[:,:,0],output[:,:,0],10.0/299,filename_1,epoch)
        # evaluate_test(target[:,:,1],output[:,:,1],10.0/265,filename_2,epoch)
    del input_data
    del target
    del output
    torch.cuda.empty_cache()
        
    in_rmse = in_rmse_total / in_count
    in_mape = in_mape_total / in_count
    out_rmse = out_rmse_total / out_count
    out_mape = out_mape_total / out_count
    return in_rmse, in_mape, out_rmse, out_mape



def validation(model, val_loader, threshold,region_mask):
    model.eval()
    rmse_total = 0
    mape_total = 0
    count = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(val_loader):
            input_data = batch_x.transpose(0, 1).cuda()
            spatial_embedding = input_data[:,:,0:200]
            temporal_embedding = input_data[:,:,200:248]
            src = input_data[:,:,248:input_data.shape[2]]
            target = batch_y.transpose(0, 1).cuda()
            output = model(src,spatial_embedding,temporal_embedding,region_mask)
            output.cuda()
            rmse, mape = eval(target, output, threshold)
            if rmse != -1:
                rmse_total += rmse.item()
                mape_total += mape.item()
                count += 1
     
    del input_data
    del target
    del output
    torch.cuda.empty_cache()       
    return rmse_total / count , mape_total / count


def train(threshold, dataset, max_epoch = 1000, patience = 20, num_layer = 4, head_num = 4, type = 'bike'):
    f = open('./result/exp_result_layer_num_' + str(num_layer) + '_head_num_' + str(head_num) + '_' + str(type) + '_1112_dim_8-2.csv','a')
    config = json.load(open(dataset, "r"))
    sample = data_loader.data_loader(config_path=dataset)
    # sample = data_loader.data_loader()
    train_loader, test_loader, val_loader = sample.load_data()
    #train_loader = sample.load_data()
    print('data loaded...')

    torch.cuda.set_device(0)
    model = ST_model.STForecasting(num_encoder_layers = num_layer, dataset = dataset, d_model = 8 * head_num, nhead = head_num).cuda()
    print('model loaded...')
    #model.load_state_dict(torch.load(  './data/model/case_study_model_epoch_290_0702.pkl'))
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trigger = 0
    last_loss = 100000
    last_mape_loss = 100000
    best_epoch = 0
    best_state = model.state_dict()
    

    for epoch in range(1, max_epoch):
        region_mask = gen_mask()
        region_mask.cuda()
        t1 = time.time()
        model.train()
        #if epoch == 50:
        #    learning_rate = 0.0001
        #    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if epoch >= 10 and epoch % 10 ==0:
            learning_rate = learning_rate * 0.9
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        '''if epoch == 120:
            learning_rate = 0.00001
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)'''
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            input_data = batch_x.transpose(0,1).cuda()
            spatial_embedding = input_data[:,:,0:200]
            temporal_embedding = input_data[:,:,200:248]
            src = input_data[:,:,248:input_data.shape[2]]
            target = batch_y.transpose(0, 1).cuda()
            output = model(src,spatial_embedding,temporal_embedding, region_mask)
            output.cuda()
            loss = evaluate(target, output, threshold)
            #print(str(epoch) + ' - ' + str(step) + ' : ' + str(loss))
            if loss == -1:
                continue
            loss.backward()
            optimizer.step()
        t2 = time.time()
        print('Epoch ' + str(epoch) + ': ' + str(t2 - t1))
        #current_loss, mape_loss = validation(model, val_loader, threshold)
        current_loss, mape_loss = validation(model, val_loader, threshold,region_mask)
        print('Epoch ' + str(epoch))
        print('Validation RMSE is ' + str(current_loss * config['volume_train_max'][0]))
        print('Validation MAPE is ' + str(mape_loss))
        print('-----')
        #test_rmse, test_mape = validation(model, test_loader, threshold)
        #print('Testing RMSE is ' + str(test_rmse * config['volume_train_max'][0]))
        #print('Testing MAPE is ' + str(test_mape))
        #print('-----')
        in_rmse, in_mape, out_rmse, out_mape = testing(model, test_loader, threshold,region_mask)
        print('In RMSE: ' + str(in_rmse * config['volume_train_max'][0]))
        print('In MAPE: ' + str(in_mape))
        print('Out RMSE: ' + str(out_rmse * config['volume_train_max'][0]))
        print('Out MAPE: ' + str(out_mape))
        print('-------------------------------------------')
        f.write(str(epoch) + ',' + str(current_loss * config['volume_train_max'][0]) + ',' + str(mape_loss) + ',' +  str(in_rmse * config['volume_train_max'][0]) + ',' + str(in_mape) + ',' + str(out_rmse * config['volume_train_max'][0]) + ',' + str(out_mape) + '\n')

        if epoch >= 100:
            if current_loss >= last_loss:
                trigger += 1
                if trigger >= patience:
                    print('Early Stopping! The best epoch is ' + str(best_epoch))
                    f.write('Early Stopping! The best epoch is ' + str(best_epoch) + '\n')
                    f.write('==========================================================\n')
                    torch.save(best_state, './data/model/teast-sqrt2-final.pkl')
                    break

            else:
                trigger = 0
                last_loss = current_loss
                last_mape_loss = mape_loss
                best_epoch = epoch
                best_state = model.state_dict()

    f.close()



if __name__ == '__main__':
    config_path = './taxi_data.json'
    #config_path = './bike_data.json'
    #config = json.load(open(config_path, "r"))
    threshold = 10 / config['volume_train_max'][0]
    num = 4
    train(threshold, config_path, max_epoch=1000, patience=20, num_layer=3, head_num=num, type='bike')
    '''for config_path in ['./bike_data.json']:
        config = json.load(open(config_path, "r"))
        threshold = 10 / config['volume_train_max'][0]
        for num in [8]:
            for i in range(3):
                train(threshold, config_path, max_epoch=1000, patience=20, num_layer = 3, head_num = num, type = 'bike')

    for config_path in ['./taxi_data.json']:
        config = json.load(open(config_path, "r"))
        threshold = 10 / config['volume_train_max'][0]
        for num in [8]:
            for i in range(3):
                train(threshold, config_path, max_epoch=1000, patience=20, num_layer = 3, head_num = num, type = 'taxi')'''
    '''num_layer = 4
    head_num = 8
    dataset =  './bike_data.json'
    model = ST_model.STForecasting(num_encoder_layers=num_layer, dataset=dataset, d_model=8 * head_num,
                                   nhead=head_num).cuda()

    get_parameter_number(model)'''

