import numpy as np
import pandas as pd
import warnings
import torch
import gpytorch
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
warnings.filterwarnings('ignore')


def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a

def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    
    m, p = x.shape
    _, h = t.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def GPR_model(X_train, y_train, X_test, tr, external_X_test = None):
    train_x, train_y = torch.Tensor(X_train.values), torch.Tensor(y_train[tr].values)
    test_x = torch.Tensor(X_test.values)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gpr_model = ExactGPModel(train_x, train_y, likelihood)
    training_iter = 200
    
    optimizer = torch.optim.Adam(gpr_model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr_model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = gpr_model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     gpr_model.covar_module.base_kernel.lengthscale.item(),
        #     gpr_model.likelihood.noise.item()))
        
        optimizer.step()  
        gpr_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_gpr = likelihood(gpr_model(test_x))
            pred_gpr = pred_gpr.mean.detach().cpu().numpy()
            if external_X_test is not None:
                external_test_x = torch.Tensor(external_X_test.values)
                pred_gpr_out = likelihood(gpr_model(external_test_x))
                pred_gpr_out = pred_gpr_out.mean.detach().cpu().numpy()
            else:
                pred_gpr_out = None
            
    return [pred_gpr, pred_gpr_out]

def PLSR_model(X_train, y_train, X_test, y_test, tr, external_X_test = None):
    PRESS = []
    for i in np.arange(10,31):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train[tr])
        
        pred = pls.predict(X_test)
        
        aa= np.array(pred.reshape(-1,).tolist())
        bb = np.array(y_test[tr].tolist())
        score = mean_squared_error(aa,bb)
        PRESS.append(score)
    n_components = PRESS.index(min(PRESS))+10
    print(tr,'n_components:',n_components)
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train[tr])
    
    pred_plsr = pls.predict(X_test)
    vvv = vip(X_train, y_train[tr], pls)
    coef = pls.coef_.reshape(-1,)
    
    if external_X_test is not None:
        pred_plsr_out = pls.predict(external_X_test)
    else:
        pred_plsr_out = None
    return [pred_plsr, pred_plsr_out, vvv, coef]

class RegressionModel(nn.Module):
    def __init__(self, input_size, nodes):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, nodes)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(nodes, nodes//2)
        self.act2 =nn.ReLU()
        self.out = nn.Linear(nodes//2, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(0)
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x =self.out(x)
        return x
    
def fine_tune_synthetic_data(synthetic_data_X, synthetic_data_y, tr, file_name):
    ### Pre-train data load
    scaler_x_LUT = StandardScaler()
    X_LUT = scaler_x_LUT.fit_transform(synthetic_data_X)
    scaler_y_LUT = StandardScaler()
    y_LUT = scaler_y_LUT.fit_transform(synthetic_data_y.reshape(-1,1))
    train_len_LUT=int(len(X_LUT)*0.9)
    X_train_tensor_LUT, y_train_tensor_LUT = torch.Tensor(X_LUT[:train_len_LUT]), torch.Tensor(y_LUT[:train_len_LUT])
    X_test_tensor_LUT, y_test_tensor_LUT = torch.Tensor(X_LUT[train_len_LUT:]), torch.Tensor(y_LUT[train_len_LUT:])
     
    train_loader_LUT = DataLoader(TensorDataset(X_train_tensor_LUT, y_train_tensor_LUT), batch_size=32, shuffle=True)
    model = RegressionModel(X_train_tensor_LUT.size()[1], 64)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
    best_mse = pow(10,10)
    history = []
    
    para_path=f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{file_name}_pre-trained model.pt'
    for epoch in range(300):                           #############             
        model.train()
        for X_batch, y_batch in train_loader_LUT:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_scheduler.step()
        model.eval()
        y_pred = model(X_test_tensor_LUT)
        mse = loss_fn(y_pred, y_test_tensor_LUT)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), para_path) # Save the model parameters to a file

    model.load_state_dict(torch.load(para_path)) # Load the saved parameters
    model.eval()
        
    output_x = model(X_train_tensor_LUT).detach().cpu().numpy()[:, 0]
    output_y = y_train_tensor_LUT.detach().cpu().numpy()
    plt.figure()
    sns.regplot(x=output_x, y=output_y)
    plt.title('Accurcy on training set')
    plt.xlabel(f'Train set predicted {tr}')
    plt.ylabel(f'Train set observed {tr}')
    plt.show()
    print(f'{file_name} accuracy on train sets $R^2$:',rsquared(y_train_tensor_LUT.detach().cpu().numpy()[:,0],model(X_train_tensor_LUT).detach().cpu().numpy()[:,0]))
    print(f"{file_name} accuracy on train sets RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    with torch.no_grad():
        pred_ann = model(X_test_tensor_LUT)
        pred_ann = pred_ann.detach().cpu().numpy()
        pred_ann=pred_ann[:,0]
        pred_ann = scaler_y_LUT.inverse_transform(pred_ann.reshape(-1,1))
        
        obs_y= y_test_tensor_LUT.detach().cpu().numpy()[:,0]
        obs_y_raw=scaler_y_LUT.inverse_transform(obs_y.reshape(-1,1))
        plt.figure()
        sns.regplot(x=pred_ann, y=obs_y_raw[:,0])
        plt.title('Accurcy on testing set')
        plt.xlabel(f'Predicted {tr}')
        plt.ylabel(f'Observed {tr}')
        plt.show()
        print(f'{file_name} accuracy on test sets $R^2$:',rsquared(obs_y_raw[:,0], pred_ann[:,0]))
        res = pd.DataFrame(np.concatenate((pred_ann,obs_y_raw), axis=1))
        res.columns = ['ANN_pred','LUT_obs']
        res.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/1_{file_name}_ANN_LUT_pred.csv',index = False)
    return pred_ann

def transfer_learning_model(X_train, y_train, X_test, y_test, tr, file_name, dataset,iteration):
    ### observational data load
    X, y =np.concatenate((X_train.values,X_test.values), axis=0),np.concatenate((y_train[tr].values,y_test[tr].values), axis=0)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y=scaler_y.fit_transform(y.reshape(-1,1))
    train_len=len(X_train)
    X_train_tensor, y_train_tensor = torch.Tensor(X[:train_len]), torch.Tensor(y[:train_len])
    X_test_tensor, y_test_tensor = torch.Tensor(X[train_len:]), torch.Tensor(y[train_len:])
    
    model = RegressionModel(X_train_tensor.size()[1], 64) 
    model_path = f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{file_name}_pre-trained model.pt'
    model.load_state_dict(torch.load(model_path))
    
    loss_fn = nn.L1Loss() 
    param_grid = {'fine_tune_lr':[0.0005, 0.001,0.005,0.01],'fine_tune_bs':[10,16,32,64]}
    # param_grid = {'fine_tune_lr':[0.005],'fine_tune_bs':[10]}
    grid = ParameterGrid(param_grid)
    best_accuracy = []
    best_params = []
    for paras in grid:
        fine_tune_learning_rate = paras['fine_tune_lr']
        fine_tune_batch_size = paras['fine_tune_bs']
        
        ### fine-tune process
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=fine_tune_batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=fine_tune_learning_rate)  # Use a smaller learning rate for fine-tuning
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
        best_mse = pow(10,10)
        best_weights = None
        history = []
        
        for epoch in range(300):                                   ############         
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_scheduler.step()
            model.eval()
            y_pred = model(X_test_tensor)
            mse = loss_fn(y_pred, y_test_tensor)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_weights = copy.deepcopy(model.state_dict())
                best_mse = mse
                     
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            pred_ann = model(X_test_tensor)
            pred_ann = pred_ann.detach().cpu().numpy()
            pred_ann=pred_ann[:,0]
            pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
            
            obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
            obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
            accu = rsquared(obs_y_raw[:,0], pred_ann[:,0])
            
            # print(paras,'R^2:', accu)
            best_accuracy.append(accu)
            best_params.append(paras)
            
    #######obtained  best  parameters   
    new_paras = best_params[best_accuracy.index(max(best_accuracy))]
    print(f'{round((len(X_train)/len(X))*100)}% fine-tune {file_name} transfer learning model: Best Parameters', new_paras)
    fine_tune_learning_rate = paras['fine_tune_lr']
    fine_tune_batch_size = paras['fine_tune_bs']
    
    model = RegressionModel(X_train_tensor.size()[1], 64) 
    model_path = f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{file_name}_pre-trained model.pt'
    model.load_state_dict(torch.load(model_path))

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=fine_tune_batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_learning_rate)  # Use a smaller learning rate for fine-tuning
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
    
    best_mse = pow(10,10)
    history = []
    para_path=f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{tr}/{tr}_{round((len(X_train)/len(X))*100)}% fine-tune_{file_name}_transfer learning model_iteration_{iteration+1}_{dataset} dataset.pt'
    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch) # Get and prepare inputs # forward pass
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_scheduler.step()
        model.eval() # evaluate accuracy at end of each epoch
        y_pred = model(X_test_tensor)
        mse = loss_fn(y_pred, y_test_tensor)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), para_path) # Save the model parameters to a file

    model.load_state_dict(torch.load(para_path)) # Load the saved parameters
    model.eval()
    ### check the model performance at training sets
    output_x = model(X_train_tensor).detach().cpu().numpy()[:, 0]
    output_y = y_train_tensor.detach().cpu().numpy()
    plt.figure()
    sns.regplot(x=output_x, y=output_y)
    plt.title('Accurcy on training set')
    plt.xlabel(f'Train set predicted {tr}')
    plt.ylabel(f'Train set observed {tr}')
    plt.show()
    print(f'{round((len(X_train)/len(X))*100)}% fine-tune {file_name} transfer learning model_{dataset}: accuracy on train sets $R^2$:',rsquared(y_train_tensor.detach().cpu().numpy()[:,0],model(X_train_tensor).detach().cpu().numpy()[:,0]))
    print(f"{round((len(X_train)/len(X))*100)}% fine-tune {file_name} transfer learning model_{dataset}: accuracy on train sets RMSE:", np.sqrt(best_mse))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    with torch.no_grad():
        pred_ann = model(X_test_tensor)
        pred_ann = pred_ann.detach().cpu().numpy()
        pred_ann=pred_ann[:,0]
        pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
        
        obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
        obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
        plt.figure()
        sns.regplot(x=pred_ann, y=obs_y_raw[:,0])
        plt.title('Accurcy on testing set')
        plt.xlabel(f'Predicted {tr}')
        plt.ylabel(f'Observed {tr}')
        plt.show()
        print(f'{round((len(X_train)/len(X))*100)}% fine-tune {file_name} transfer learning model: accuracy on test sets $R^2$:',rsquared(obs_y_raw[:,0], pred_ann[:,0]))
    return pred_ann

def transfer_learning_model_transferability(X_train, y_train, X_test, y_test, tr, file_name, trans_type, external_X_test = None):
    ### observational data load
    X, y =np.concatenate((X_train.values,X_test.values), axis=0),np.concatenate((y_train[tr].values,y_test[tr].values), axis=0)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y=scaler_y.fit_transform(y.reshape(-1,1))
    train_len=len(X_train)
    X_train_tensor, y_train_tensor = torch.Tensor(X[:train_len]), torch.Tensor(y[:train_len])
    X_test_tensor, y_test_tensor = torch.Tensor(X[train_len:]), torch.Tensor(y[train_len:])
    
    model = RegressionModel(X_train_tensor.size()[1], 64) 
    model_path = f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{file_name}_pre-trained model.pt'
    model.load_state_dict(torch.load(model_path))
    
    loss_fn = nn.L1Loss() 
    param_grid = {'fine_tune_lr':[0.0005, 0.001,0.005,0.01],'fine_tune_bs':[10,16,32,64]}
    grid = ParameterGrid(param_grid)
    best_accuracy = []
    best_params = []
    
    for paras in grid:
        fine_tune_learning_rate = paras['fine_tune_lr']
        fine_tune_batch_size = paras['fine_tune_bs']
        
        ### fine-tune process
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=fine_tune_batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=fine_tune_learning_rate)  # Use a smaller learning rate for fine-tuning
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
        best_mse = pow(10,10)
        best_weights = None
        history = []
        
        for epoch in range(300):                                   ############         
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_scheduler.step()
            model.eval()
            y_pred = model(X_test_tensor)
            mse = loss_fn(y_pred, y_test_tensor)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_weights = copy.deepcopy(model.state_dict())
                best_mse = mse
                     
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            pred_ann = model(X_test_tensor)
            pred_ann = pred_ann.detach().cpu().numpy()
            pred_ann=pred_ann[:,0]
            pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
            
            obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
            obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
            accu = rsquared(obs_y_raw[:,0], pred_ann[:,0])
            
            # print(paras,'R^2:', accu)
            best_accuracy.append(accu)
            best_params.append(paras)
            
    #######obtained  best  parameters   
    new_paras = best_params[best_accuracy.index(max(best_accuracy))]
    print(f'{trans_type} fine-tune {file_name} transfer learning model: Best Parameters', new_paras)
    fine_tune_learning_rate = paras['fine_tune_lr']
    fine_tune_batch_size = paras['fine_tune_bs']
    
    model = RegressionModel(X_train_tensor.size()[1], 64) 
    model_path = f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{file_name}_pre-trained model.pt'
    model.load_state_dict(torch.load(model_path))

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=fine_tune_batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_learning_rate)  # Use a smaller learning rate for fine-tuning
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
    
    ### fine-tune process
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=fine_tune_batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_learning_rate)  # Use a smaller learning rate for fine-tuning
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)
    best_mse = pow(10,10)
    best_weights = None
    history = []
    
    para_path=f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{tr}/{tr}_{trans_type} transfer learning model.pt'
    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch) # Get and prepare inputs # forward pass
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_scheduler.step()
        model.eval() # evaluate accuracy at end of each epoch
        y_pred = model(X_test_tensor)
        mse = loss_fn(y_pred, y_test_tensor)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), para_path) # Save the model parameters to a file

    model.load_state_dict(torch.load(para_path)) # Load the saved parameters
    model.eval()
    ### check the model performance at training sets
    output_x = model(X_train_tensor).detach().cpu().numpy()[:, 0]
    output_y = y_train_tensor.detach().cpu().numpy()
    plt.figure()
    sns.regplot(x=output_x, y=output_y)
    plt.title('Accurcy on training set')
    plt.xlabel(f'Train set predicted {tr}')
    plt.ylabel(f'Train set observed {tr}')
    plt.show()
    print(f'{tr}_{trans_type}: accuracy on train sets $R^2$:',rsquared(y_train_tensor.detach().cpu().numpy()[:,0],model(X_train_tensor).detach().cpu().numpy()[:,0]))
    print(f"{tr}_{trans_type}: accuracy on train sets RMSE:", np.sqrt(best_mse))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    with torch.no_grad():
        pred_ann = model(X_test_tensor)
        pred_ann = pred_ann.detach().cpu().numpy()
        pred_ann=pred_ann[:,0]
        pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
        
        obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
        obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
        plt.figure()
        sns.regplot(x=pred_ann, y=obs_y_raw[:,0])
        plt.title('Accurcy on testing set')
        plt.xlabel(f'Predicted {tr}')
        plt.ylabel(f'Observed {tr}')
        plt.show()
        print(f'{tr}_{trans_type}: accuracy on test sets $R^2$:',rsquared(obs_y_raw[:,0], pred_ann[:,0]))
        if external_X_test is not None:
            external_X_test=scaler_x.transform(external_X_test)
            external_test_x = torch.Tensor(external_X_test)
            pred_ann_out = model(external_test_x)
            pred_ann_out = pred_ann_out.detach().cpu().numpy()
            pred_ann_out = pred_ann_out[:,0]
            pred_ann_out = scaler_y.inverse_transform(pred_ann_out.reshape(-1,1))
        else:
            pred_ann_out = None
    return [pred_ann,pred_ann_out]

def DNN_model(X_train, y_train, X_test, y_test, tr, dataset, iteration, external_X_test = None):
    X, y =np.concatenate((X_train.values,X_test.values), axis=0),np.concatenate((y_train[tr].values,y_test[tr].values), axis=0)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y=scaler_y.fit_transform(y.reshape(-1,1))

    train_len=len(X_train)
    X_train_tensor, y_train_tensor = torch.Tensor(X[:train_len]), torch.Tensor(y[:train_len])
    X_test_tensor, y_test_tensor = torch.Tensor(X[train_len:]), torch.Tensor(y[train_len:])
    
    param_grid = {'learning_rate': [0.001, 0.005],'batch_size': [10,16,32,64],'nodes': np.arange(64,66,2)}
    # param_grid = {'learning_rate': [0.005],'batch_size': [10],'nodes': np.arange(64,66,2)}
    grid = ParameterGrid(param_grid)

    best_accuracy = []
    best_params = []
    for paras in grid:
        nodes = paras['nodes']
        learning_rate = paras['learning_rate']
        batch_size = paras['batch_size']
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        model = RegressionModel(X_train_tensor.size()[1], nodes)
    
        loss_fn = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
        best_mse = pow(10,10)
        best_weights = None
        history = []
        for epoch in range(300):
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_scheduler.step()
            model.eval()
            y_pred = model(X_test_tensor)
            mse = loss_fn(y_pred, y_test_tensor)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_weights = copy.deepcopy(model.state_dict())
                best_mse = mse
        
        model.load_state_dict(best_weights)
        model.eval()
        
        with torch.no_grad():
            pred_ann = model(X_test_tensor)
            pred_ann = pred_ann.detach().cpu().numpy()
            pred_ann=pred_ann[:,0]
            pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
            
            obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
            obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
            accu = rsquared(obs_y_raw[:,0], pred_ann[:,0])
            
            # print(paras,'R^2:', accu)
            best_accuracy.append(accu)
            best_params.append(paras)
    
    #######obtained  best  parameters   
    new_paras = best_params[best_accuracy.index(max(best_accuracy))]
    print(tr,f'{round((len(X_train)/len(X))*100)}% train_DNN_iteration_{iteration+1}_{dataset}_dataset', 'Best Parameters', new_paras)
    nodes = new_paras['nodes']
    learning_rate = new_paras['learning_rate']
    batch_size = new_paras['batch_size']    
     
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    model = RegressionModel(X_train_tensor.size()[1], nodes)
    # loss function and optimizer
    loss_fn = nn.L1Loss()  #nn.MSELoss() mean square error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
    
    n_epochs = 300    # number of epochs to run
    # Hold the best model
    best_mse = pow(10,10)   # init to infinity
    history = []
    para_path=f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{tr}/{tr}_{round((len(X_train)/len(X))*100)}% train_DNN model_iteration_{iteration+1}_{dataset} dataset.pt'
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch) # Get and prepare inputs # forward pass
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_scheduler.step()
        model.eval() # evaluate accuracy at end of each epoch
        y_pred = model(X_test_tensor)
        mse = loss_fn(y_pred, y_test_tensor)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), para_path) # Save the model parameters to a file
    
    model.load_state_dict(torch.load(para_path)) # Load the saved parameters
    model.eval()
    ### check the model performance at training sets
    # output_x = model(X_train_tensor).detach().cpu().numpy()[:, 0]
    # output_y = y_train_tensor.detach().cpu().numpy()
    # plt.figure()
    # sns.regplot(x=output_x, y=output_y)
    # plt.title('Accurcy on training set')
    # plt.xlabel(f'Train set predicted {tr}')
    # plt.ylabel(f'Train set observed {tr}')
    # plt.show()
    print(tr,f'{round((len(X_train)/len(X))*100)}% train_DNN_iteration_{iteration+1}_{dataset}_dataset: accuracy on train sets $R^2$:',rsquared(y_train_tensor.detach().cpu().numpy()[:,0],model(X_train_tensor).detach().cpu().numpy()[:,0]))
    print(tr,f'{round((len(X_train)/len(X))*100)}% train_DNN_iteration_{iteration+1}_{dataset}_dataset: accuracy on train sets RMSE:',  np.sqrt(best_mse))
    # plt.plot(history)
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.show()
    with torch.no_grad():
        pred_ann = model(X_test_tensor)
        pred_ann = pred_ann.detach().cpu().numpy()
        pred_ann=pred_ann[:,0]
        pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
        
        obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
        obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
        # plt.figure()
        # sns.regplot(x=pred_ann, y=obs_y_raw[:,0])
        # plt.title('Accurcy on testing set')
        # plt.xlabel(f'Predicted {tr}')
        # plt.ylabel(f'Observed {tr}')
        # plt.show()
        print(tr,f'{round((len(X_train)/len(X))*100)}% train_DNN_iteration_{iteration+1}_{dataset}_dataset', '$R^2$:',rsquared(obs_y_raw[:,0], pred_ann[:,0]))
        if external_X_test is not None:
            external_X_test=scaler_x.transform(external_X_test)
            external_test_x = torch.Tensor(external_X_test)
            pred_ann_out = model(external_test_x)
            pred_ann_out = pred_ann_out.detach().cpu().numpy()
            pred_ann_out = pred_ann_out[:,0]
            pred_ann_out = scaler_y.inverse_transform(pred_ann_out.reshape(-1,1))
        else:
            pred_ann_out = None
    return [pred_ann, pred_ann_out]

def DNN_model_transferabiliy(X_train, y_train, X_test, y_test, tr, trans_type, external_X_test = None):
    X, y =np.concatenate((X_train.values,X_test.values), axis=0),np.concatenate((y_train[tr].values,y_test[tr].values), axis=0)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y=scaler_y.fit_transform(y.reshape(-1,1))

    train_len=len(X_train)
    X_train_tensor, y_train_tensor = torch.Tensor(X[:train_len]), torch.Tensor(y[:train_len])
    X_test_tensor, y_test_tensor = torch.Tensor(X[train_len:]), torch.Tensor(y[train_len:])
    
    param_grid = {'learning_rate': [0.001, 0.005],'batch_size': [10,16,32,64],'nodes': np.arange(64,66,2)}
    # param_grid = {'learning_rate': [0.005],'batch_size': [10],'nodes': np.arange(64,66,2)}
    grid = ParameterGrid(param_grid)

    best_accuracy = []
    best_params = []
    for paras in grid:
        nodes = paras['nodes']
        learning_rate = paras['learning_rate']
        batch_size = paras['batch_size']
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        model = RegressionModel(X_train_tensor.size()[1], nodes)
    
        loss_fn = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
        best_mse = pow(10,10)
        best_weights = None
        history = []
        for epoch in range(300):
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_scheduler.step()
            model.eval()
            y_pred = model(X_test_tensor)
            mse = loss_fn(y_pred, y_test_tensor)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_weights = copy.deepcopy(model.state_dict())
                best_mse = mse
        
        model.load_state_dict(best_weights)
        model.eval()
        
        with torch.no_grad():
            pred_ann = model(X_test_tensor)
            pred_ann = pred_ann.detach().cpu().numpy()
            pred_ann=pred_ann[:,0]
            pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
            
            obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
            obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
            accu = rsquared(obs_y_raw[:,0], pred_ann[:,0])
            
            # print(paras,'R^2:', accu)
            best_accuracy.append(accu)
            best_params.append(paras)
    
    #######obtained  best  parameters   
    new_paras = best_params[best_accuracy.index(max(best_accuracy))]
    print('Best Parameters', new_paras)
    nodes = new_paras['nodes']
    learning_rate = new_paras['learning_rate']
    batch_size = new_paras['batch_size']    
     
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    model = RegressionModel(X_train_tensor.size()[1], nodes)
    # loss function and optimizer
    loss_fn = nn.L1Loss()  #nn.MSELoss() mean square error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
    
    n_epochs = 300    # number of epochs to run
    # Hold the best model
    best_mse = pow(10,10)   # init to infinity
    history = []
    para_path=f'/scratch/fji7/transfer_learning_paper/2_saved_ML_model/{tr}/{tr}_{trans_type} train_DNN model.pt'
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch) # Get and prepare inputs # forward pass
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_scheduler.step()
        model.eval() # evaluate accuracy at end of each epoch
        y_pred = model(X_test_tensor)
        mse = loss_fn(y_pred, y_test_tensor)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), para_path) # Save the model parameters to a file
    
    model.load_state_dict(torch.load(para_path)) # Load the saved parameters
    model.eval()
    ### check the model performance at training sets
    output_x = model(X_train_tensor).detach().cpu().numpy()[:, 0]
    output_y = y_train_tensor.detach().cpu().numpy()
    plt.figure()
    sns.regplot(x=output_x, y=output_y)
    plt.title('Accurcy on training set')
    plt.xlabel(f'Train set predicted {tr}')
    plt.ylabel(f'Train set observed {tr}')
    plt.show()
    print(f'{tr}_{trans_type} accuracy on train sets $R^2$:',rsquared(y_train_tensor.detach().cpu().numpy()[:,0],model(X_train_tensor).detach().cpu().numpy()[:,0]))
    print(f"{tr}_{trans_type} accuracy on train sets RMSE:", np.sqrt(best_mse))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    with torch.no_grad():
        pred_ann = model(X_test_tensor)
        pred_ann = pred_ann.detach().cpu().numpy()
        pred_ann=pred_ann[:,0]
        pred_ann = scaler_y.inverse_transform(pred_ann.reshape(-1,1))
        
        obs_y= y_test_tensor.detach().cpu().numpy()[:,0]
        obs_y_raw=scaler_y.inverse_transform(obs_y.reshape(-1,1))
        plt.figure()
        sns.regplot(x=pred_ann, y=obs_y_raw[:,0])
        plt.title('Accurcy on testing set')
        plt.xlabel(f'Predicted {tr}')
        plt.ylabel(f'Observed {tr}')
        plt.show()
        print(f'{tr}_{trans_type} accuracy on test sets $R^2$:',rsquared(obs_y_raw[:,0], pred_ann[:,0]))
        if external_X_test is not None:
            external_X_test=scaler_x.transform(external_X_test)
            external_test_x = torch.Tensor(external_X_test)
            pred_ann_out = model(external_test_x)
            pred_ann_out = pred_ann_out.detach().cpu().numpy()
            pred_ann_out = pred_ann_out[:,0]
            pred_ann_out = scaler_y.inverse_transform(pred_ann_out.reshape(-1,1))
        else:
            pred_ann_out = None
    return [pred_ann, pred_ann_out]

    
    
        
        
        
        
            
            
    
    
    
    






