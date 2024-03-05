from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'G:\code\autoML4_suiji28_43site_ev\data\shiyan1_ev.csv',header=0,index_col=None)
label = 'GPP_NT_VUT_MEAN'
results = []
for i in range(10):
    print(f"Iteration {i+1}")
    X = data.drop(columns=[label]) 
    y = data[label] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.head()
    print("Summary of class variable: \n", train_data[label].describe())
    save_path = f'agModels-predictClass{i}'  
    predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
    y_test = test_data[label]  
    test_data_nolab = test_data.drop(columns=[label])  
    test_data_nolab.head()
    predictor = TabularPredictor.load(save_path) 
    y_pred = predictor.predict(test_data_nolab)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    predictor.leaderboard(test_data, silent=True)
    feature_importance = predictor.feature_importance(test_data)
    feature_importance['Feature'] = feature_importance.index
    feature_importance = feature_importance[['Feature'] + feature_importance.columns[:-1].tolist()]
    feature_importance.to_csv(f'feature_importance{i}.csv', index=False)
    pre_train_NeuralNetFastAI = predictor.predict(train_data, model='NeuralNetFastAI')
    pre_train_LightGBMXT = predictor.predict(train_data, model='LightGBMXT')
    pre_train_LightGBM = predictor.predict(train_data, model='LightGBM')
    pre_train_CatBoost = predictor.predict(train_data, model='CatBoost')
    pre_train_WeightedEnsemble_L2 = predictor.predict(train_data, model='WeightedEnsemble_L2')
    pre_train_NeuralNetTorch = predictor.predict(train_data, model='NeuralNetTorch')
    pre_train_LightGBMLarge = predictor.predict(train_data, model='LightGBMLarge')
    pre_train_ExtraTreesMSE = predictor.predict(train_data, model='ExtraTreesMSE')
    pre_train_XGBoost = predictor.predict(train_data, model='XGBoost')
    pre_train_RandomForestMSE = predictor.predict(train_data, model='RandomForestMSE')
    pre_train_KNeighborsUnif = predictor.predict(train_data, model='KNeighborsUnif')
    pre_train_KNeighborsDist = predictor.predict(train_data, model='KNeighborsDist')
    pf_train = pd.DataFrame({'GPP_NT_VUT_MEAN':train_data['GPP_NT_VUT_MEAN'],
                            'pre_train_NeuralNetFastAI':pre_train_NeuralNetFastAI,
                            'pre_train_LightGBMXT':pre_train_LightGBMXT,
                            'pre_train_LightGBM':pre_train_LightGBM,
                            'pre_train_CatBoost':pre_train_CatBoost,
                            'pre_train_WeightedEnsemble_L2':pre_train_WeightedEnsemble_L2,
                            'pre_train_NeuralNetTorch':pre_train_NeuralNetTorch,
                            'pre_train_LightGBMLarge':pre_train_LightGBMLarge,
                            'pre_train_ExtraTreesMSE':pre_train_ExtraTreesMSE,
                            'pre_train_XGBoost':pre_train_XGBoost,
                            'pre_train_RandomForestMSE':pre_train_RandomForestMSE,
                            'pre_train_KNeighborsUnif':pre_train_KNeighborsUnif,
                            'pre_train_KNeighborsDist':pre_train_KNeighborsDist})
    pf_train.to_csv(f'G:/code/autoML4_suiji28_43site_ev/output_shiyan1ev_train{i}.csv')
    pre_test_NeuralNetFastAI = predictor.predict(test_data, model='NeuralNetFastAI')
    pre_test_LightGBMXT = predictor.predict(test_data, model='LightGBMXT')
    pre_test_LightGBM = predictor.predict(test_data, model='LightGBM')
    pre_test_CatBoost = predictor.predict(test_data, model='CatBoost')
    pre_test_WeightedEnsemble_L2 = predictor.predict(test_data, model='WeightedEnsemble_L2')
    pre_test_NeuralNetTorch = predictor.predict(test_data, model='NeuralNetTorch')
    pre_test_LightGBMLarge = predictor.predict(test_data, model='LightGBMLarge')
    pre_test_ExtraTreesMSE = predictor.predict(test_data, model='ExtraTreesMSE')
    pre_test_XGBoost = predictor.predict(test_data, model='XGBoost')
    pre_test_RandomForestMSE = predictor.predict(test_data, model='RandomForestMSE')
    pre_test_KNeighborsUnif = predictor.predict(test_data, model='KNeighborsUnif')
    pre_test_KNeighborsDist = predictor.predict(test_data, model='KNeighborsDist')
    pf_test = pd.DataFrame({'GPP_NT_VUT_MEAN':test_data['GPP_NT_VUT_MEAN'],
                            'pre_test_NeuralNetFastAI':pre_test_NeuralNetFastAI,
                            'pre_test_LightGBMXT':pre_test_LightGBMXT,
                            'pre_test_LightGBM':pre_test_LightGBM,
                            'pre_test_CatBoost':pre_test_CatBoost,
                            'pre_test_WeightedEnsemble_L2':pre_test_WeightedEnsemble_L2,
                            'pre_test_NeuralNetTorch':pre_test_NeuralNetTorch,
                            'pre_test_LightGBMLarge':pre_test_LightGBMLarge,
                            'pre_test_ExtraTreesMSE':pre_test_ExtraTreesMSE,
                            'pre_test_XGBoost':pre_test_XGBoost,
                            'pre_test_RandomForestMSE':pre_test_RandomForestMSE,
                            'pre_test_KNeighborsUnif':pre_test_KNeighborsUnif,
                            'pre_test_KNeighborsDist':pre_test_KNeighborsDist})

    pf_test.to_csv(f'G:/code/autoML4_suiji28_43site_ev/output_shiyan1ev_test{i}.csv')