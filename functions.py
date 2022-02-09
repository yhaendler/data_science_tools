## make a table summarizing performance metrics on training and testing data sets for binary classification models
def metricsTable(X_train, y_train, X_test, y_test, model):
    
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import pandas as pd
    
    ## training data scores
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, model.predict(X_train)).ravel()
    train_accuracy = (train_tn+train_tp) / (train_tn+train_tp+train_fn+train_fp)
    train_specificity = train_tn / (train_tn+train_fp)
    train_recall = train_tp / (train_tp+train_fn)
    train_precision = train_tp / (train_tp+train_fp)
    train_f1 = 2 * (train_precision*train_recall)/(train_precision+train_recall)
    train_roc_auc = roc_auc_score(y_train, model.predict(X_train))

    ## testing data scores
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    test_accuracy = (test_tn+test_tp) / (test_tn+test_tp+test_fn+test_fp)
    test_specificity = test_tn / (test_tn+test_fp)
    test_recall = test_tp / (test_tp+test_fn)
    test_precision = test_tp / (test_tp+test_fp)
    test_f1 = 2 * (test_precision*test_recall)/(test_precision+test_recall)
    test_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    
    ## data frame
    scores_df = pd.DataFrame({
        'accuracy' : [train_accuracy, test_accuracy],
        'specificity': [train_specificity, test_specificity],
        'recall': [train_recall, test_recall],
        'precision': [train_precision, test_precision],
        'f1': [train_f1, test_f1],
        'roc_auc': [train_roc_auc, test_roc_auc]
    }, index=['train','test'] )
    
    return scores_df

#--------------------------------------------------------------------------

## create a table summarizing how much missing data we have
def missingData(dataframe):
    
    import pandas as pd

    var_dict = {
        'feature_name': [],
        'n_missing_data': [],
        'percentage_missing_data': []    
    }

    for col in dataframe.columns:
        n_miss = dataframe[col].isnull().sum()
        if n_miss > 0:
            var_dict['feature_name'].append(col)
            var_dict['n_missing_data'].append(n_miss)
            var_dict['percentage_missing_data'].append(round(n_miss / dataframe.shape[0] * 100, 2))

    return pd.DataFrame(var_dict).sort_values(by='percentage_missing_data', ascending=False).reset_index(drop=True)

#--------------------------------------------------------------------------

## plot a heatmap showing the correlation between each feature and the target
def plotCorrWithTarget(dataframe, target):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    corr = dataframe.corr()

    plt.figure(figsize=(10,10))

    sns.heatmap(corr[[target]].sort_values(by=target, ascending=False),
               cmap = 'PuBu',
               annot = True, annot_kws={"size": 20},
               vmin = -1, vmax = 1, linewidth=0.1, linecolor='white');

    plt.yticks(size=16);
    plt.xticks(size=16);
    plt.show()

#--------------------------------------------------------------------------

## plot a heatmap showing the correlation among the features
def plotCorrHeatmap(dataframe, full_map=False, text_size=12):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    corr = round(dataframe.corr(),4)

    plt.figure(figsize=(24,24))
    
    if full_map:
        sns.heatmap(corr,
                   cmap = 'PuBu',
                   annot = True, annot_kws={"size": text_size},
                   vmin = -1, vmax = 1, linewidth=0.1, linecolor='white');
        
    else:
        sns.heatmap(corr,
                   cmap = 'PuBu',
                   annot = True, annot_kws={"size": text_size},
                   vmin = -1, vmax = 1, linewidth=0.1, linecolor='white',
                   mask = np.triu(np.ones_like(corr)));
    
    plt.yticks(size=14);
    plt.xticks(size=14);
    plt.tight_layout()
    plt.show()
    
#--------------------------------------------------------------------------

## regression imputation
def regressionImputation(df, predictor_feature, target_feature, stochastic = True):
    
    import numpy as np
    from sklearn.linear_model import LinearRegression
        
    # training data set - the predictor and target features without NA's
    train = df[[predictor_feature, target_feature]].dropna()
    # testing data set - the predictor and target features with NA's
    test = df[[predictor_feature, target_feature]][df[target_feature].isna()]

    # instantiate and fit the model on the training data
    model = LinearRegression().fit(X=train.drop(columns=target_feature), y=train[target_feature])

    # predictions for the NA's in the target feature
    preds = model.predict(test.drop(columns=target_feature))
    
    # if stochastic == True, add random noise to the predictions
    if stochastic:
        
        # create residuals - model errors on the training data
        residuals = train[target_feature] - model.predict(train.drop(columns=target_feature))
        
        # standard deviation of the residuals
        stdev = np.sqrt(residuals.var())
        
        # create random noise - a normal distribution with mean 0 and SD stdev
        random_noise = np.random.normal(loc = 0, scale = stdev, size = len(preds))
        
        # add random noise to the predictions to create random predictions
        random_preds = preds + random_noise
        
        # fill in the NA's in the target feature with the random predictions
        df[target_feature][df[target_feature].isna()] = random_preds
        
        return df
    
    # if stochastic == False, fill in the NA's in the target feature with the predictions (no random noise added)
    else:
        
        df[target_feature][df[target_feature].isna()] = preds
        return df