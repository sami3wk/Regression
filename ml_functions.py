class Missing_data:
    def catconsep(self,df):
        cat = []
        con = []
        for i in df.columns:
            if(df[i].dtypes == "object"):
                cat.append(i)
            else:
                con.append(i)
        return cat,con

    def replacer(self,df):
        misscols=[]
        rowcount = df.shape[0]
        for i in df.columns:
            if(df[i].count() < rowcount):
                misscols.append(i)

        cat,con = self.catconsep(df[misscols])
        for i in cat:
            x = df[i].mode()[0]
            df[i]=df[i].fillna(x)

        for i in con:
            x = round(df[i].mean(),2)
            df[i]=df[i].fillna(x)


    def preprocessing(self,X):
        import pandas as pd
        cat = []
        con = []
        for i in X.columns:
            if(X[i].dtypes == "object"):
                cat.append(i)
            else:
                con.append(i)

        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
        X2 = pd.get_dummies(X[cat])
        X3 = X1.join(X2)
        return X3
    
    
class EDA:
       
    def univariate(self,df,cat,con):
        from warnings import filterwarnings
        filterwarnings("ignore")
        import seaborn as sb
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,40))
        x = 1
        u = int(df.shape[1]/3)+1
        for i in df.columns:
            if(df[i].dtypes == "object"):
                plt.subplot(u,3,x)
                sb.countplot(df[i])
                x = x + 1
            else:
                plt.subplot(u,3,x)
                sb.distplot(df[i])
                x = x + 1
                
        plt.show()
        
    def bivariate_ycat(self,df,y_df,cat,con):
        from warnings import filterwarnings
        filterwarnings("ignore")
        import seaborn as sb
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,40))
        x = 1
        u = int(df.shape[1]/3)+1
        y_col = y_df.columns
        y_fin = y_col[0]
        for i in df.columns:
            if(df[i].dtypes == "object"):
                plt.subplot(u,3,x)
                sb.countplot(y=df[i],hue=y_df[y_fin])
                x = x + 1
            else:
                plt.subplot(u,3,x)
                sb.boxplot(y_df[y_fin],df[i])
                x = x + 1
                
        plt.show()
        
    def bivariate_ycon(self,df,y_df,cat,con):
        from warnings import filterwarnings
        filterwarnings("ignore")
        import seaborn as sb
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,40))
        x = 1
        u = int(df.shape[1]/3)+1
        y_col = y_df.columns
        y_fin = y_col[0]
        for i in df.columns:
            if(df[i].dtypes == "object"):
                plt.subplot(u,3,x)
                sb.boxplot(df[i],y_df[y_fin])
                x = x + 1
            else:
                plt.subplot(u,3,x)
                sb.scatterplot(df[i],y_df[y_fin])
                x = x + 1
                
        plt.show()
    
    
class Data_split:
    def train_test(self,X,Y,rs):
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=rs)
        return xtrain,xtest,ytrain,ytest
    
    
class Model_maker:
    def maker_Logistic_model(self,xtrain,ytrain):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr_model = lr.fit(xtrain,ytrain)
        return lr_model
    
    def maker_naive_bayes_model(self,xtrain,ytrain):
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb_model = nb.fit(xtrain,ytrain)
        return nb_model
    
    def maker_dt_classifier_model(self,xtrain,ytrain,rs,mx_depth,mn_samples_leaf,mn_samples_split):
        from sklearn.tree import DecisionTreeClassifier
        dtr_model = DecisionTreeClassifier(random_state=rs,max_depth=mx_depth,min_samples_leaf=mn_samples_leaf,min_samples_split=mn_samples_split)
        model = dtr_model.fit(xtrain,ytrain)
        return model
    
    def maker_dt_regressor_model(self,xtrain,ytrain,rs,mx_depth,mn_samples_leaf,mn_samples_split):
        from sklearn.tree import DecisionTreeRegressor
        dtr1  = DecisionTreeRegressor(random_state=rs,max_depth=mx_depth,min_samples_leaf=mn_samples_leaf,min_samples_split=mn_samples_split)
        model_dtr1 = dtr1.fit(xtrain,ytrain)
        return model_dtr1
    
    def maker_linear_regressor_model(self,xtrain,ytrain):
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        model = lm.fit(xtrain,ytrain)
        return model
    
    def maker_knn_regressor_model(self,xtrain,ytrain,n):
        from sklearn.neighbors import KNeighborsRegressor
        knr = KNeighborsRegressor(n_neighbors=n)
        model = knr.fit(xtrain,ytrain)
        return model
    
    def maker_knn_classifier_model(self,xtrain,ytrain,n):
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier(n_neighbors=n)
        model = knc.fit(xtrain,ytrain)
        return model
    
    def maker_rand_for_reg_model(self,xtrain,ytrain,n_est,rs,max_dep):
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(n_estimators=n_est, random_state=rs, max_depth=max_dep)
        model = rfr.fit(xtrain,ytrain)
        return model
    
    def maker_rand_for_classi_model(self,xtrain,ytrain,n_est,rs,max_dep):
        from sklearn.ensemble import RandomForestClassifier
        rfr = RandomForestClassifier(n_estimators=n_est, random_state=rs, max_depth=max_dep)
        model = rfr.fit(xtrain,ytrain)
        return model
    
    def maker_Adb_reg_model(self,xtrain,ytrain,rs1,rs2,n_est,max_dep):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        boost = AdaBoostRegressor(DecisionTreeRegressor(random_state=rs1,max_depth=max_dep),n_estimators=n_est,random_state=rs2)
        model = boost.fit(xtrain,ytrain)
        return model
    
    def maker_Adb_classi_model(self,xtrain,ytrain,rs1,rs2,n_est,max_dep):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        boost = AdaBoostClassifier(DecisionTreeClassifier(random_state=rs1,max_depth=max_dep),n_estimators=n_est,random_state=rs2)
        model = boost.fit(xtrain,ytrain)
        return model
    
    
class Model_accuracy_measure:
    def regressor_accuracy(self,xtrain,ytrain,xtest,ytest,model):
        from sklearn.metrics import mean_absolute_error
        
        pred1 = model.predict(xtrain)
        err_tr_model = round(mean_absolute_error(ytrain,pred1),2)
        
        pred2 = model.predict(xtest)
        err_ts_model = round(mean_absolute_error(ytest,pred2),2)
        
        return err_tr_model,err_ts_model
    
    def classifier_acccuracy(self,xtrain,ytrain,xtest,ytest,model):
        from sklearn.metrics import accuracy_score
        
        pred1 = model.predict(xtrain)
        acc_tr_model = round(accuracy_score(ytrain,pred1),2)
        
        pred2 = model.predict(xtest)
        acc_ts_model = round(accuracy_score(ytest,pred2),2)
        
        return acc_tr_model,acc_ts_model