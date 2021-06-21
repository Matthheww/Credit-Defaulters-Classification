#!/usr/bin/env python
# coding: utf-8

# In[54]:


#all the required libraries are imported
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[55]:


#Class is defined 
class Credit_default_classification:
    
#first method is called as data to read the data and store it to a dataframe
    def data(self):
        global df
        df = pd.read_excel('C:\\Users\\kkuri\\Downloads\\Default of credit card clients.xls')
        #data is preprocessed
        headers = df.iloc[0]
        df = df[1:]
        df = df.rename(columns = headers)
        df =df.rename(columns = {'default payment next month': 'Output', 'PAY_0' : 'PAY_1'})
        #undefined variables are replaced with the mode of the variable to remove outliers
        for i in ('PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'):
            df[i] = df[i].replace([-2],-1)
        #Creating dummy variables for all the categorical variables
        categorical_columns = ['SEX','EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5', 'PAY_6']
        df = pd.get_dummies(df, columns = categorical_columns)
        #dropping unknown parameters
        df = df.drop(['MARRIAGE_0','EDUCATION_5','EDUCATION_6'], axis = 1)
        return df
    
    #defining method to split the data between target variable and the feature columns
    def feature_column_definition(self):
        global x,y
        x = df
        x = x.drop(['Output', 'ID'], axis = 1)
        y = df.Output
        y = y.astype(int)
        
    #splitting the dataset to training and validation sets for modelling in a 70:30 ratio
    def data_splitting(self, array_x, array_y):
        global X_train, X_test, Y_train, Y_test
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=1)
        return 'Data is splitted to 70% training and 30% testing subsets'
    
    #plotting distribution of defaulters and non-defaulters in the data
    def output_distribution(self):
        output = df['Output']
        output = output.to_numpy()
        Defaulters = 0
        Non_defaulters = 0
        for i in output:
            if (i==0):
                Defaulters += 1
            else:
                Non_defaulters += 1
        objects = ['Defaulters', 'Non Defaulters']
        frequency = [Defaulters,Non_defaulters]
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, frequency, align='center', alpha=0.3, color = 'blue')
        plt.xticks(y_pos, objects)
        plt.ylabel('Number of customers')
        plt.title('Status of customers')
        plt.show()
    
   #plotting feature importance to identify important parameters in the model     
    def decision_tree_feature_importance_plot(self):
        random = pd.DataFrame({'Feature': x.columns, 'Feature importance': clf.feature_importances_})
        random = random.sort_values(by='Feature importance',ascending=False)
        top30 = random.head(30)
        plt.figure(figsize = (15,5))
        plt.title('Features importance',fontsize=14)
        s = sea.barplot(x='Feature',y='Feature importance',data=top30)
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
        plt.show()
        
    #plot to identify data distribution based on 'marriage' variable
    def Marrige_pie_chart(self):
        Married = df['MARRIAGE_1'].sum()
        Single = df['MARRIAGE_2'].sum()
        Others = df['MARRIAGE_3'].sum()
        data=[Married,Single,Others]
        my_labels = 'Married','Single','Others'
        my_explode = (0.2, 0, 0)
        my_colors = ['orange','grey','green']
        plt.pie(data, labels=my_labels, autopct='%1.1f%%', startangle=15,colors=my_colors, shadow = True, explode=my_explode,radius=1.5)
        plt.title('Maritial Status ',y=1.25)
        plt.show()

    #plot to identify data distribution based on 'age' variable 
    def Age_frequency_plot(self):
        Agelist=df['AGE']
        plt.figure(figsize = (10,5))
        plt.hist(Agelist, bins=50,histtype='barstacked',color='mediumseagreen',rwidth = 0.9)
        plt.gca().set(title='Age Distribution',xlabel='Age', ylabel='Frequency')
        plt.show()
        
    #plot to identify data distribution based on 'sex' variable    
    def Gender_pie_chart(self):
        Male = df['SEX_1'].sum()
        Female = df['SEX_2'].sum()
        data_gender=[Male,Female]
        my_labels = 'Male','Female'
        my_explode = (0.2, 0, )
        my_colors = ['springgreen','lightcoral']
        plt.pie(data_gender, labels=my_labels, autopct='%1.1f%%', startangle=15,colors=my_colors, shadow = True, explode=my_explode,radius=1.5)
        plt.title('Gender Based Distribution ',y=1.25)
        centre_circle = plt.Circle((0,0),0.75,color='white', fc='white',linewidth=1.25)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.show()
        
    #plot to identify data distribution based on 'Education' variable    
    def Education_plot(self):
        School = df['EDUCATION_1'].sum()
        University = df['EDUCATION_2'].sum()
        High_School = df['EDUCATION_3'].sum()
        Others = df['EDUCATION_4'].sum()
        objects = ['Graduate School', 'University', 'High School', 'Others']
        frequency = [School, University, High_School, Others]
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, frequency, align='center', alpha=0.8, color = 'mediumslateblue')
        plt.xticks(y_pos, objects)
        plt.ylabel('Number of customers')
        plt.title('Education of the customers')
        plt.show()
      
    #defining method to implement logistic regression learning algorithm and return the accuracy
    def logistic_regression(self):
        #defining global variable
        global logistic_regr
        #calling the global variable and defining the logistic regression algorithm
        logistic_regr= LogisticRegression()
        #fit the model with training data
        logistic_regr.fit(X_train,Y_train)
        #predict the response of the test dataset
        Y_pred=logistic_regr.predict(X_test)
        return(f'Accuracy of logistic regression model is {metrics.accuracy_score(Y_test, Y_pred)}')
   
    #all the machine learning algorithms follow the above structure
    
    #defining method to implement support vector machine learning algorithm and return the accuracy
    def support_vector_machine(self):
        global svr
        svr = svm.SVC(kernel="linear", C =1e3, gamma = 0.1)
        svr.fit(X_train,Y_train)
        pred = svr.predict(X_test)
        return(f'Accuracy of svm model is{metrics.accuracy_score(Y_test, pred)}')
    
    #defining method to implement random forest machine learning algorithm and return the accuracy
    def random_forest(self):
        global ran
        ran = RandomForestClassifier(n_estimators=100)
        ran.fit(X_train,Y_train)
        y_pred=ran.predict(X_test)
        return(f'Accuracy of random forest model is {metrics.accuracy_score(Y_test, y_pred)}')
    
    #defining method to implement neural network machine learning algorithm and return the accuracy
    def neural_network(self):
        global mlp
        mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
        mlp.fit(X_train,Y_train)
        Y_pred = mlp.predict(X_test)
        return(f'Accuracy of neural network model is {metrics.accuracy_score(Y_test, Y_pred)}')
    
    #defining method to implement decision tree machine learning algorithm and return the accuracy
    def decision_tree(self):
        global clf
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,Y_train)
        Y_pred = clf.predict(X_test)
        return(f'Accuracy of decision tree model is {metrics.accuracy_score(Y_test, Y_pred)}')
    
    #defining method to prune the decision tree and return the accuracy
    def decision_tree_trim(self):
        global clf_t
        # Create classifier object for trimming the decision tree
        clf_t = DecisionTreeClassifier(criterion="entropy", max_depth=7)
        clf_t = clf_t.fit(X_train,Y_train)
        Y_pred = clf_t.predict(X_test)
        #classification report is generated and the confusion matrix is created for further analysis
        print("Classification Report of the Decision Tree model is \n\n\n",classification_report(Y_test,Y_pred))
        confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])
        sea.heatmap(confusion_matrix, annot=True)
        return(f'Decision tree trim model is the best model with accuracy {metrics.accuracy_score(Y_test, Y_pred)}')
    
    #Method to generate ROC curve to analyse the best model
    def ROC_Graph_decisiontree(self):
        y_score = clf_t.predict_proba(X_test)[:,1]
        false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, y_score)
        plt.subplots(1, figsize=(8,8))
        plt.title('DecisionTree - Pruned')
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".8"), plt.plot([1, 1] , c=".8")
        plt.ylabel('Rate of True Positive')
        plt.xlabel('Rate of False Positive')
        plt.show()
        
    #method to plot the decision tree    
    def decision_tree_plot(self):
        plt.figure(figsize=(50,20))
        tree.plot_tree(clf_t,max_depth=3,label='all',filled=True)
        plt.show()

    #method is called to predict the output of 5 created profiles    
    def prediction(self):
        global score_df
        score_df = pd.read_excel('C:\\Users\\kkuri\\Downloads\\Credit Testing Data.xlsx')
        original_data = score_df
        categorical_columns = ['SEX','EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5', 'PAY_6']
        score_df = pd.get_dummies(score_df, columns = categorical_columns)
        missing_columns = set( x.columns ) - set( score_df.columns )
        for i in missing_columns:
            score_df[i] = 0
        score_df = score_df[score_df.columns]
        score_df = score_df.drop(['ID', 'Output'], axis = 1)
        column_headers = score_df.columns
        score_df = score_df.to_numpy()
        Output = []
        c = 0
        for j in score_df:
            d = clf_t.predict(score_df[[c]])
            c+=1
            Output.append(d)
        Output = pd.DataFrame(Output, columns=['Output'])
        original_data['Output'] = Output
        original_data.to_excel('C:\\Users\\kkuri\\Downloads\\Output.xlsx')
        print('The file is saved to Downloads')


# In[52]:


credit = Credit_default_classification()


# In[7]:


credit.data()


# In[8]:


df.describe()


# In[9]:


credit.feature_column_definition()


# In[10]:


credit.Marrige_pie_chart()


# In[11]:


credit.Age_frequency_plot()


# In[12]:


credit.Gender_pie_chart()


# In[13]:


credit.Education_plot()


# In[53]:


credit.output_distribution()


# In[14]:


credit.data_splitting(x,y)


# In[13]:


credit.random_forest()


# In[18]:


credit.logistic_regression()


# In[15]:


credit.neural_network()


# In[293]:


credit.support_vector_machine()


# In[16]:


credit.decision_tree()


# In[17]:


credit.decision_tree_trim()


# In[19]:


credit.decision_tree_plot()


# In[20]:


credit.ROC_Graph_decisiontree()


# In[21]:


credit.decision_tree_feature_importance_plot()


# In[23]:


credit.prediction()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




