#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import os
import pandas as pd
import numpy as np


# In[2]:


dataset=[]
folder_paths=['/Users/hp/Desktop/internship/gender/Train/female',
              '/Users/hp/Desktop/internship/gender/Train/male']
for i in folder_paths:
    folder_name=os.path.basename(i)
    for file_name in os.listdir(i):
        img_path=os.path.join(i,file_name)
        if os.path.isfile(img_path):
            img=cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                resize_img=cv.resize(img,(150,150))
                flattened_img=resize_img.flatten().tolist()
                dataset.append(flattened_img+[folder_name])



# In[3]:


df = pd.DataFrame(dataset)


# In[4]:


df.rename(columns={df.iloc[:,-1].name:'Target'},inplace=True)


# In[5]:


df.head()


# **Randomize the Data**

# In[6]:


#get num of rows of dataset
num_rows=len(df)
#generate permutated indices
permuted_indices=np.random.permutation(num_rows)
#generate random data
random_df=df.iloc[permuted_indices]


# **Encoding the label**

# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


x=random_df.drop('Target',axis=1)
x=x/255
x.head()


# In[9]:


encoder=LabelEncoder()
y=random_df.Target
y_encoded=encoder.fit_transform(y)
y_series=pd.Series(y_encoded,name='target')


# In[10]:


df_encoded=pd.concat([x,y_series],axis=1)
df_encoded.head()


# ## **Classification Using SVM **

# In[11]:


from sklearn.svm import SVC
model_svc=SVC()


# In[12]:


from sklearn.model_selection import train_test_split
np.random.seed(42)
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42)


# **Fitting the Model On Training data**

# In[13]:


model_svc.fit(x_train,y_train);


# **Model Evaluation**

# In[49]:


y_pred_svc=model_svc.predict(x_test)


# In[15]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns


# In[51]:


Accuracy_svc=accuracy_score(y_pred_svc,y_test)
print('Accuracy:',Accuracy_svc)
CR=classification_report(y_pred_svc,y_test)
print('Classification Report\n',CR)
cm=confusion_matrix(y_pred_svc,y_test)
sns.heatmap(cm,annot=True)


# **Deployment**

# In[20]:


import matplotlib.pyplot as plt


# In[21]:


image_path = "/Users/hp/Desktop/internship/image_check.jpg"
user_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Resize the image to match the input size expected by the model
resized_image = cv.resize(user_image, (150, 150))

# Flatten the image
flattened_img = resized_image.flatten()

# Normalize the flattened image data
normalized_user_image = flattened_img / 255.0

# Convert the normalized flattened image to a NumPy array and reshape it
user_input = normalized_user_image.reshape(1, -1)

# Make a prediction using the trained model
user_prediction = model_svc.predict(user_input)
image=cv.cvtColor(resized_image,cv.COLOR_BGR2RGB)
# Decode the predicted label
predicted_class = encoder.inverse_transform(user_prediction)[0]
plt.imshow(image)
plt.title(predicted_class)


# # **Classification using Logistic Regression**

# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


model_log=LogisticRegression()


# In[24]:


model_log.fit(x_train,y_train);


# **Model Evaluation**

# In[52]:


y_pred_log=model_log.predict(x_test)


# In[53]:


accuracy_log=accuracy_score(y_pred_log,y_test)
print("Accuracy Score is :",accuracy_log)
C_report=classification_report(y_pred_log,y_test)
print('Classification report:',C_report)
cm=confusion_matrix(y_pred_log,y_test)
sns.heatmap(cm,annot=True,cmap='Blues')


# **Testing our model by giving a random input image**

# In[54]:


image_path="/Users/hp/Desktop/internship/image_check.jpg"
#read input image
img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
#resize input image
resized_img=cv.resize(img,(150,150))
#flatten input image
flattened_img=resized_img.flatten()
#normalizing image pixels
normalized_img=flattened_img/255.0
#reshape user image
reshaped_img=normalized_img.reshape(1,-1)
#make prediction
prediction=model_log.predict(reshaped_img)
#decode the input class
decoded_prediction=encoder.inverse_transform(prediction)[0]
image = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(decoded_prediction)


# # **Classification Using Random Forest **

# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


Rf_model=RandomForestClassifier(n_estimators=100)


# In[30]:


Rf_model.fit(x_train,y_train)


# In[55]:


y_pred_rf=Rf_model.predict(x_test)


# In[56]:


accuracy_rf=accuracy_score(y_pred_rf,y_test)
print("Accuracy Score is :",accuracy_rf)
C_report=classification_report(y_pred_rf,y_test)
print('Classification report:',C_report)
cm=confusion_matrix(y_pred_rf,y_test)
sns.heatmap(cm,annot=True,cmap='Blues')


#  **Testing our model by giving a random input image**
# 
# 

# In[57]:


image_path="/Users/hp/Desktop/internship/image_check.jpg"
#read input image
img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
#resize input image
resized_img=cv.resize(img,(150,150))
#flatten input image
flattened_img=resized_img.flatten()
#normalizing image pixels
normalized_img=flattened_img/255.0
#reshape user image
reshaped_img=normalized_img.reshape(1,-1)
#make prediction
prediction=Rf_model.predict(reshaped_img)
#decode the input class
decoded_prediction=encoder.inverse_transform(prediction)[0]
image = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(decoded_prediction)


# **Saving model**

# ## **Classification Using Decision Tree**

# In[35]:


from sklearn.tree import DecisionTreeClassifier ,plot_tree


# In[36]:


model_dt=DecisionTreeClassifier(splitter='random')


# In[37]:


model_dt.fit(x_train,y_train)


# In[59]:


y_pred_dt=model_dt.predict(x_test)


# In[60]:


accuracy_dt=accuracy_score(y_pred_dt,y_test)
print("Accuracy Score is :",accuracy_dt)
C_report=classification_report(y_pred_dt,y_test)
print('Classification report:',C_report)
cm=confusion_matrix(y_pred_dt,y_test)
sns.heatmap(cm,annot=True,cmap='Blues')


# **Testing our model by giving a random input image**

# In[61]:


image_path="/Users/hp/Desktop/internship/image_check.jpg"
#read input image
img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
#resize input image
resized_img=cv.resize(img,(150,150))
#flatten input image
flattened_img=resized_img.flatten()
#normalizing image pixels
normalized_img=flattened_img/255.0
#reshape user image
reshaped_img=normalized_img.reshape(1,-1)
#make prediction
prediction=model_dt.predict(reshaped_img)
#decode the input class
decoded_prediction=encoder.inverse_transform(prediction)[0]
image = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(decoded_prediction)


# **Classification Using KNN**

# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[42]:


model_KNN=KNeighborsClassifier()


# In[43]:


model_KNN.fit(x_train,y_train)


# In[46]:


y_pred_knn=model_KNN.predict(np.array(x_test))


# In[62]:


accuracy_knn=accuracy_score(y_pred_knn,y_test)
print("Accuracy Score is :",accuracy_knn)
C_report=classification_report(y_pred_knn,y_test)
print('Classification report:',C_report)
cm=confusion_matrix(y_pred_knn,y_test)
sns.heatmap(cm,annot=True,cmap='Blues')


# **Testing our model by giving a random input image**

# In[63]:


image_path="/Users/hp/Desktop/internship/image_check.jpg"
#read input image
img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)
#resize input image
resized_img=cv.resize(img,(150,150))
#flatten input image
flattened_img=resized_img.flatten()
#normalizing image pixels
normalized_img=flattened_img/255.0
#reshape user image
reshaped_img=normalized_img.reshape(1,-1)
#make prediction
prediction=model_KNN.predict(reshaped_img)
#decode the input class
decoded_prediction=encoder.inverse_transform(prediction)[0]
image = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(decoded_prediction)


# **Comparison of Different models performance**

# In[67]:


import matplotlib.pyplot as plt

algorithms = ['SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Logistic Regression']
accuracy = [Accuracy_svc, accuracy_knn, accuracy_dt, accuracy_rf, accuracy_log]

colors = ['red', 'pink', 'blue', 'brown', 'black']
bar_width = 0.5  # Width of the bars
bar_positions = range(len(algorithms))

plt.bar(bar_positions, accuracy, color=colors, width=bar_width)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Machine Learning Algorithms')
plt.ylim(0, 1)  # Set y-axis limits between 0 and 1

# Annotate each bar with its accuracy value
for i, acc in enumerate(accuracy):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')

# Adjust x-axis labels and positions
plt.xticks(bar_positions, algorithms, rotation=15, ha='right')

plt.tight_layout()  # To prevent labels from being cut off
plt.show()


# ****

# # **Conclusion**

# **In conclusion the SVm was the best at making predictions with a  accuracy of 84%. The random forest and Logistic Regression models did well too, with accuracies of 81% for both. KNN was decent with 74% accuracy, but the Decision Tree model didn't perform as well, only achieving 69%. This information helps us choose the right model for different jobs where we need predictions. also the machine learning approach is very time taking. Logistic regression, Random Forest and Decision tree give wrong prediction while deploying this is due to these models underfit **

# In[ ]:




