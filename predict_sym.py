import json
with open('dieseas_short.json','r')as f:
    intents=json.load(f)
print(intents)

all_words=[]
tags=[]
xy=[]

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch #creating deep neural networks
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

#Initialize Variables and Preprocess Data
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
print(xy)

xy_test = [
    (['ca', "n't", 'think', 'straight'], 'altered_sensorium'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['bloody', 'poop'], 'bloody_stool'),
    (['blurred', 'vision'], 'blurred_and_distorted_vision'),
    (['ca', "n't", 'breathe'], 'breathlessness'),
    (['Yellow', 'liquid', 'pimple'], 'yellow_crust_ooze'),
    (['lost', 'weight'], 'weight_loss'),
    (['side', 'weaker'], 'weakness_of_one_body_side'),
    (['watering', 'eyes'], 'watering_from_eyes'),
    (['brief', 'blindness'], 'visual_disturbances'),
    (['throat', 'hurts'], 'throat_irritation'),
    (['extremities', 'swelling'], 'swollen_extremeties'),
    (['swollen', 'lymph', 'nodes'], 'swelled_lymph_nodes'),
    (['dark', 'under', 'eyes'], 'sunken_eyes'),
    (['stomach', 'blood'], 'stomach_bleeding'),
    (['blood', 'urine'], 'spotting_urination'),
    (['sinuses', 'hurt'], 'sinus_pressure'),
    (['watery', 'from', 'nose'], 'runny_nose'),
    (['have', 'to', 'move'], 'restlessness'),
    (['red', 'patches', 'body'], 'red_spots_over_body'),
    (['sneeze'], 'continuous_sneezing'),
    (['coughing'], 'cough'),
    (['skin', 'patches'], 'dischromic_patches'),
    (['skin', 'bruised'], 'bruising'),
    (['burning', 'pee'], 'burning_micturition'),
    (['hurts', 'pee'], 'burning_micturition'),
    (['Burning', 'sensation'], 'burning_micturition'),
    (['chest', 'pressure'], 'chest_pain'),
    (['pain', 'butt'], 'pain_in_anal_region'),
    (['heart', 'bad', 'beat'], 'palpitations'),
    (['fart', 'lot'], 'passage_of_gases'),
    (['cough', 'phlegm'], 'phlegm'),
    (['lot', 'urine'], 'polyuria'),
    (['Veins', 'bigger'], 'prominent_veins_on_calf'),
    (['Veins', 'emphasized'], 'prominent_veins_on_calf'),
    (['yellow', 'pimples'], 'pus_filled_pimples'),
    (['red', 'nose'], 'red_sore_around_nose'),
    (['skin', 'yellow'], 'yellowish_skin'),
    (['eyes', 'yellow'], 'yellowing_of_eyes'),
    (['large', 'thyroid'], 'enlarged_thyroid'),
    (['really', 'hunger'], 'excessive_hunger'),
    (['always', 'hungry'], 'excessive_hunger')
]
print(len(xy_test))

lemmatizer = WordNetLemmatizer()
all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if (w not in set(stopwords.words('english')) and w.isalpha())]
all_words = sorted(set(all_words))
print(all_words)

tags=sorted(set(tags))
print(tags)

def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[lemmatizer.lemmatize(w.lower())for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

#converting text into numerical features
x_train = []
y_train = []
for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train)
print(y_train)

#testing numerical data     
x_test=[]
y_test=[]
for (pattern, tag) in xy_test:
    bag = bag_of_words(pattern, all_words)
    x_test.append(bag)
    label = tags.index(tag)
    y_test.append(label) 
x_test = np.array(x_test)
y_test = np.array(y_test)
print(y_test)


import torch
import torch.nn as nn
class ChatDataset(Dataset):#PyTorch's DataLoader for training neural networks
    def __init__(self,x_data,y_data):
        self.n_samples=len(x_data)
        self.x_data=x_data#This represents the input features
        self.y_data=y_data#This represents the target labels
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples
                    
import torch.nn as nn
class NeuralNet(nn.Module):#The NeuralNet class, as defined, is a simple feedforward neural network with two hidden layers 
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)#The number of neurons in the hidden layers.
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,num_classes)# The number of output classes 
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.l1(x)#The first linear layer that maps input_size to hidden_size.
        out=self.relu(out)
        out=self.l2(out)#The second linear layer that maps hidden_size to hidden_size.
        out=self.relu(out)
        out=self.l3(out)#The third linear layer that maps hidden_size to num_classes.
        return out
#x=The input tensor to the network.


from sklearn.metrics import accuracy_score
batch_size=8
hidden_size=8
output_size=len(tags)
input_size=len(all_words)
learning_rates=[0.01,0.05,0.1,0.15]
num_epochs=1000

import torch
from torch import nn
from torch.utils.data  import DataLoader, Dataset
from sklearn.metrics import accuracy_score

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rates = [0.01, 0.05, 0.1, 0.15]
num_epochs = 1000

from sklearn.metrics import accuracy_score

batch_size = 8
hidden_size = 8
output_size = len(tags) 
input_size = len(all_words)
learning_rates = [0.01, 0.05, 0.1, 0.15]
num_epochs = 1000

def nn_validation():
    dataset_train = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device('cpu') 
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    loss_train = []
    loss_test = []
    for lr in learning_rates:
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
        print(f"lr: {lr}, train")
        for epoch in range(num_epochs):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % (num_epochs / 2) == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')
        print(f'final loss = {loss.item():.4f}')
        loss_train.append(loss.item())
        y_predicted = []
            
        for x in x_test:
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x)
            output = model(x)
            _, predicted = torch.max(output, dim=1)
            y_pred = predicted.item()
            y_predicted.append(y_pred)
        print("y_predicted:", y_predicted)
        y_predicted = np.array(y_predicted)
        loss_test.append(accuracy_score(y_test, y_predicted))
        print()
    return loss_train, loss_test
train_errors, test_errors = nn_validation()

import matplotlib.pyplot as plt
fig=plt.figure()
plt.title("train and test errors for differnet learning rates")
plt.plot(learning_rates,train_errors,c='purple',label='training error')
plt.plot(learning_rates,test_errors,c='orange',label='testing error')
plt.legend()
plt.show()
    
batch_size=8
hidden_size=8
output_size=len(tags)
input_size=len(all_words)
learning_rates=0.01
num_epochs=1000
dataset_train=ChatDataset(x_train, y_train)
train_loader=DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=2)
device=torch.device('cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.ASGD(model.parameters(),lr=learning_rates)
import multiprocessing
for epoch in range(num_epochs):
    if __name__=='__main__':
        for (words,labels)in train_loader:
            words=words.to(device)
            labels=labels.to(device)
            output=model(words)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if(epoch + 1)%(num_epochs/10)==0:
        print(f'epoch{epoch + 1}/{num_epochs},{loss.item():.4f}')
    print(f'final loss={loss.item():.4f}')
    data={
        "model_state":model.state_dict(),
        "input_size":input_size,
        "output_size":output_size,
        "hidden_size":hidden_size,
        "all_words":all_words,
        "tags":tags
    }
    file="data.json"
    torch.save(data,file)
    
    a=torch.load("data.json")
    device=torch.device('cpu')
    model=NeuralNet(a['input_size'],a['hidden_size'],a['output_size']).to(device)
    model.load_state_dict(a['model_state'])
    model.eval()

    all_words=a['all_words']
    tags=a['tags']
    
    sentence="i have a pain in my head"
    sentence=nltk.word_tokenize(sentence)
    X=bag_of_words(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)
    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    print("prob",prob)
    print(tag)

