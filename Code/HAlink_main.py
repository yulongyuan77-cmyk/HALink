import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code.scGNN import GENELink  
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from Code.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from Code.PytorchTools  import EarlyStopping
import numpy as np
import random
import glob
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[6,6], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=42, help='Random seed')  
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


seed = 42  
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class UnsupervisedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_genes):
        super(UnsupervisedVAE, self).__init__()
        self.num_genes = num_genes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_genes * num_genes)  

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        adj_matrix = self.fc4(h3)  
        adj_matrix = adj_matrix.view(-1, self.num_genes, self.num_genes)  
        adj_matrix = (adj_matrix - adj_matrix.mean()) / (adj_matrix.std() + 1e-6)
        adj_matrix = torch.sigmoid(adj_matrix)
        return adj_matrix

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_adj, mu, logvar, lambda_l1=0.001):
        similarity_matrix = torch.mm(mu, mu.T)
        loss_self_supervised = F.mse_loss(recon_adj, similarity_matrix.unsqueeze(0))
        KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_self_supervised + KL_div

     
def top_k_sparsify(matrix, k):
    new_matrix = np.zeros_like(matrix)  
    for i in range(matrix.shape[0]):
        top_k_indices = np.argpartition(matrix[i], -k)[-k:]
        new_matrix[i, top_k_indices] = matrix[i, top_k_indices]  
    return new_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Load gene expression matrix
path = '/......./' # your path
exp_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/BL--ExpressionData.csv'
data_input = pd.read_csv(exp_file, index_col=0)  

feature = data_input.values.astype(float)

# Normalized to [0,1]
if feature.min() < 0 or feature.max() > 1:
    feature = (feature - feature.min()) / (feature.max() - feature.min())

feature = torch.tensor(feature, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature = feature.to(device)

print(f"Feature shape: {feature.shape}")  # (num_genes, num_samples)

num_genes = 821 #Number of genes in the gene expression matrix
input_dim = feature.size()[1]  
hidden_dim = 256  
latent_dim = 128  
epochs = 200
learning_rate = 0.001

vae_model = UnsupervisedVAE(input_dim, hidden_dim, latent_dim, num_genes).to(device)
optimizer_vae = optim.Adam(vae_model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    vae_model.train()
    optimizer_vae.zero_grad()
    recon_adj, mu, logvar = vae_model(feature)
    loss = vae_model.loss_function(recon_adj, mu, logvar)
    loss.backward()
    optimizer_vae.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    torch.cuda.empty_cache() 

data_input = pd.read_csv(exp_file, index_col=0)  
gene_names = data_input.index.values  

vae_model.eval()
with torch.no_grad():
    recon_adj, _, _ = vae_model(feature)

adj_matrix = recon_adj[0].cpu().numpy()  

# Apply Top-K Sparsity
adj_matrix = top_k_sparsify(adj_matrix, k=45)

sparse_csv_path = f"{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/sparse_vae_adj_matrix.csv"
sparse_adj_df = pd.DataFrame(adj_matrix, index=gene_names, columns=gene_names)
sparse_adj_df.to_csv(sparse_csv_path)


threshold = 0.5
adj_matrix = (adj_matrix > threshold).astype(float)

adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

if len(adj_matrix.shape) == 3:
    adj_matrix = adj_matrix[0]  

adj_matrix_df = pd.DataFrame(adj_matrix.cpu().numpy(), index=gene_names, columns=gene_names)

save_path_npy = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/vae_generated_adj_with_names.npy'
np.save(save_path_npy, adj_matrix_df.values)

save_path_csv = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/vae_generated_adj_with_names.csv'
adj_matrix_df.to_csv(save_path_csv)


vae_adj_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/vae_generated_adj_with_names.csv', index_col=0)

label_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Train_set.csv')  


tf_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/TF.csv') 
target_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Target.csv')  

tf_df_cleaned = tf_df[['TF', 'index']]  
target_df_cleaned = target_df[['Gene', 'index']]  

tf_df = tf_df_cleaned
target_df = target_df_cleaned

tf_genes = tf_df['TF'].values  
target_genes = target_df['Gene'].values 

for _, row in label_df.iterrows():
    if row['Label'] == 1:
        tf_idx = row['TF']
        target_idx = row['Target']

        tf_gene_name = tf_df.loc[tf_df['index'] == tf_idx, 'TF'].values[0]  
        target_gene_name = target_genes[target_idx]  
        vae_adj_df.loc[tf_gene_name, target_gene_name] = 1  


vae_adj_df.to_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_vae_adj_with_names.csv')

vae_adj_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/vae_generated_adj_with_names.csv'
vae_adj_df = pd.read_csv(vae_adj_file, index_col=0)

vae_positive_edges = []
for tf_gene in vae_adj_df.index:
    for target_gene in vae_adj_df.columns:
        if vae_adj_df.loc[tf_gene, target_gene] == 1:
            vae_positive_edges.append([tf_gene, target_gene])

vae_positive_edges_count = len(vae_positive_edges)
total_edges = vae_adj_df.size - vae_adj_df.shape[0]  
positive_edge_ratio = vae_positive_edges_count / total_edges




# Load TF and Target
tf_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/TF.csv')  
target_df = pd.read_csv(f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Target.csv')  

tf_gene_to_index = {gene: idx for gene, idx in zip(tf_df['TF'], tf_df['index'])}
target_gene_to_index = {gene: idx for gene, idx in zip(target_df['Gene'], target_df['index'])}
existing_tf_indexes = set(tf_gene_to_index.values())
existing_target_indexes = set(target_gene_to_index.values())
unknown_genes_tf = {tf_gene for tf_gene, target_gene in vae_positive_edges if tf_gene not in tf_gene_to_index}
unknown_genes_target = {target_gene for tf_gene, target_gene in vae_positive_edges if target_gene not in target_gene_to_index}

new_index_tf = max(existing_tf_indexes) + 1
new_index_target = max(existing_target_indexes) + 1

for tf_gene in unknown_genes_tf:
    while new_index_tf in existing_tf_indexes or new_index_tf in existing_target_indexes:
        new_index_tf += 1  
    tf_gene_to_index[tf_gene] = new_index_tf
    existing_tf_indexes.add(new_index_tf)  
    new_index_tf += 1

for target_gene in unknown_genes_target:
    while new_index_target in existing_target_indexes or new_index_target in existing_tf_indexes:
        new_index_target += 1  
    target_gene_to_index[target_gene] = new_index_target
    existing_target_indexes.add(new_index_target)  
    new_index_target += 1


#  map_gene_to_index 
def map_gene_to_index(vae_edges, tf_gene_to_index, target_gene_to_index, num_genes=560):
    mapped_edges = []
    new_index_tf = max(tf_gene_to_index.values()) + 1 if tf_gene_to_index else num_genes
    new_index_target = max(target_gene_to_index.values()) + 1 if target_gene_to_index else num_genes

    for tf_gene, target_gene in vae_edges:
        tf_index = tf_gene_to_index.get(tf_gene, new_index_tf)
        target_index = target_gene_to_index.get(target_gene, new_index_target)

        if tf_index >= num_genes:
            tf_index = tf_index % num_genes  
        if target_index >= num_genes:
            target_index = target_index % num_genes  

        if tf_index == new_index_tf:
            tf_gene_to_index[tf_gene] = new_index_tf
            new_index_tf += 1
        if target_index == new_index_target:
            target_gene_to_index[target_gene] = new_index_target
            new_index_target += 1
        mapped_edges.append((tf_index, target_index))

    return mapped_edges

vae_positive_edges = map_gene_to_index(vae_positive_edges, tf_gene_to_index, target_gene_to_index)

from collections import Counter
vae_positive_edges_count = Counter(tuple(edge) for edge in vae_positive_edges)

# Load the training, validation, and test sets
train_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Train_set.csv'
val_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Validation_set.csv'
test_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Test_set.csv'

train_df = pd.read_csv(train_file, index_col=0)
val_df = pd.read_csv(val_file, index_col=0)
test_df = pd.read_csv(test_file, index_col=0)

train_pos_edges = set(zip(train_df[train_df['Label'] == 1]['TF'], train_df[train_df['Label'] == 1]['Target']))
val_pos_edges = set(zip(val_df[val_df['Label'] == 1]['TF'], val_df[val_df['Label'] == 1]['Target']))
test_pos_edges = set(zip(test_df[test_df['Label'] == 1]['TF'], test_df[test_df['Label'] == 1]['Target']))

existing_edges = train_pos_edges | val_pos_edges | test_pos_edges
common_edges = set(vae_positive_edges) & existing_edges

common_edges = set(vae_positive_edges) & existing_edges
vae_new_edges = list(set(vae_positive_edges) - common_edges)

vae_train, temp = train_test_split(vae_new_edges, test_size=0.4, random_state=42)
vae_val, vae_test = train_test_split(temp, test_size=0.5, random_state=42)

vae_train_df = pd.DataFrame(vae_train, columns=['TF', 'Target'])
vae_train_df['Label'] = 1
vae_val_df = pd.DataFrame(vae_val, columns=['TF', 'Target'])
vae_val_df['Label'] = 1
vae_test_df = pd.DataFrame(vae_test, columns=['TF', 'Target'])
vae_test_df['Label'] = 1

train_df = pd.concat([train_df, vae_train_df], ignore_index=True)
val_df = pd.concat([val_df, vae_val_df], ignore_index=True)
test_df = pd.concat([test_df, vae_test_df], ignore_index=True)

def add_index_column(df):
    df.insert(0, 'Index', range(len(df)))  
    return df

train_df = add_index_column(train_df)
val_df = add_index_column(val_df)
test_df = add_index_column(test_df)

new_train_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_train_set.csv'
new_val_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_val_set.csv'
new_test_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_test_set.csv'

train_df.to_csv(new_train_file, index=False)
val_df.to_csv(new_val_file, index=False)
test_df.to_csv(new_test_file, index=False)

def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

exp_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/BL--ExpressionData.csv'
tf_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/TF.csv'
target_file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Target.csv'

train__file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_train_set.csv'
test__file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_test_set.csv'
val__file = f'{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_val_set.csv'

tf_embed_path = rf'{path}/HALink-main/Code/Result/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Channel1.csv'
target_embed_path = rf'{path}/HALink-main/Code/Result/Benchmark Dataset/Specific Dataset/mDC/TFs+500/Channel2.csv'


data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)


train_data = pd.read_csv(train__file, index_col=0).values
validation_data = pd.read_csv(val__file, index_col=0).values
test_data = pd.read_csv(test__file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

vae_adj_file = f"{path}/HALink-main/Dataset/Benchmark Dataset/Specific Dataset/mDC/TFs+500/updated_vae_adj_with_names.csv"

vae_adj_df = pd.read_csv(vae_adj_file, index_col=0)
vae_adj_matrix = vae_adj_df.values

sparse_adj = torch.tensor(vae_adj_matrix, dtype=torch.float32)

train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

model = GENELink(input_dim=feature.size()[1],
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                )

sparse_adj = sparse_adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)


optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = f'{path}/HALink-main/Code/model/Benchmark Dataset/Specific Dataset/mDC/TFs+500'
if not os.path.exists(model_path):
    os.makedirs(model_path)



#early_stopping = EarlyStopping(save_dir=model_path+'trained_model', patience=10, verbose=True)

print("train Hybrid Framework model")
for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        pred = model(data_feature, sparse_adj, train_x)

        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)

        loss_BCE.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss_BCE.item()

    model.eval()
    score = model(data_feature, sparse_adj, validation_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
        #
    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))

    #early_stopping(AUC, model) 

    #if early_stopping.early_stop:
        #print("Early stopping triggered, stopping training.")
        #break

torch.save(model.state_dict(), model_path +'trained_model.pkl')

model.load_state_dict(torch.load(model_path +'trained_model.pkl',weights_only=True))
model.eval()
tf_embed, target_embed = model.get_embedding()
embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

score = model(data_feature, sparse_adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)

print('test AUC:{}'.format(AUC),
     'test AUPRC:{}'.format(AUPR))
























