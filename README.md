## A VAE-Based and Hybrid GCN–GAT Architecture for Inferring Single-Cell Gene Regulatory Networks


## Dependencies

- Python == 3.8 
- Pytorch == 1.6.0
- scikit-learn==1.0.2

- numpy==1.20.3
- scanpy==1.7.2
- gseapy==0.10.8

## Usage

1. __Preparing  for gene expression profiles and  gene-gene adjacent matrix__
   
  This method integrating a variational autoencoder-based model and Top-K sparsification strategy to address the data sparsity, reduce technical noise, and insufficient prior knowledge. Furthermore, it ulitizes a hybrid Graph Convolutional Network–Graph Attention Network (GCN–GAT) architecture that simultaneously captures both global structural information and local dependencies within GRNs, enabling more comprehensive and accurate GRN construction.  

2. **Command to run HAlink**
   
   To train an ab initio model, simply uses the script 'HAlink_main.py'.
   
   `` python HAlink_main.py``
   
   To apply dot product as score metric:
   
   Type == 'dot', flag== False
   
   To apply causal inference:
   
   Type == 'MLP', flag==True
   
   

