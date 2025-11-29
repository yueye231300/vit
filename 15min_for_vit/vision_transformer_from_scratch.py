import numpy as np

from tqdm import tqdm,trange

import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(42)
torch.manual_seed(42)


def pathchify(images,n_patches):
    n,c,h,w = images.shape 
    assert h % n_patches ==0 and w % n_patches ==0, "Image dimensions must be divisible by number of patches"
    assert h == w, "Currently only square images are supported"
    patches = torch.zeros((n,n_patches**2,c*h*w//n_patches**2))
    patch_size = h // n_patches 
    for index, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                patches[index,i*n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length,d):
    results = torch.ones(sequence_length,d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 ==0:
                results[i,j] = np.sin(i / (10000 ** (2 * (j // 2) / d)))
            else:
                results[i,j] = np.cos(i / (10000 ** (2 * (j // 2) / d)))
    return results

class MyMSA(nn.Module):
    def __init__(self,d,n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads ==0, "d must be divisible by n_heads"
        d_head = int(d/n_heads)
        # define the linear mappings for Q,K,V for each head
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # sequence shape is (batch_size, seq_len, d)
        # go into shape   (batch_size, seq_len, n_heads, d_head)
        # come back to (batch_size, seq_len, item_dim)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                seq_head = sequence[:,head*self.d_head:(head+1)*self.d_head]
                Q = self.q_mappings[head](seq_head)
                K = self.k_mappings[head](seq_head)
                V = self.v_mappings[head](seq_head)

                # compute attention scores 
                scores = torch.matmul(Q,K.T) / np.sqrt(self.d_head)
                attn_weights = self.softmax(scores)
                head_out = torch.matmul(attn_weights,V)
                seq_result.append(head_out)
            seq_result = torch.cat(seq_result,dim=-1)
            result.append(seq_result)
        return torch.stack(result)


class MyViTblock(nn.Module):
    def __init__(self,hidden_dim=8,n_heads =2,mlp_ratio=4):
        super(MyViTblock, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norml = nn.LayerNorm(hidden_dim)
        self.mhsa = MyMSA(hidden_dim,n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim*mlp_ratio,hidden_dim)
        )  
    def forward(self,x):
        out = x + self.mhsa(self.norml(x))
        out = out + self.mlp(self.norm2(out))
        return out
    




class MyViT(nn.Module):
    def __init__(self,chw=(1,28,28),n_patches=7,n_blocks=2,hidden_dim=8,n_heads =2,out_dim = 10):
        super(MyViT, self).__init__()

        # attributes 
        self.chw = chw 
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.out_dim = out_dim

        assert chw[1] % n_patches ==0 and chw[2] % n_patches ==0, "Image dimensions must be divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        #1) linear mapper 
        self.input_d = int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)

        #2) learnable class token 
        self.class_token = nn.Parameter(torch.randn(1,self.hidden_dim))

        #3） positional embeddings 
        self.pos_embed = nn.Parameter(get_positional_embeddings(n_patches**2 + 1, self.hidden_dim))
        self.pos_embed.requires_grad = False  # non-trainable

        # 4) transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [MyViTblock(hidden_dim=self.hidden_dim,n_heads=self.n_heads) for _ in range(self.n_blocks)]
        )
        # 5) final classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_dim,self.out_dim)
        )
    


    def forward(self, image):
        patches = pathchify(image, self.n_patches).to(image.device) 
        tokens = self.linear_mapper(patches)
        
        # 使用高效的批量操作
        batch_size = tokens.shape[0]
        class_tokens = self.class_token.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1)
        
        # add positional embeddings
        pos_embed = self.pos_embed.unsqueeze(0)
        out = tokens + pos_embed
        
        # transformer blocks
        for block in self.transformer_blocks:
            out = block(out)

        out = out[:, 0]  # get the class token output
        out = self.classification_head(out)
        return out




def main():
    # loading data 
    transform = ToTensor()
    train_dataset = MNIST(root='./data',train=True,transform=transform,download=True)
    test_dataset = MNIST(root='./data',train=False,transform=transform,download=True)

    train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)

    test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False)

    # defining model and traing options 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MyViT((1,28,28),n_patches=7,n_blocks=4,hidden_dim=8,n_heads = 2,out_dim = 10).to(device)
    N_epochs = 5
    LR = 0.005 
    # training loop 
    optimizer = Adam(model.parameters(),lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_epochs,desc="training",leave = False):
        model.train()
        running_loss = 0.0
        for images,labels in tqdm(train_loader,desc="Training",leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logist = model(images)
            loss = criterion(logist,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{N_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # testing loop 
    with torch.no_grad():
        correct , total =0,0
        test_loss = 0.0
        for batch in tqdm(test_loader,desc="Testing",leave=False):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logist = model(images)
            output = torch.softmax(logist,dim=1)
            loss = criterion(logist,labels)
            test_loss += loss.item()
            total += labels.size(0)
            correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":

    main()