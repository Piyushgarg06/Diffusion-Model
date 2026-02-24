import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

T=200
betas=torch.linspace(1e-4,0.02,T)
alphas=1.0-betas
alpha_bar=torch.cumprod(alphas,dim=0)

def generate_sparse_batch(batch_size=16,size=32, jitter=3):
    x=torch.zeros(batch_size,1,size,size)
    centre = size//2
    for i in range(batch_size):
        dx = torch.randint(-jitter,jitter+1,(1,))
        dy = torch.randint(-jitter,+jitter,(1,))

        cx = centre+dx
        cy=centre+dy
        x[i,0,cx,cy] = torch.rand(1)*5
    return x 

class smallDiffusionModel(nn.Module):
    def __init__(self,T,time_dim=32):
        super().__init__()
        self.time_embed = nn.Embedding(T,time_dim)
        self.time_mlp = nn.Linear(time_dim,16)
        self.conv1=nn.Conv2d(1,16,3,padding=1)
        self.conv2=nn.Conv2d(16,1,3,padding=1)
    def forward(self,x,t):
        h=self.conv1(x)

        time_emb = self.time_embed(t)
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)

        h=h+time_emb
        h=torch.relu(h)
        
        out=self.conv2(h)
        return out
model = smallDiffusionModel(T)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for step in range(3000):
    x0 = generate_sparse_batch()
    batch = x0.shape[0]

    t=torch.randint(0,T,(batch,))
    epsillon = torch.randn_like(x0)
    alpha_t = alpha_bar[t].view(batch,1,1,1)
    x_t = torch.sqrt(alpha_t)*x0 + torch.sqrt(1-alpha_t)*epsillon
    pred_noise = model(x_t,t)
    mask = (x0 > 0).float()
    weight = 1 + 80 * mask
    loss = ((pred_noise - epsillon) ** 2 * weight).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mask = (x0 > 0)
    bg_mask = (x0 == 0)

    spike_loss = F.mse_loss(pred_noise[mask], epsillon[mask])
    bg_loss = F.mse_loss(pred_noise[bg_mask], epsillon[bg_mask])

    

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
        print("Spike loss:", spike_loss.item(), "BG loss:", bg_loss.item())
    

@torch.no_grad()
def sample(model,T,shape):
    x=torch.randn(shape)
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        pred_noise = model(x,t_tensor)
        alpha_t = alphas[t].view(1,1,1,1)
        alpha_bar_t = alpha_bar[t]
        alpha_t= alpha_t.view(1,1,1,1)
        alpha_bar_t = alpha_bar_t.view(1,1,1,1)
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )
        if t > 0:
            pass
    return x 

samples = sample(model, T, shape=(16, 1, 32, 32))
print("Sampled shape:", samples.shape)
print("Sample values:\n", samples[0, 0])

with torch.no_grad():
    x0 = generate_sparse_batch()
    batch = x0.shape[0]
    t = torch.randint(0, T, (batch,))
    eps = torch.randn_like(x0)
    alpha_t = alpha_bar[t].view(batch,1,1,1)
    x_t = torch.sqrt(alpha_t)*x0 + torch.sqrt(1-alpha_t)*eps

    pred = model(x_t, t)
    print("MSE:", F.mse_loss(pred, eps).item())

with torch.no_grad():
    x0 = generate_sparse_batch()
    batch = x0.shape[0]

    t = torch.full((batch,), T-1)   # force largest timestep
    eps = torch.randn_like(x0)

    alpha_t = alpha_bar[t].view(batch,1,1,1)
    x_t = torch.sqrt(alpha_t)*x0 + torch.sqrt(1-alpha_t)*eps

    pred = model(x_t, t)
    print("High-t MSE:", F.mse_loss(pred, eps).item())

plt.imshow(samples[0,0].cpu(), cmap='gray')
plt.colorbar()
plt.show()