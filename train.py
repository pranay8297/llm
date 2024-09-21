from model import *
from data import *

save_after_every = 50

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'

if device == 'cuda': 
    torch.set_float32_matmul_precision('high')

config = GPTConfig(vocab_size = 50304)
model = GPT(config)
model = model.to(device)

tokens_per_grad_update = 2**19
B = 16; T = 1024
assert tokens_per_grad_update % (B*T) == 0
grad_accumulation_steps = int(tokens_per_grad_update/(B*T))
print(f"Gradiant Accumulation Steps: {grad_accumulation_steps}")

# 
# simple training loop for just one batch
iterations = 1e09//tokens_per_grad_update

if device == 'cuda': model = torch.compile(model)

opt = model.configure_optmizers(lr = 6e-04, wd = 1e-01, betas = (0.9, 0.95), eps = 1e-08, device_type = device)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = 6e-04, total_steps = iterations, final_div_factor=10.0)
dl = DataLoaderLite(B = B, T = T)

losses = []

opt = model.configure_optmizers(lr = 6e-04, wd = 1e-01, betas = (0.9, 0.95), eps = 1e-08, device_type = device)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = 6e-04, total_steps = iterations, final_div_factor=10.0)
dl = DataLoaderLite(B = B, T = T)

for i in range(iterations): 
    opt.zero_grad()
    accumulated_loss = 0.
    
    t1 = time.time()
    for j in range(grad_accumulation_steps):
        x, y = dl.next_batch()
        x, y = x.to(device), y.to(device)
        
        if device == 'cuda': 
            with torch.autocast(device_type = device, dtype = torch.bfloat16):
                logits, loss = model(x, y)
        else: 
            logits, loss = model(x, y)

        loss /= grad_accumulation_steps    
        losses.append(loss.item())
        accumulated_loss += loss.item()
        loss.backward()
    
    if device == 'cuda': torch.cuda.synchronize()
    t2 = time.time()
    elapsed_time = t2 - t1
    tps = B*T*grad_accumulation_steps/elapsed_time

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    lr_scheduler.step()
    print(f'Iteration: {i} | Loss: {accumulated_loss:.5f} | norm: {norm:.5f} | time: {(elapsed_time/grad_accumulation_steps):.4f} | tps: {tps:.3f}')

    if i%save_after_every == 0: 
        torch.save(model, f'./ckpts/checkpoint_{i//save_after_every}.pt')