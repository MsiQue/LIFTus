import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

def train(F, model, Data, optimizer, batch_size, n_epochs):
    dataloader = DataLoader(Data, batch_size, shuffle=True)
    num_steps = (len(Data) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    for epoch in range(1, n_epochs+1):
        model.train()
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = F(model, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 10 == 0:
                print(f"step: {i}, loss: {loss.item()}")
            del loss

def inference_model(F, model, Data, batch_size):
    test_loader = DataLoader(Data, batch_size=batch_size, shuffle=False)
    model.eval()
    res = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = F(model, batch)
            if isinstance(outputs, tuple):
                res.append([o.detach().cpu() for o in outputs])
            else:
                res.append([outputs.detach().cpu()])

            del batch
            del outputs
            torch.cuda.empty_cache()


    L = len(res[0])
    return [torch.cat([t[i] for t in res], dim = 0) for i in range(L)]
