import time 
import torch
import torch.nn.functional as F
from torch.nn import Sequential

#----------------------------------------------------------------------------

def s5cl(
    args,
    encoder, 
    embedder, 
    classifier, 
    optimizer_enc, 
    optimizer_emb, 
    optimizer_cls, 
    criterion_l, 
    criterion_u, 
    criterion_p, 
    criterion_c, 
    weight_l, 
    weight_u, 
    weight_p, 
    weight_c, 
    dataloader_l, 
    dataloader_u, 
    dataloader_v, 
    device
):
    
    encoder.train()
    embedder.train()
    classifier.train()
    
    dataiter_l = iter(dataloader_l)
    dataiter_u = iter(dataloader_u)
    
    loss_history = []
     
    for step in range(args['start_step'], args['total_steps']):
        if step > 0 and step % args['eval_step'] == 0:
            model = Sequential(encoder, embedder, classifier)
            evaluate(model, dataloader_v, device)
        
        try:
            (weak_l, strong_l), label_l = next(dataiter_l)
        except StopIteration:
            dataiter_l = iter(dataloader_l)
            (weak_l, strong_l), label_l = next(dataiter_l)         
        
        try:
            (weak_u, strong_u), label_u = next(dataiter_u)
        except StopIteration:
            dataiter_u = iter(dataloader_u)
            (weak_u, strong_u), label_u = next(dataiter_u)
            
        size_l = weak_l.size()[0] 
        size_u = weak_u.size()[0]
        
        weak_l = weak_l.to(device)
        weak_u = weak_u.to(device)
        strong_l = strong_l.to(device)
        strong_u = strong_u.to(device)
        label_l = label_l.to(device)
        label_u = torch.arange(size_u)

        data = torch.cat((weak_l, strong_l, weak_u, strong_u), 0)
        target_l = torch.cat((label_l, label_l), 0)
        target_u = torch.cat((label_u, label_u), 0)
         
        optimizer_enc.zero_grad()
        optimizer_emb.zero_grad()
        optimizer_cls.zero_grad()
        
        encoding = encoder(data)   
        embedding = embedder(encoding)  
        embedding_l = embedding[:2*size_l]
        embedding_u = embedding[2*size_l:]
        embedding_c = embedding[:size_l]
        output = classifier(embedding_c)
        
        if step > args['threshold']:
            with torch.no_grad():
                _, label_p = torch.max(classifier(embedding[2*size_l:2*size_l+size_u]), 1)
            target_u = torch.cat((label_p, label_p), 0)
            criterion_u = criterion_p
            weight_u = weight_p        
         
        loss_l = criterion_l(embedding_l, target_l)  
        loss_u = criterion_u(embedding_u, target_u)
        loss_c = criterion_c(output, label_l) 
        loss_t = weight_l * loss_l + weight_u * loss_u + weight_c * loss_c
        
        loss_t.backward()
        
        optimizer_enc.step()
        optimizer_emb.step()
        optimizer_cls.step()
        
        if step % 10 == 0:
            print('[' +  '{:5}'.format(step) + '/' + '{:5}'.format(args['total_steps']) +
                  ' (' + '{:3.0f}'.format(100 * step / args['total_steps']) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss_t.item()))
            loss_history.append(loss_t.item())
            
#----------------------------------------------------------------------------

def scl(
    args,
    encoder, 
    embedder, 
    classifier, 
    optimizer_enc, 
    optimizer_emb, 
    optimizer_cls, 
    criterion_m, 
    criterion_c, 
    weight_m, 
    weight_c, 
    dataloader_t, 
    dataloader_v, 
    device
):
    
    encoder.train()
    embedder.train()
    classifier.train()
    
    dataiter_t = iter(dataloader_t)
    
    loss_history = []
     
    for step in range(args['start_step'], args['total_steps']):
        if step > 0 and step % args['eval_step'] == 0:
            model = Sequential(encoder, embedder, classifier)
            evaluate(model, dataloader_v, device)
        
        try:
            data, target = next(dataiter_t)
        except StopIteration:
            dataiter_t = iter(dataloader_t)
            data, target = next(dataiter_t)         
            
        data = data.to(device)
        target = target.to(device)
         
        optimizer_enc.zero_grad()
        optimizer_emb.zero_grad()
        optimizer_cls.zero_grad()
        
        encoding = encoder(data)   
        embedding = embedder(encoding)  
        output = classifier(embedding)
         
        loss_m = criterion_m(embedding, target)  
        loss_c = criterion_c(output, target)
        loss_t = weight_m * loss_m + weight_c * loss_c
        
        loss_t.backward()
        
        optimizer_enc.step()
        optimizer_emb.step()
        optimizer_cls.step()
        
        if step % 10 == 0:
            print('[' +  '{:5}'.format(step) + '/' + '{:5}'.format(args['total_steps']) +
                  ' (' + '{:3.0f}'.format(100 * step / args['total_steps']) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss_t.item()))
            loss_history.append(loss_t.item())
            
#----------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    
    loss_history = []
    total_samples = len(dataloader.dataset)
    correct_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for image, target in dataloader:
            image, target = image.to(device), target.to(device)
            output = F.log_softmax(model(image), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')