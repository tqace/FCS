import yaml
import wandb
import argparse
from src import build_dataset, build_model
import torch
import os

def main(config):
    
    wandb.init(project="FCS", name=config['exp']['name'])
    
    # Checkpoint dir
    save_path = os.path.join('checkpoints','stage_{}'.format(config['exp']['stage']),config['exp']['name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create dataset
    train_set,val_set = build_dataset(config['dataset'])

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config['training']['batch_size'],
            shuffle=True, 
            num_workers=config['training']['num_workers'],
            drop_last=True
            )
    
    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['training']['val_batch_size'],
            shuffle=True, 
            num_workers=config['training']['num_workers'],
            drop_last=True
            )

    ## Create  model
    model = build_model(config['model'])
    pretrained_path = config['training']['pretrained_path']
    if pretrained_path:
        model.load(pretrained_path)

    ## Will use all visible GPUs if parallel=True
    parallel = config['training']['parallel']
    if parallel and torch.cuda.device_count() > 1:
      print('Using {} GPUs'.format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
      module = model.module

    ## Set up optimizer and data logger
    optimizer = torch.optim.Adam(model.parameters(),lr=config['optimizer']['lr'], weight_decay=config['optimizer']['regularization'])
    model = model.to('cuda:0')
    for epoch in range(config['training']['epochs']):
            
            print('On epoch {}'.format(epoch))
            mbatch_cnt = 0
            for i,mbatch in enumerate(train_loader):
    
                x = mbatch.to('cuda:0')		
                ## Forward pass 
                ret = model.forward(x)	
                loss_rec = ret['loss_rec']
                mse = ret['mse']
                loss_preds = ret['loss_preds']
                mse_preds = ret['mse_preds']


                ## Backwards Pass
                if parallel:
                    loss_rec = loss_rec.mean()
                    mse = mse.mean()
                    loss_preds = loss_preds.mean()
                    mse_preds = mse_preds.mean()
                optimizer.zero_grad()
                total_loss = loss_rec+loss_preds
                total_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
                optimizer.step()

                ## Print outputs & log
                if i%50==0:
                    wandb.log({"train_mse": mse.cpu().item()})
                    print('epoch:{}    mbatch:{}    loss_rec:{:.0f}    loss_preds:{:.0f}    MSE:{:.5f}    MSE_preds:{:.5f}'.format(epoch, mbatch_cnt, loss_rec.item(),loss_preds.item(), mse.item(), mse_preds.item()))
                mbatch_cnt += 1

            ## Validation
            val_mse = 0
            val_mse_preds = 0
            if epoch % config['training']['val_interval'] == 0:
                model.eval()
                for mbatch in val_loader:
                    x = mbatch.to('cuda:0')
                    ret = model.forward(x)
                    val_mse += ret['mse'].mean().cpu().item()
                    val_mse_preds += ret['mse_preds'].mean().cpu().item()
                val_mse = val_mse / len(val_loader.dataset)
                val_mse_preds = val_mse_preds / len(val_loader.dataset)
                wandb.log({"val_mse": val_mse})
                wandb.log({"val_mse_preds": val_mse_preds})
                wandb.log({"input_image": wandb.Image(mbatch[0])})
                wandb.log({"reconstructed_image": wandb.Image(ret['reconstruction'][0].cpu())})
                wandb.log({"predicted_image": wandb.Image(ret['prediction'][0].cpu())})
            if epoch % config['training']['save_interval'] == 0: 
                module.save(save_path,epoch=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InphyGPT training')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config,'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    main(config)
