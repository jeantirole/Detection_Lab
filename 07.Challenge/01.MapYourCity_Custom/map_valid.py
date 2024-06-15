def validate(model):     
  print('Now validating')                     
  model.eval()    
  running_loss = 0.0  
  running_correct = 0
  with torch.no_grad():
      #for _, data in tqdm(enumerate(dataloaders['val'])):       
      for idx, batch in enumerate(valid_dataloader):
          pixel_values, pixel_values2, pixel_values3, labels = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32), batch[2].to(device, dtype=torch.float32), batch[3].to(device)
          pixel_values = pixel_values.permute(0, 3, 1, 2)         
          pixel_values2 = pixel_values2.permute(0, 3, 1, 2)   
          pixel_values3 = pixel_values3.permute(0, 3, 1, 2)
          outputs = model(pixel_values, pixel_values2, pixel_values3)
          _, preds = torch.max(outputs, 1)
          running_correct += (preds == labels).sum().item() 
      accuracy = 100. * running_correct / len(valid_dataloader.dataset)
      print(f'Val Acc: {accuracy:.2f}')
      return accuracy
