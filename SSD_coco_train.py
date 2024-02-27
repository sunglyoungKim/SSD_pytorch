import os
import shutil
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A


import src
import numpy as np
from tqdm.autonotebook  import tqdm
import pathlib



class SSD_coco_train():
  def __init__(self, project_name, train_path, epochs, batch_size, n_class, train_img_aug, val_img_aug):
      
      self.project_name = project_name
      self.train_path = train_path
      
      # self.img_train_list = self.find_files(train_path, ['*.jpg', '*.png'])
      # self.target_train_list = self.find_files(train_path, ['*.json'])
      # train_label_path = pathlib.Path(train_path).parents[1].joinpath('labels/train')
      # self.target_train_list = self.find_files(train_label_path, ['*.txt'])
      
      # self.img_val_list = self.find_files(val_path, ['*.jpg', '*.png'])
      # self.target_val_list = self.find_files(val_path, ['*.json'])
      # val_label_path = pathlib.Path(val_path).parents[1].joinpath('labels/val')
      # self.target_val_list = self.find_files(val_path, ['*.txt'])
      
      self.save_folder = f"trained_models/{project_name}"
      self.log_path = f"tensorboard/SSD_{project_name}"
      self.epochs = epochs
      self.batch_size = batch_size
      self.num_workers=20
      self.check_p = 10
      self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
      self.n_class = n_class
      self.dboxes = src.utils.generate_dboxes(model="ssd")
      
      self.train_img_aug = train_img_aug
      self.val_img_aug = val_img_aug
      
      # self.train_set = src.dataset.(self.data_path, 2017, "train", src.transform.SSDTransformer(self.dboxes, (300, 300), val=False))
      self.train_set = src.coco_dataset.CocoTrainDataset(self.train_path, 'train', self.dboxes, img_aug = self.train_img_aug)
      self.train_loader = DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
      
      self.val_set = src.coco_dataset.CocoValDataset(self.train_path, 'val', self.dboxes, img_aug = self.val_img_aug)
      self.val_loader = DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers)

      self.criterion = src.loss.Loss(self.dboxes)
      self.checkpoint_path = os.path.join(self.save_folder, f'SSD_checkpoint_{self.project_name}.pth')


      # self.layout = {"SSD_loss":{"loss":["Multiline", ["loss/train", "loss/validation"]] }}        
      

        
      
  def find_files(self, dir_path, patterns=[None], exclusive_patterns=[None]):
      """
      Returns a generator yielding files matching the given patterns

      dir_path: Directory to search for files under. Defaults to current dir.
      patterns: Patterns of files to search for. Defaults to ["*"]. Example: ["*.json", "*.xml"]
      exclusive patterns: patterns of files not to serach for. Defaults to [None]

      """

      path = dir_path
      inclusive_path_patterns = patterns
      exclusive_path_patterns = exclusive_patterns
      all_files = pathlib.Path(dir_path)

      filtered_set = set()

      for pattern in inclusive_path_patterns:
          filtered_set = filtered_set.union(set(all_files.rglob(pattern)))

      for exclusive in exclusive_path_patterns:
          if exclusive == None:

              filtered_set = (file for file in filtered_set)

          else:

              filtered_set = filtered_set - set(all_files.rglob(exclusive))

      return sorted(list(filtered_set))




  def train_one_epoch(self, model, epoch):
      '''
      training for one epoch
      
      model : Pytorch training model
      epoch : number of epoch
      '''
              
      model.train()
      train_loss = 0.0
      train_bar = tqdm(self.train_loader, desc=f"batch Loss: ", leave=False)
      for i, (img, gloc, glabel) in enumerate(train_bar):
          
          img = img.to(self.device)
          gloc = gloc.to(self.device).transpose(1, 2).contiguous()
          glabel = glabel.to(self.device)

          self.optimizer.zero_grad()
          
          ploc, plabel = model(img)
          loss = self.criterion(ploc, plabel, gloc, glabel)

          loss.backward()
          self.optimizer.step()
          train_loss += loss.item()
          c_train_loss = train_loss  / (i+1)
          train_bar.set_description(f"Epoch: {epoch} Train loss: {c_train_loss:.3f} batch Loss: {loss.item():.3f}")

      
      if epoch % self.check_p == 0:
          checkpoint = {"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": self.optimizer.state_dict()}
  
          checkpoint_path = os.path.join(self.save_folder, f"SSD_checkpoint_{self.project_name}.pth")
          torch.save(checkpoint, checkpoint_path)

        
      return c_train_loss
      
  def eval_one_epoch(self, model, epoch, eval_loss):
      '''
      evaluation for one epoch
      
      model : Pytorch training model
      epoch : number of epoch
      eval_loss : evaluation loss
      '''
      
      model.eval()
      val_loss = 0.0
      val_bar = tqdm(self.val_loader, desc=f"batch Loss: ", leave=False)
      
      with torch.no_grad():
          for i, (img, gloc, glabel) in enumerate(val_bar):

              img = img.to(self.device)
              gloc = gloc.to(self.device).transpose(1, 2).contiguous()
              glabel = glabel.to(self.device)

              ploc, plabel = model(img)
              loss = self.criterion(ploc, plabel, gloc, glabel)
            
              val_loss += loss.item()
              c_val_loss = val_loss  / (i+1)
              val_bar.set_description(f"Epoch: {epoch} val loss: {c_val_loss:.3f} batch Loss: {loss.item():.3f}")
              
          if c_val_loss < eval_loss:
              eval_loss = c_val_loss
              checkpoint = {"epoch": epoch, 
                            "model_state_dict": model.state_dict(),
                            "optimizer": self.optimizer.state_dict()}

              checkpoint_path = os.path.join(self.save_folder, f"best_SSD_{self.project_name}.pth")
              torch.save(checkpoint, checkpoint_path)
          
      return eval_loss, c_val_loss
    

  def train(self):
      
      model = src.model.SSD(backbone=src.model.ResNet(), num_classes=self.n_class)
      
      model.to(self.device)
      
      self.criterion.to(self.device)
      self.optimizer = torch.optim.Adam(model.parameters())

      if os.path.isdir(self.log_path):
          shutil.rmtree(self.log_path)
      os.makedirs(self.log_path)

      if not os.path.isdir(self.save_folder):
          os.makedirs(self.save_folder)

      
          
      writer = SummaryWriter(self.log_path)
      
      if os.path.isfile(self.checkpoint_path):
          checkpoint = torch.load(self.checkpoint_path)
          first_epoch = checkpoint["epoch"] + 1
          model.load_state_dict(checkpoint["model_state_dict"])
          self.optimizer.load_state_dict(checkpoint["optimizer"])
      else:
          first_epoch = 0

      eval_loss = np.inf

      for epoch in range(first_epoch, self.epochs):
          
          train_loss = self.train_one_epoch(model, epoch)
          eval_loss, val_loss = self.eval_one_epoch(model, epoch, eval_loss)

          writer.add_scalars('Training vs. Validation Loss',
              { 'Training' : train_loss, 'Validation' : val_loss },
              epoch + 1)
      
          writer.flush()

        
      return
  
# if __name__ == "__main__":
#     img_path = '/home/sk/Rewire_Image/Rewire_original_models/c-fos/train'
#     val_path = '/home/sk/Rewire_Image/Rewire_original_models/c-fos/val'  
#     project_name = 'c-fos'
#     t = SSD_train(project_name, train_path, val_path)
#     t.train()