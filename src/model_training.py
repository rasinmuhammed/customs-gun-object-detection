import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunDataset
from torch.utils.tensorboard import SummaryWriter
import time

logger = get_logger(__name__)

model_save_path = "artifacts/models/" 
os.makedirs(model_save_path, exist_ok=True)

class ModelTrainer:
    def __init__(self, model_class, num_classes, learning_rate, epochs, dataset_path, device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        #### Tensorboard
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)


        try:
            self.model = self.model_class(self.num_classes, self.device)
            self.model.model.to(self.device)

            self.optimizer = optim.Adam(self.model.model.parameters(), lr=self.learning_rate)
            logger.info("Model Trainer initialized.")

        except Exception as e:
            logger.error(f"Error in ModelTrainer initialization: {e}")
            raise CustomException("Failed to initialize ModelTrainer", e)
        
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try: 
            dataset = GunDataset(self.dataset_path, self.device)
            dataset = torch.utils.data.Subset(dataset, range(300))
            
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            logger.info(f"Dataset split into {train_size} training samples and {val_size} validation samples.")

            train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn= self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn= self.collate_fn)

            return train_loader, val_loader
        
        except Exception as e:
            logger.error(f"Error in splitting dataset: {e}")
            raise CustomException("Failed to split dataset", e)
        
    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()

            for epoch in range(self.epochs):
                logger.info(f"Epoch {epoch+1}/{self.epochs}")
                self.model.model.train()
                
                for i,(images, targets) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model.model(images, targets)

                    if isinstance(losses, dict):
                        total_loss = 0
                        for key, value in losses.items():
                            if isinstance(value, torch.Tensor):
                                total_loss += value

                        if total_loss == 0:
                            logger.error("There was error in losses capturing")
                            raise ValueError("Losses are zero")
                        
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch * len(train_loader) + i)
                        
                    else:
                        total_loss = losses[0]

                    total_loss.backward()
                    self.optimizer.step()

                self.writer.flush()

                self.model.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for images, targets in val_loader:
                        val_losses = self.model.model(images, targets)
                        logger.info(type(val_losses))
                        logger.info(f"VAL_LOSSES: {val_losses}")

                model_path = os.path.join(model_save_path, "fasterrcnn.pth")
                torch.save(self.model.model.state_dict(), model_path)
                logger.info(f"Model saved at {model_path}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise CustomException("Training failed", e)
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training = ModelTrainer(model_class=FasterRCNNModel, 
                            num_classes=2, 
                            learning_rate=0.0001, 
                            epochs=1, 
                            dataset_path="artifacts/raw", 
                            device=device)
    
    training.train()
