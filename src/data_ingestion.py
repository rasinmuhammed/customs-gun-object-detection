import os
import kagglehub
import shutil
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import *
import zipfile

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        raw_dir = os.path.join(self.target_dir, "raw")
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created directory: {raw_dir}")
            except Exception as e:
                logger.error(f"Error creating directory: {raw_dir}")
                raise CustomException(f"Error creating directory: {raw_dir}", e)
            
        return raw_dir
    
    def extract_images_and_labels(self, path: str, raw_dir: str):
        try:
            extract_dir = os.path.dirname(path)

            # Step 1: Extract the zip file
            if path.endswith('.zip'):
                logger.info("Extracting zip file...")
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

            # Step 2: Search for folders containing 'image' or 'label'
            found_images = False
            found_labels = False

            for root, dirs, files in os.walk(extract_dir):
                for dir_name in dirs:
                    lower_dir = dir_name.lower()
                    full_path = os.path.join(root, dir_name)

                    if 'image' in lower_dir:
                        shutil.copytree(full_path, os.path.join(raw_dir, 'Images'), dirs_exist_ok=True)
                        logger.info(f"Copied image folder from {full_path} to {raw_dir}/Images")
                        found_images = True

                    elif 'label' in lower_dir or 'annot' in lower_dir:
                        shutil.copytree(full_path, os.path.join(raw_dir, 'Labels'), dirs_exist_ok=True)
                        logger.info(f"Copied label folder from {full_path} to {raw_dir}/Labels")
                        found_labels = True

            if not found_images:
                logger.warning(f"No folder with 'image' found under extracted path {extract_dir}")
            if not found_labels:
                logger.warning(f"No folder with 'label' or 'annot' found under extracted path {extract_dir}")

        except Exception as e:
            logger.error(f"Error extracting images and labels: {e}")
            raise CustomException(f"Error extracting images and labels: {e}", e)

        
    def download_dataset(self, raw_dir: str):
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            path = kagglehub.dataset_download(self.dataset_name)
            
            self.extract_images_and_labels(path, raw_dir)
            logger.info(f"Dataset downloaded and extracted to: {raw_dir}")

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise CustomException(f"Error downloading dataset: {e}", e)
        

    def run(self):
        try:
            logger.info("Starting data ingestion process")
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data ingestion process completed successfully")

        except Exception as e:
            logger.error(f"Error in data ingestion process: {e}")
            raise CustomException(f"Error in data ingestion process: {e}", e)
    

if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    data_ingestion.run()