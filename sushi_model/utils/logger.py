import logging

import os

from datetime import datetime



class SushiLogger:

    _instance = None

    

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._initialize_logger()

        return cls._instance

    

    def _initialize_logger(self):

        self.logger = logging.getLogger('SashimiModel')

        self.logger.setLevel(logging.INFO)

        

        # Create logs directory if it doesn't exist

        os.makedirs('logs', exist_ok=True)

        

        # Create session-based log file

        session_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_handler = logging.FileHandler(f'logs/model_{session_time}.log')

        

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(formatter)

        

        self.logger.addHandler(file_handler)

    

    def log_model_state(self, component: str, metadata: dict):

        self.logger.info(f"Component: {component} - Metadata: {metadata}")

    

    def log_error(self, error: Exception, component: str):

        self.logger.error(f"Error in {component}: {str(error)}")
