import csv
import os

class Logger:
    def __init__(self, directory, fname, fields):
        self.directory = directory
        self.filename = directory + fname
        self.fields = fields
        self._initialize_file()

    def _initialize_file(self):
        file_exists = os.path.isfile(self.filename)
        os.makedirs(self.directory, exist_ok=True)
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()
        print("Opened log file at "+self.filename)

    def log(self, data):
        if set(data.keys()) != set(self.fields):
            raise ValueError("Data keys must match the defined fields.")
        
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow(data)