import os

class PCMU:
    def __init__(self):
        
        return 


    def reset(self,folder_path):
        self.folder_path = folder_path
        self.file_list = []
        self.collect_files(self.folder_path)
        self.file_index = 0
        file=self.read_next_file()
        return file

    def collect_files(self, path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                self.collect_files(item_path)
            elif item.endswith('.csv'):
                self.file_list.append(item_path)

    def read_next_file(self):
        if self.file_index < len(self.file_list):
            file_path = self.file_list[self.file_index]
            self.file_index += 1
            return file_path
        else:
            return None

# 使用示例
pcmu = PCMU()

while True:
    file=pcmu.reset('sm_dataset')
    if file is None:
        break
    print(f'Reading file: {file}')