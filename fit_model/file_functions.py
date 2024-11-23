import os

data_process_path = os.path.join('..', 'data_process')

def get_filepaths_in_folder_with_ending(folder, ending):
    target_dir = os.path.join(data_process_path, folder)
    file_infos = [(os.path.join(target_dir, file), os.path.splitext(file)[0])
             for file in os.listdir(target_dir) if file.endswith(ending)]
    
    return [path for path, _ in file_infos]