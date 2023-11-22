import zipfile
import concurrent.futures
import os
import time

def unzip_file(zip_file, output_dir):
    start_time = time.time()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    end_time = time.time()
    print(f"Unzipped {zip_file} in {end_time - start_time:.2f} seconds")

def parallel_unzip(zip_file, output_dir, num_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_zip = {executor.submit(unzip_file, zip_file, output_dir): zip_file}
        for future in concurrent.futures.as_completed(future_to_zip):
            zip_file = future_to_zip[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to unzip {zip_file}: {str(e)}")

if __name__ == "__main__":
    zip_file_path = "/run/media/tobias/Archiv/CheXpert_Dataset/chexpertchestxrays-u20210408/CheXpert-v1.0.zip"  # Replace with the path to your .zip file
    output_directory = "/run/media/tobias/Archiv/CheXpert_Dataset/chexpertchestxrays-u20210408"  # Replace with the directory where you want to extract the files
    num_workers = 14

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Call the parallel_unzip function to unzip the file with parallel processing
    parallel_unzip(zip_file_path, output_directory, num_workers)
