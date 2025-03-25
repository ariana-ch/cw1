import gdown
import zipfile
import os


def download_data():
    data_path = os.path.join("data", "casia-webface")

    if os.path.isdir(data_path):
        print("Data already exists. Skipping download.")
    else:
        print("Data not found. Proceeding to download and extract.")
    
    file_id = "1guvJwIdHKi4-AFRPs0v-6kfHZhEuo8x8"
    url = f"https://drive.google.com/uc?id={file_id}"

    output_zip = "data.zip"

    print("Downloading...")
    gdown.download(url, output_zip, quiet=False)

    print("Extracting...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove(output_zip)
    print("Done.")
