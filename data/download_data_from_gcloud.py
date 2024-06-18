import os
from google.cloud import storage

# Set credentials key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_key_gcloud.json'

# Download folder
def download_folder(bucket_name, destination_folder, dataset_subdir_name):
    """Download a folder from the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix = dataset_subdir_name)

    for blob in blobs:
        if not blob.name.endswith('/'):
            # Create local directories if necessary
            local_file_path = os.path.join(destination_folder, blob.name)
            local_file_path = local_file_path.replace('\\', '/')
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download the file
            blob.download_to_filename(local_file_path)
            print(f'Downloaded {blob.name} to {local_file_path}.')


# Download folder
bucket_name = 'airfoil-shape-optimization-dl-data'
destination_folder = 'downloaded_airfoils'
dataset_subdir_name = 'HV0.2_MV0.15_LV0.1'
download_folder(bucket_name, destination_folder, dataset_subdir_name)