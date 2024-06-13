import os
from google.cloud import storage


# Set credentials key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_key_gcloud.json'


# Upload folder
def upload_folder(bucket_name, source_folder, dataset_subdir_name):
    """Upload a folder to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Change \ to / on windows
            local_file_path = local_file_path.replace('\\', '/')
            upload_file_path = dataset_subdir_name + '/' + local_file_path
            blob = bucket.blob(upload_file_path)
            blob.upload_from_filename(local_file_path)
            print(f'Uploaded {local_file_path} to {upload_file_path} in bucket {bucket_name}.')


# Upload folder
bucket_name = 'airfoil-shape-optimization-dl-data'
folder_to_upload = 'generated_airfoils'
dataset_subdir_name = 'HV0.7_MV0.3_LV0.1'
upload_folder(bucket_name, folder_to_upload, dataset_subdir_name)