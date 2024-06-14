from google.cloud import storage

def list_buckets():
    """Lists all buckets."""
    storage_client = storage.Client(project='auto-diarizer')

    buckets = list(storage_client.list_buckets())

    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)

list_buckets()