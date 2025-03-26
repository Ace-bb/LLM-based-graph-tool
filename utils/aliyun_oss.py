import oss2, os
from dotenv import load_dotenv
load_dotenv()

def upload_file_to_oss(file_path, object_name):
    object_name = object_name.replace(' ', '')
    bucket_name, access_key_id, access_key_secret, endpoint = os.environ.get('OSS_BUCKET_NAME'), os.environ.get('OSS_ACCESS_KEY_ID'), os.environ.get('OSS_ACCESS_KEY_SECRET'), os.environ.get('OSS_ENDPOINT')
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.put_object_from_file(object_name, file_path)
    return f"https://{bucket_name}.{endpoint}/{object_name}"

if __name__=="__main__":
    print(upload_file_to_oss("../data/datasets/Flowchart2DotDatasets/dotV2/FlowchartImages/内分泌科/Donohue 综合征/2_2.png", "内分泌科/Donohue 综合征/2_2.png"))