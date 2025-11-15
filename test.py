import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

access_key = os.getenv('BUCKET_COMPATIBLE_ACCESS_KEY')
secret_key = os.getenv('BUCKET_COMPATIBLE_SECRET_KEY')

print("Testing Backblaze B2 Native API...")
print("=" * 60)
print(f"Access Key (keyID): {access_key}")
print(f"Secret Key: {secret_key[:20]}...")
print("=" * 60)

# Backblaze uses Basic Auth for authorization
auth_string = f"{access_key}:{secret_key}"
auth_header = base64.b64encode(auth_string.encode()).decode()

headers = {
    'Authorization': f'Basic {auth_header}'
}

try:
    
    print("\n1️⃣  Authorizing with B2 API...")
    response = requests.get(
        'https://api.backblazeb2.com/b2api/v2/b2_authorize_account',
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Authorization successful!")
        print(f"   Account ID: {data.get('accountId')}")
        print(f"   API URL: {data.get('apiUrl')}")
        print(f"   Download URL: {data.get('downloadUrl')}")
        print(f"   S3 Endpoint: {data.get('s3ApiUrl')}")
        

        auth_token = data.get('authorizationToken')
        api_url = data.get('apiUrl')
        
        print(f"\n2️⃣  Listing buckets...")
        list_response = requests.post(
            f"{api_url}/b2api/v2/b2_list_buckets",
            headers={'Authorization': auth_token},
            json={'accountId': data.get('accountId')}
        )
        
        if list_response.status_code == 200:
            buckets_data = list_response.json()
            print("✅ Buckets found:")
            for bucket in buckets_data.get('buckets', []):
                print(f"   - {bucket.get('bucketName')} (ID: {bucket.get('bucketId')})")
            
            print("\n🎉 Connection successful! Your credentials are valid!")
        else:
            print(f"❌ Failed to list buckets: {list_response.text}")
    else:
        print(f"❌ Authorization failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
