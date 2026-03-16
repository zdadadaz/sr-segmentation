import urllib.request
import re

def download_file_from_google_drive(id, destination):
    url = "https://docs.google.com/uc?export=download"

    session = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    response = session.open(url + '&id=' + id)
    token = get_confirm_token(response)

    if token:
        response = session.open(url + "&id=" + id + "&confirm=" + token)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for cookie in response.info().get_all('Set-Cookie', []):
        if cookie.startswith('download_warning'):
            return cookie.split('=')[1].split(';')[0]

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if chunk:
                f.write(chunk)
            else:
                break

if __name__ == "__main__":
    file_id = '154JgKpzCPW82qINcVieuPH3fZ2e0P812'
    destination = 'models/face_parsing.pth'
    print(f"Downloading from Google Drive {file_id} to {destination}...")
    try:
        download_file_from_google_drive(file_id, destination)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
