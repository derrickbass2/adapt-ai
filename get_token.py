import requests

url = 'http://127.0.0.1:5000/login'
credentials = {
    'username': 'dbuser',
    'password': 'mypassword'
}

try:
    response = requests.post(url, json=credentials)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

    if response.status_code == 200:
        token = response.json().get('access_token')
        print(f"Token: {token}")
    else:
        print(f"Failed to obtain token. Status code: {response.status_code}, Response: {response.text}")
except requests.RequestException as e:
    print(f"An error occurred: {e}")
