import requests

url = 'http://127.0.0.1:5000/login'
credentials = {
    'username': 'dbuser',
    'password': 'mypassword'
}

response = requests.post(url, json=credentials)

if response.status_code == 200:
    token = response.json().get('access_token')
    print("Token: {token}")
else:
    print("Failed to obtain token")
