import requests
from selenium import webdriver
import time
driver = webdriver.Chrome()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
}
base_url = 'https://unsplash.com/napi/search/photos?query=job&xp=&per_page=20&page=2'
response = requests.get(base_url, headers=headers)
data = response.json()



def get_data(base_url):
    response = requests.get(base_url, headers=headers)
    data = response.json()
    return data

B
def download(d):
    datas = d.get('results')
    for data in datas:
        id = data.get('id')
        #rEn-AdBr3Ig
        t_url = data.get('urls').get('full')
        #https://images.unsplash.com/photo-1604357209793-fca5dca89f97?crop=entropy&cs=srgb&fm=jpg&ixid=M3wxMjA3fDB8MXxzZWFyY2h8Mnx8bWFwfGVufDB8fHx8MTY5ODg5MzI2M3ww&ixlib=rb-4.0.3&q=85
        t = t_url.index('ixid=')
        t_url = t_url[:t]
        name = data.get('user').get('name')
        name = name.replace(" ", '-')
        #tamas-tuzes-katai - (id)rEn-AdBr3Ig
        url = t_url + 'dl=' + name + '-' + id + '-unsplash.jpg'
        driver.get(url)


data = get_data(base_url)
download(data)
time.sleep(30)
driver.quit()
