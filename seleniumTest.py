from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Selenium 웹 드라이버 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창을 열지 않고 실행
chrome_options.add_argument("--disable-gpu")  # GPU 가속 비활성화 (필요한 경우)
chrome_options.add_argument("--no-sandbox")  # 사이트 격리 비활성화 (필요한 경우)

# 원하는 헤더 정보 추가
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)")  # User-Agent 정보 추가
chrome_options.add_argument("--accept-language=en-US,en")  # Accept-Language 정보 추가

# service = Service('D:\Project\driver\chromedriver')  # 크롬 드라이버 경로
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

# 웹 페이지 렌더링
driver.get('https://www.nasdaq.com/news-and-insights/topic/markets/stocks/page/2')

# 최종 랜더링된 HTML 소스 얻기
html_source = driver.page_source

# BeautifulSoup을 사용하여 파싱
soup = BeautifulSoup(html_source, 'html.parser')
article_tag = soup.findAll('a', class_='jupiter22-c-article-list__item_title')

for article in article_tag:
    print(article.text)

# 원하는 작업 수행
# ...

# 드라이버 종료
driver.quit()