import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger()

code = '208340'
ticker = 'YETI'

try:
    # url_ticker = 'http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A ' +code +'&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701'
    url_ticker = "https://www.nasdaq.com/market-activity/stocks/{}".format(ticker.lower())
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/84.0.4147.135 Safari/537.36'
    }
    res = requests.get(url_ticker, headers=headers)
    content_type = res.headers['content-type']
    if not 'charset' in content_type:
        res.encoding = res.apparent_encoding

    soup = BeautifulSoup(res.text, 'lxml')

    object1 = soup.find("span", attrs={"class": "symbol-page-header__name"})
    # value1 = object1.find('h3')

    # object1 = soup.find("div", attrs={"class": "um_bssummary"})
    # value1 = object1.find('h3')
    print(object1.get_text().strip())

    value2 = object1.find('li')
    print(value2.get_text().strip())

except Exception as ex:
    logger.info(f'기업 상세 자료 가져오기 실패!! : {ex}')