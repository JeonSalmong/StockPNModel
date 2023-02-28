import requests
from bs4 import BeautifulSoup
from datetime import datetime

from tensorflow import keras
import numpy as np
import json
import os
from konlpy.tag import Okt
import nltk

import pandas as pd
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

import cx_Oracle

import openai

import cryptocode

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

YOUR_API_KEY = '85Qs4r6Sa+pJp5zE/t7g7cOOg84nQ9jSw4I9ncRF4EWdt2o8p+wRv4KIx02uUR053gN4*WNddlDiZ242Rf30v/pT3Ag==*eUD3YPnrpi8gCR0in+RUDg==*+TwdK7nlgA8kOnpP9kZ8Jg=='

class Main():

    def __init__(self):
        # 날짜, 시간, 종목코드, 종목명, content, 긍정or부정, 확률, 종가, 전일비, 시가, 고가, 저가, 거래량
        self.date = datetime.now().strftime('%Y%m%d')
        self.time = datetime.now().strftime('%H%M')

        ###################### Oracle cloud DB #############################################################################
        # cx_Oracle.init_oracle_client(lib_dir=r".\resource\instantclient_19_17")
        cx_Oracle.init_oracle_client(lib_dir="/usr/lib/oracle/21/client64/lib")

        self.conn = cx_Oracle.connect(user='HDBOWN', password='Qwer1234!@#$', dsn='ppanggoodoracledb_high')
        self.cursor = self.conn.cursor()

        # query = """
        #         SELECT * FROM HDBOWN.HR_ORG_M
        # """
        # df = pd.read_sql(query, self.conn)

        # rows = [('CVE', '20230116', '1622', 'Cenovus Energy (CVE) Gains But Lags Market: What You Should Know', 'Cenovus Energy '),
        #         ('AR', '20230116', '1622', 'Antero Resources (AR) Stock Sinks As Market Gains: What You Should Know', 'Antero Resources ')]
        #
        # logger.info(rows)
        #
        # self.cursor.executemany(
        #     "insert into HDBOWN.prediction_pn_us (ticker, date_, time_, headline, company_name) values (:1, :2, :3, :4, :5)",
        #     rows)
        # self.conn.commit()


        ## Mysql DB
        # self.engine = create_engine("mysql+pymysql://itbuser:" + "itbuser!23" + "@35.202.78.12:3306/itbdb?charset=utf8",
        #                             encoding='utf-8')
        # self.conn = self.engine.connect()
        # self.conn = self.conn.execution_options(autocommit=True)

        self.stock_data_dict = {'key': [], 'date': [], 'time': [], 'code': [], 'name': [], 'content': [], 'pn':[], 'ratio':[],
                         'close': [], 'diff':[], 'open': [], 'high': [], 'low': [], 'volume': [], 'gpt_pn': []}

        self.stock_data_detail_dict = {'key': [], 'date': [], 'code': [], 'ticker_desc1': [], 'ticker_desc2':[], 'sise_52_price':[],
                         'sise_revenue_rate': [], 'sise_siga_tot':[], 'sise_siga_tot2': [], 'sise_issue_stock_normal': [], 'toja_discision': [], 'toja_prop_price': [],
                         'toja_eps': [], 'toja_per':[], 'toja_comp': [], 'srim_revenue_rate': [], 'srim_jibea': [], 'srim_roa': [], 'srim_roe': [],
                         'srim_value': [], 'srim_issue_stock': [], 'srim_prop_price': [], 'srim_10_price': [], 'srim_20_price': []}

        self.total_article = []

        self.total_article_us = []
        self.get_article_usa()
        self.nlp_article_usa()

        self.get_article()
        self.negative_word_df = None
        self.get_negative_word()

        self.model = keras.models.load_model('./resource/pnmodel.h5', compile=False)
        self.train_data = self.read_data('./resource/ratings_train.txt')
        self.test_data = self.read_data('./resource/ratings_test.txt')
        self.okt = Okt()
        self.train_docs = None
        self.test_docs = None
        self.make_docs_from_json()
        self.tokens = [t for d in self.train_docs for t in d[0]]
        self.text = nltk.Text(self.tokens, name='NMSC')
        self.selected_words = [f[0] for f in self.text.vocab().most_common(5000)]

        self.stock_df = None
        self.get_stock_df()

        self.exec_predict_article()

        self.save_stock_data_to_mysql()

    ##### KOR Article #####
    def get_data_http(self, page_no):

        page = page_no
        url = 'https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date=' + self.date +'&page=' + page

        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.135 Safari/537.36'
        }
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup

    def get_article(self):
        articles = []

        for page_no in range(1, 10):
            soup = self.get_data_http(str(page_no))
            article_tag = soup.select_one('#contentarea_left')
            article_list_dd = article_tag.select('li > dl > dd')
            title_list_dd = self.make_content_list(article_list_dd)
            articles.append(title_list_dd)

            article_list_dt = article_tag.select('li > dl > dt')
            title_list_dt = self.make_content_list(article_list_dt)
            articles.append(title_list_dt)

        for article in articles:
            if article not in self.total_article:
                self.total_article.append(article)

    def make_content_list(self, article_list):
        title_list = []
        for content in article_list:
            try:
                title = content.select_one('a')['title']
                title_list.append(title)
            except:
                continue

        return title_list

    ##### USA Article #####
    def get_data_http_usa(self, page_no):

        # https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
        # https://blog.datahut.co/scraping-nasdaq-news-using-python/
        page = page_no
        url = 'https://www.nasdaq.com/news-and-insights/topic/markets/stocks/page/' + str(page)

        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.135 Safari/537.36'
        }
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup

    def get_article_usa(self):
        articles = []
        for page_no in range(1, 20):
            soup = self.get_data_http_usa(page_no)
            article_tag = soup.findAll('a', class_='content-feed__card-title-link')

            for article in article_tag:
                articles.append(article.text)

        for article in articles:
            try:
                company_name = article.split('(', 1)[0]
                ticker = article.split('(', 1)[1].split(')')[0]
                if len(ticker) > 10:
                    continue
                self.total_article_us.append([ticker, self.date, self.time, article, company_name])
            except:
                continue

        # logger.info(self.total_article_us)

    ##### USA NLP #####
    def nlp_article_usa(self):
        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()

        # Set column names
        columns = ['ticker', 'date_', 'time_', 'headline', 'company_name']

        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_and_scored_news = pd.DataFrame(self.total_article_us, columns=columns)

        # Iterate through the headlines and get the polarity scores using vader
        scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

        # Convert the 'scores' list of dicts into a DataFrame
        scores_df = pd.DataFrame(scores)

        # Join the DataFrames of the news and the list of dicts
        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

        # Convert the date column from string to datetime
        # parsed_and_scored_news['date_'] = pd.to_datetime(parsed_and_scored_news.date_).dt.date

        # logger.info(parsed_and_scored_news.info())

        # parsed_and_scored_news = parsed_and_scored_news[columns]

        # logger.info(parsed_and_scored_news.values.tolist())

        # parsed_and_scored_news.to_sql('prediction_pn_us', schema="HDBOWN", con=self.conn, if_exists='append', index=False)
        self.cursor.executemany("insert into HDBOWN.prediction_pn_us (ticker, date_, time_, headline, company_name, neg, neu, pos, compound) values (:1, :2, :3, :4, :5, :6, :7, :8, :9)", parsed_and_scored_news.values.tolist())
        self.conn.commit()


    ##### KOR NLP #####
    def get_negative_word(self):
        sql = 'select nag_word, lang from nag_word'
        self.negative_word_df = pd.read_sql(sql, self.conn)

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            # txt 파일의 헤더(id document label)는 제외하기
            data = data[1:]
        return data

    def tokenize(self, doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in self.okt.pos(doc, norm=True, stem=True)]

    def make_docs_from_json(self):
        if os.path.isfile('./resource/train_docs.json'):
            with open('./resource/train_docs.json', encoding='utf-8') as f:
                self.train_docs = json.load(f)
            with open('./resource/test_docs.json', encoding='utf-8') as f:
                self.test_docs = json.load(f)
        else:
            self.train_docs = [(self.tokenize(row[1]), row[2]) for row in self.train_data]
            self.test_docs = [(self.tokenize(row[1]), row[2]) for row in self.test_data]
            # JSON 파일로 저장
            with open('./resource/train_docs.json', 'w', encoding="utf-8") as make_file:
                json.dump(self.train_docs, make_file, ensure_ascii=False, indent="\t")
            with open('./resource/test_docs.json', 'w', encoding="utf-8") as make_file:
                json.dump(self.test_docs, make_file, ensure_ascii=False, indent="\t")

    def term_frequency(self, doc):
        return [doc.count(word) for word in self.selected_words]

    def predict_pos_neg(self, review):
        result_list = []
        token = self.tokenize(review)
        tf = self.term_frequency(token)
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
        score = float(self.model.predict(data))
        if (score > 0.5):
            #logger.info("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
            result_ratio = ('{:.2f}'.format(score * 100))
            #review 내용이 부정어에 포함되어 있으면 'N'으로 저장
            result_PN = self.apply_rule_keyword('P', review)
            result_list = [result_PN, result_ratio]
        else:
            #logger.info("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
            result_ratio = ('{:.2f}'.format((1 - score) * 100))
            # review 내용이 부정어에 포함되어 있지 않으면 'C'로 저장
            result_PN = self.apply_rule_keyword('N', review)
            result_list = [result_PN, result_ratio]

        return result_list

    def apply_rule_keyword(self, currentPN, content):
        str_result = ''
        if self.negative_word_df.size > 0:
            for i, row in self.negative_word_df.iterrows():
                nag_word = str(row['nag_word'])
                if currentPN == 'P':
                    if nag_word in content:
                        str_result = 'N'
                        break
                    else:
                        str_result = 'P'
                else:
                    if nag_word in content:
                        str_result = 'N'
                        break
                    else:
                        str_result = 'C'
        else:
            str_result = currentPN
        return str_result

    def get_stock_df(self):
        stock_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

        # 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
        stock_df.종목코드 = stock_df.종목코드.map('{:06d}'.format)
        # 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
        stock_df = stock_df[['회사명', '종목코드']]
        # 한글로된 컬럼명을 영어로 바꿔준다.
        self.stock_df = stock_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

    def cleanText(self, readData):
        # 텍스트에 포함되어 있는 특수 문자 제거

        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)

        return text

    def exec_predict_article(self):
        pre_article = ''
        for articles in self.total_article:
            for article in articles:
                selection_stock_df = self.stock_df[self.stock_df['name'].str.contains(self.cleanText(article.split(',')[0]))]
                #logger.info(article.split(',')[0])
                #logger.info(selection_stock_df)

                if not selection_stock_df.empty:
                    article = article.replace('-', '')

                    chatresult = ''

                    if pre_article != article:
                        time.sleep(65.0)  # ChatGPT API 호출 타임
                        # ChatGPT result
                        prompt = article + ' 이 문장이 긍정문이야? 부정문이야?'

                        # 파파고 번역 추가
                        prompt = self.trans_papago(prompt)

                        result_gpt_txt = self.chatGPT(prompt).strip()
                        result_gpt = 0
                        # 결과 값이 영문인지 한글인지 여부 체크
                        if self.is_korean(result_gpt_txt):
                            result_gpt = result_gpt_txt.find('부정')
                        elif self.is_english(result_gpt_txt):
                            result_gpt = result_gpt_txt.find('negative')
                        else:
                            result_gpt = -1

                        if result_gpt == -1:
                            chatresult = 'P'
                        else:
                            chatresult = 'N'

                    pre_article = article

                    self.stock_data_dict['gpt_pn'].append(chatresult)

                    logger.info(f'openai result : {chatresult}')

                    result_list = self.predict_pos_neg(article)
                    result_stock_df = selection_stock_df.head(1)

                    for i, row in result_stock_df.iterrows():
                        stock_code = row['code']
                        stock_name = row['name']

                    pn = result_list[0]
                    ratio = result_list[1]

                    stock_info_df = self.get_stock_info_df(stock_code)
                    for i, row in stock_info_df.iterrows():
                        close = row['close']
                        diff = row['diff']
                        open = row['open']
                        high = row['high']
                        low = row['low']
                        volume = row['volume']

                    now_day = datetime.now().strftime('%Y-%m-%d')
                    current_time = datetime.now().strftime('%H:%M')
                    self.stock_data_dict['key'].append(self.date + self.time + stock_code)
                    self.stock_data_dict['date'].append(now_day)
                    self.stock_data_dict['time'].append(current_time)
                    self.stock_data_dict['code'].append(stock_code)
                    self.stock_data_dict['name'].append(stock_name)
                    self.stock_data_dict['content'].append(article)
                    self.stock_data_dict['pn'].append(pn)
                    self.stock_data_dict['ratio'].append(float(ratio))

                    self.stock_data_dict['close'].append(int(close))
                    self.stock_data_dict['diff'].append(int(diff))
                    self.stock_data_dict['open'].append(int(open))
                    self.stock_data_dict['high'].append(int(high))
                    self.stock_data_dict['low'].append(int(low))
                    self.stock_data_dict['volume'].append(int(volume))

                    #logger.info(self.stock_data_dict)

                    self.get_stock_info_detail_kor(stock_code)

    def chatGPT(self, prompt, API_KEY=YOUR_API_KEY):

        str_decoded = cryptocode.decrypt(API_KEY, "openai")
        # set api key
        openai.api_key = str_decoded

        # Call the chat GPT API
        try:
            completion = openai.Completion.create(
                engine='text-davinci-003'  # 'text-curie-001'  # 'text-babbage-001' #'text-ada-001'
                , prompt=prompt
                , temperature=0.5
                , max_tokens=1024
                , top_p=1
                , frequency_penalty=0
                , presence_penalty=0)

            return completion['choices'][0]['text']
        except Exception as ex:
            logger.info('ChatGPT 에러가 발생 했습니다')
            return 'ChatGPT 에러가 발생 했습니다'

    def trans_papago(self, content):
        # 파파고 API URL
        url = "https://openapi.naver.com/v1/papago/n2mt"

        # 파라미터 설정
        data = {
            "source": "ko",
            "target": "en",
            "text": content
        }

        # Header 설정
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Naver-Client-Id": "KOfQjnVLH_96w5zeZOJN",
            "X-Naver-Client-Secret": "aVDbTdSfVP"
        }

        try:
            # POST 요청
            response = requests.post(url, data=data, headers=headers)

            # 결과 출력
            result = json.loads(response.text)
            translated_text = result['message']['result']['translatedText']
            logger.info(f'파파고 번역 : {translated_text}')
            return translated_text
        except Exception as ex:
            logger.info('파파고 번역시 에러가 발생 했습니다')
            return content

    def is_korean(self, text):
        for char in text:
            if '가' <= char <= '힣':
                return True
        return False

    def is_english(self, text):
        for char in text:
            if 'a' <= char.lower() <= 'z':
                return True
        return False

    def get_stock_info_df(self, code):
        date = datetime.now().strftime('%Y.%m.%d')
        df = pd.DataFrame()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page=1'.format(code=code)
        res = requests.get(url, headers=headers)
        _soap = BeautifulSoup(res.text, 'lxml')
        # logger.info(_soap)
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        df = _df.dropna()

        # 한글로 된 컬럼명을 영어로 바꿔줌
        df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low',
                                '거래량': 'volume'})
        # 데이터의 타입을 int형으로 바꿔줌
        df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

        # 컬럼명 'date'의 타입을 date로 바꿔줌
        df['date'] = pd.to_datetime(df['date'])
        # 일자(date)를 기준으로 오름차순 정렬
        df = df.sort_values(by=['date'], ascending=False)

        return df.head(1)

    def get_stock_info_detail_kor(self, code):
        ticker_desc1 = '-'
        ticker_desc2 = '-'
        sise_52_price = '-'
        sise_revenue_rate = '-'
        sise_siga_tot = '-'
        sise_siga_tot2 = '-'
        sise_issue_stock_normal = '-'
        toja_discision = '-'
        toja_prop_price = '-'
        toja_eps = '-'
        toja_per = '-'
        toja_comp = '-'
        srim_revenue_rate = '-'
        srim_jibea = '-'
        srim_roa = '-'
        srim_roe = '-'
        srim_value = '-'
        srim_issue_stock = '-'
        srim_prop_price = '-'
        srim_10_price = '-'
        srim_20_price = '-'

        C1, C2, C3, C4, C5 = 1, 1, 1, 1, 1

        try:
            url_rating = 'http://www.rating.co.kr/disclosure/QDisclosure029.do'
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/84.0.4147.135 Safari/537.36'
            }
            res = requests.get(url_rating, headers=headers)

            content_type = res.headers['content-type']
            if not 'charset' in content_type:
                res.encoding = res.apparent_encoding

            soup = BeautifulSoup(res.text, 'lxml')

            table = soup.findAll('table')
            read_html = str(table[0])
            # 요구수익율(C3) : BBB-등급 5년 채권 수익율
            dfs = pd.read_html(read_html)
            df = dfs[0]
            np_arr = df.loc[[10], ['5년']].values
            list_to_np = np_arr[0].tolist()
            C3 = float(list_to_np[0])
            srim_revenue_rate = C3
            logger.info(srim_revenue_rate)

        except Exception as ex:
            logger.info('상세1 데이터 에러가 발생 했습니다')


        try:

            url_ticker = 'http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A'+code+'&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701'
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/84.0.4147.135 Safari/537.36'
            }
            res = requests.get(url_ticker, headers=headers)
            content_type = res.headers['content-type']
            if not 'charset' in content_type:
                res.encoding = res.apparent_encoding

            soup = BeautifulSoup(res.text, 'lxml')

            stxt1_tag = soup.find("span", attrs={"class": "stxt stxt1"})
            ticker_desc1 = stxt1_tag.text
            logger.info(ticker_desc1)

            stxt2_tag = soup.find("span", attrs={"class": "stxt stxt2"})
            ticker_desc2 = stxt2_tag.text
            logger.info(ticker_desc2)

            # 시세현황
            table = soup.findAll('table')
            read_html = str(table[0])
            dfs = pd.read_html(read_html)
            df = dfs[0]

            np_arr = df.loc[[1], :].values
            list_to_np = np_arr[0].tolist()
            sise_52_price = str(list_to_np[1])
            logger.info(sise_52_price)

            np_arr = df.loc[[2], :].values
            list_to_np = np_arr[0].tolist()
            sise_revenue_rate = str(list_to_np[1])
            logger.info(sise_revenue_rate)

            np_arr = df.loc[[3], :].values
            list_to_np = np_arr[0].tolist()
            sise_siga_tot = str(format(int(list_to_np[1]), ','))
            logger.info(sise_siga_tot)

            np_arr = df.loc[[4], :].values
            list_to_np = np_arr[0].tolist()
            sise_siga_tot2 = str(format(int(list_to_np[1]), ','))
            logger.info(sise_siga_tot2)

            np_arr = df.loc[[6], :].values
            list_to_np = np_arr[0].tolist()
            sise_issue_stock_normal = str(list_to_np[1])
            logger.info(sise_issue_stock_normal)

            # 투자의견컨센서스
            table = soup.findAll('table')
            read_html = str(table[7])
            dfs = pd.read_html(read_html)
            df = dfs[0]

            np_arr = df.loc[[0], :].values
            list_to_np = np_arr[0].tolist()
            toja_discision = str(list_to_np[0])
            logger.info(toja_discision)

            toja_prop_price = str(format(int(list_to_np[1]), ','))
            logger.info(toja_prop_price)

            toja_eps = str(format(int(list_to_np[2]), ','))
            logger.info(toja_eps)

            toja_per = str(list_to_np[3])
            logger.info(toja_per)

            toja_comp = str(int(list_to_np[4]))
            logger.info(toja_comp)

            # SRIM
            table = soup.findAll('table')
            read_html = str(table[10])
            dfs = pd.read_html(read_html)
            df = dfs[0]
            # logger.info(df)
            # 지배주주 지분(C4) : Annuel 기준 전년도 말 값 (리스트 값 중 3번째 값)
            np_arr = df.loc[[9], :].values
            list_to_np = np_arr[0].tolist()
            C4 = float(list_to_np[4])
            srim_jibea = str(format(float(list_to_np[4]), ','))
            logger.info(srim_jibea)

            # ROA(C5) : Annuel 기준 가장 마지막 값으로 함
            np_arr = df.iloc[[16], 1:5].values
            list_to_np = np_arr[0].tolist()
            ROA = float(list_to_np[-1])
            srim_roa = ROA
            logger.info(srim_roa)

            # ROE(C5) : Annuel 기준 가장 마지막 값으로 함
            np_arr = df.iloc[[17], 1:5].values
            list_to_np = np_arr[0].tolist()
            # logger.info(list_to_np)
            ROE = float(list_to_np[-1])
            C5 = float(list_to_np[-1])
            srim_roe = ROE
            logger.info(srim_roe)

            # ROA와 ROE차가 5%이상이면 ROA로 계산함
            RO_Compare = ROE - ROA
            if RO_Compare > 5:
                C5 = ROA
            else:
                C5 = ROE

            # 기업가치 : (C4+(C4*(C5-C3)/C3))*100000000
            company_value = (C4 + (C4 * (C5 - C3) / C3)) * 100000000
            srim_value = str(format(float(company_value), ','))
            logger.info(srim_value)

            # 발행주식수
            issue_stock_val = sise_issue_stock_normal.split("/")[0]
            issue_stock_cnt = int(issue_stock_val.replace(",", ""))

            # 자기주식수
            read_html = str(table[4])
            dfs = pd.read_html(read_html)
            df = dfs[0]
            np_arr = df.loc[[4], ['보통주']].values
            list_to_np = np_arr[0].tolist()
            if str(list_to_np[0]) == 'nan':
                self_stock_cnt = 0
            else:
                self_stock_cnt = int(list_to_np[0])

            # 발행주식수(보통주) - 자기주식(보통주)수
            ticker_issue_stock_cnt = issue_stock_cnt - self_stock_cnt
            srim_issue_stock = str(format(int(ticker_issue_stock_cnt), ','))
            logger.info(srim_issue_stock)

            # 적정가격
            price_hope = company_value / ticker_issue_stock_cnt
            srim_prop_price = str(format(int(price_hope), ','))
            logger.info(srim_prop_price)

            price_over = C4 * (C5 - C3) / 100

            down_rate_9 = (C4 + price_over * (0.9 / (1 + (C3 / 100) - 0.9))) * 100000000
            down_rate_8 = (C4 + price_over * (0.8 / (1 + (C3 / 100) - 0.8))) * 100000000

            srim_10_price = down_rate_9 / ticker_issue_stock_cnt
            srim_20_price = down_rate_8 / ticker_issue_stock_cnt

            if price_over < 0:
                srim_10_price = price_hope - (srim_10_price - price_hope)
                srim_20_price = price_hope - (srim_20_price - price_hope)

            srim_10_price = str(format(int(srim_10_price), ','))
            srim_20_price = str(format(int(srim_20_price), ','))

            logger.info(srim_10_price)
            logger.info(srim_20_price)
        except Exception as ex:
            logger.info('상세2 데이터 에러가 발생 했습니다')

        self.stock_data_detail_dict['key'].append(self.date + code)
        self.stock_data_detail_dict['date'].append(self.date)
        self.stock_data_detail_dict['code'].append(code)
        self.stock_data_detail_dict['ticker_desc1'].append(str(ticker_desc1))
        self.stock_data_detail_dict['ticker_desc2'].append(str(ticker_desc2))
        self.stock_data_detail_dict['sise_52_price'].append(str(sise_52_price))
        self.stock_data_detail_dict['sise_revenue_rate'].append(str(sise_revenue_rate))
        self.stock_data_detail_dict['sise_siga_tot'].append(str(sise_siga_tot))
        self.stock_data_detail_dict['sise_siga_tot2'].append(str(sise_siga_tot2))
        self.stock_data_detail_dict['sise_issue_stock_normal'].append(str(sise_issue_stock_normal))
        self.stock_data_detail_dict['toja_discision'].append(str(toja_discision))
        self.stock_data_detail_dict['toja_prop_price'].append(str(toja_prop_price))
        self.stock_data_detail_dict['toja_eps'].append(str(toja_eps))
        self.stock_data_detail_dict['toja_per'].append(str(toja_per))
        self.stock_data_detail_dict['toja_comp'].append(str(toja_comp))
        self.stock_data_detail_dict['srim_revenue_rate'].append(str(srim_revenue_rate))
        self.stock_data_detail_dict['srim_jibea'].append(str(srim_jibea))
        self.stock_data_detail_dict['srim_roa'].append(str(srim_roa))
        self.stock_data_detail_dict['srim_roe'].append(str(srim_roe))
        self.stock_data_detail_dict['srim_value'].append(str(srim_value))
        self.stock_data_detail_dict['srim_issue_stock'].append(str(srim_issue_stock))
        self.stock_data_detail_dict['srim_prop_price'].append(str(srim_prop_price))
        self.stock_data_detail_dict['srim_10_price'].append(str(srim_10_price))
        self.stock_data_detail_dict['srim_20_price'].append(str(srim_20_price))

    def save_stock_data_to_mysql(self):
        df = pd.DataFrame(self.stock_data_dict, columns=['key', 'date', 'time', 'code', 'name', 'content', 'pn', 'ratio', 'close', 'diff', 'open', 'high', 'low', 'volume', 'gpt_pn'],
                          index=self.stock_data_dict['key'])
        # df.to_sql(name='prediction_pn', con=self.engine, if_exists='append', index=False)

        self.cursor.executemany(
            "insert into HDBOWN.prediction_pn (key_, date_, time_, code_, name_, content_, pn_, ratio_, close_, diff_, open_, high_, low_, volume_, gpt_pn_) values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15)", df.values.tolist())
        self.conn.commit()

        df2 = pd.DataFrame(self.stock_data_detail_dict,
                          columns=['key', 'date', 'code', 'ticker_desc1', 'ticker_desc2', 'sise_52_price', 'sise_revenue_rate', 'sise_siga_tot', 'sise_siga_tot2', 'sise_issue_stock_normal', 'toja_discision',
                                   'toja_prop_price', 'toja_eps', 'toja_per', 'toja_comp', 'srim_revenue_rate', 'srim_jibea', 'srim_roa', 'srim_roe', 'srim_value', 'srim_issue_stock', 'srim_prop_price', 'srim_10_price', 'srim_20_price'],
                          index=self.stock_data_detail_dict['key'])
        # df2.to_sql(name='prediction_detail_ko', con=self.engine, if_exists='append', index=False)
        df2 = df2.fillna('')
        self.cursor.executemany(
            "insert into HDBOWN.prediction_detail_ko (key_, date_, code_, ticker_desc1, ticker_desc2, sise_52_price, sise_revenue_rate, sise_siga_tot, sise_siga_tot2, sise_issue_stock_normal, toja_discision, toja_prop_price, toja_eps, toja_per, toja_comp, srim_revenue_rate, srim_jibea, srim_roa, srim_roe, srim_value, srim_issue_stock, srim_prop_price, srim_10_price, srim_20_price) values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21, :22, :23, :24)", df2.values.tolist())
        self.conn.commit()
        logger.info('DB저장 완료')


import time, traceback


def every(delay, task):
  next_time = time.time() + delay
  while True:
    time.sleep(max(0, next_time - time.time()))
    try:
      task()
    except Exception:
      traceback.print_exc()
      # in production code you might want to have this instead of course:
      # logger.exception("Problem while executing repetitive task.")
    # skip tasks if we are behind schedule:
    next_time += (time.time() - next_time) // delay * delay + delay


def play():
    logger.info("play to StockPNModel!")
    Main()


play()

# every(60*60*1, play)



