import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

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

import platform

import math

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

YOUR_API_KEY = 'kyUZPxnXbxK9jCuS6JM4F8hUlZ9b/L5Eyy8gEwKnY69h+ckQ9a7nIKidJ+e+FqdMaXvU*tJZDdp20DvfKJhXMK5x8qQ==*E7T61eyVsy1lSf7PFlPTrg==*F2SUy7OVdpa4tU9bHvSOhA=='

os_type = platform.system()
IS_GPT = True

class Main():

    def __init__(self):
        '''
        초기화
        '''

        # 날짜, 시간, 종목코드, 종목명, content, 긍정or부정, 확률, 종가, 전일비, 시가, 고가, 저가, 거래량
        self.date = datetime.now().strftime('%Y%m%d')
        self.time = datetime.now().strftime('%H%M')

        logger.info("OS Type : " + os_type)
        logger.info(f"실행시작시간 : {self.date} {self.time}")

        ###################### Oracle cloud DB #############################################################################
        if os_type == 'Windows':
            cx_Oracle.init_oracle_client(lib_dir=r".\resource\instantclient_19_17")
        else:
            cx_Oracle.init_oracle_client(lib_dir="/usr/lib/oracle/21/client64/lib")

        # openai call sleep time 설정
        if os_type == 'Windows':
            self.sleep_time = 0
        else:
            self.sleep_time = 65

        self.conn = cx_Oracle.connect(user='HDBOWN', password='Qwer1234!@#$', dsn='ppanggoodoracledb_high')
        self.cursor = self.conn.cursor()

        self.stock_data_dict = {'key': [], 'date': [], 'time': [], 'code': [], 'name': [], 'content': [], 'pn':[], 'ratio':[],
                         'close': [], 'diff':[], 'open': [], 'high': [], 'low': [], 'volume': [], 'gpt_pn': [], 'report': []}

        self.stock_data_detail_dict = {'key': [], 'date': [], 'code': [], 'ticker_desc1': [], 'ticker_desc2':[], 'sise_52_price':[],
                         'sise_revenue_rate': [], 'sise_siga_tot':[], 'sise_siga_tot2': [], 'sise_issue_stock_normal': [], 'toja_discision': [], 'toja_prop_price': [],
                         'toja_eps': [], 'toja_per':[], 'toja_comp': [], 'srim_revenue_rate': [], 'srim_jibea': [], 'srim_roa': [], 'srim_roe': [],
                         'srim_value': [], 'srim_issue_stock': [], 'srim_prop_price': [], 'srim_10_price': [], 'srim_20_price': []}

        self.total_article = []

        ## USA 분석 Part ####################################################
        self.total_article_us = []
        self.get_article_usa()
        self.nlp_article_usa()

        ## KOR 분석 Part ####################################################
        articleCnt = 1
        self.get_article(self.date)
        while True:
            if len(self.total_article) < 20:
                now = datetime.now()
                one_day_ago = now - timedelta(days=articleCnt)
                self.get_article(one_day_ago.strftime('%Y%m%d'))
                articleCnt += 1
            else:
                break

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

    def get_data_http(self, date, page_no):
        '''
        (KOR)
        네이버 뉴스 기사 가져오기
        :param page_no:
        :return:
        '''

        page = page_no
        url = 'https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date=' + date +'&page=' + page

        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.135 Safari/537.36'
        }
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup

    def get_article(self, date):
        '''
        (KOR)
        네이버기사 html 파싱
        :return:
        '''

        articles = []

        for page_no in range(1, 10):

            soup = self.get_data_http(date, str(page_no))
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
        '''
        (KOR)
        네이버기사 article에서 title 추출
        :param article_list:
        :return:
        '''

        title_list = []
        for content in article_list:
            try:
                title = content.select_one('a')['title']
                title_list.append(title)
            except:
                continue

        return title_list

    def get_data_http_usa(self, page_no):
        '''
        USA 기사 가져오기
        :param page_no:
        :return:
        '''

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
        '''
        USA 기사 html 파싱
        :return:
        '''

        articles = []
        for page_no in range(1, 5):
            try:
                soup = self.get_data_http_usa(page_no)
                article_tag = soup.findAll('a', class_='content-feed__card-title-link')

                for article in article_tag:
                    articles.append(article.text)
            except:
                continue

        for article in articles:
            try:
                ticker = article.split('(', 1)[1].split(')')[0]
                if len(ticker) > 4:
                    continue

                # 해당코드가 이미 분석한 결과가 있는지 여부 체크
                if self.is_exists(ticker, 'US'):
                    continue

                self.get_company_info_usa(ticker, article)
                if os_type == 'Windows':
                    break

            except:
                continue

    def get_company_info_usa(self, ticker, article):
        '''
        USA ticker에 해당하는 기업 정보 가져오기
        :param ticker:
        :return:
        '''
        url = "https://www.nasdaq.com/market-activity/stocks/{}".format(ticker.lower())
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/84.0.4147.135 Safari/537.36'
        }
        res = requests.get(url, headers=headers)
        content_type = res.headers['content-type']
        if not 'charset' in content_type:
            res.encoding = res.apparent_encoding

        soup = BeautifulSoup(res.text, 'lxml')

        company_name = soup.find("span", attrs={"class": "symbol-page-header__name"}).text.strip()

        # price = soup.find('div', {'class': 'symbol-page-header__pricing-price'}).text.strip()
        # company_info = soup.find('div', {'class': 'symbol-page-header__description'}).text.strip()

        logger.info("Company: {}".format(company_name))
        report = ''
        report = self.get_exists_report(ticker, 'US')
        if len(report) == 0:
            if IS_GPT:
                report = self.get_company_report_usa(ticker, company_name)
                logger.info("Summary-GPT: {}".format(report))
            else:
                report = ''
        else:
            logger.info("Summary-DB: {}".format(report))
        self.total_article_us.append([ticker, self.date, self.time, article, company_name, report])

    def truncate_sentence(self, sentence: str, num_tokens: int) -> str:
        '''
        주어진 토큰 수로 자르기
        :param sentence:
        :param num_tokens:
        :return:
        '''
        # Split the sentence into tokens
        tokens = sentence.split()

        # Check if the sentence is already shorter than the desired length
        if len(tokens) <= num_tokens:
            return sentence

        # Truncate the sentence by removing the last tokens
        truncated_tokens = tokens[:num_tokens]

        # Join the truncated tokens back into a sentence
        truncated_sentence = ' '.join(truncated_tokens)

        # Add ellipsis if the original sentence was longer than the truncated sentence
        if len(tokens) > num_tokens:
            truncated_sentence += '...'

        return truncated_sentence

    def get_company_report_usa(self, ticker, company_name):
        '''
        USA 기업 리포트 요약
        :param ticker:
        :return:
        '''
        try:
            # 보고서 종류와 url을 지정합니다.
            report_type = "10-K"
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={report_type}&dateb=&owner=exclude&count=40"
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/84.0.4147.135 Safari/537.36'
            }
            # url에서 보고서 링크를 가져옵니다.
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"class": "tableFile2"})
            rows = table.findAll("tr")
            document_link = ""
            for row in rows:
                cells = row.findAll("td")
                if len(cells) > 3 and report_type in cells[0].text:
                    document_link = "https://www.sec.gov" + cells[1].a["href"]
                    break

            # 보고서 링크에서 텍스트 데이터를 가져옵니다.
            response = requests.get(document_link, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"class": "tableFile"})
            rows = table.findAll("tr")
            document_link = ""
            for row in rows:
                cells = row.findAll("td")
                find_word = 'ix?doc=/'
                if len(cells) > 1 and report_type in cells[1].text:
                    document_link = "https://www.sec.gov" + cells[2].a["href"]
                    document_link = document_link.replace(find_word, '')
                    break

            response = requests.get(document_link, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            document = soup.find("body").get_text()

            # 텍스트 전처리를 수행합니다.
            document = re.sub(r"\n", " ", document)  # 개행문자 제거
            document = re.sub(r"\s+", " ", document)  # 여러 개의 공백을 하나의 공백으로 변경

            delimiter = 'FORM 10-K'
            document_parts = document.split(delimiter)
            document = document_parts[1]
            # 가져온 텍스트 데이터를 출력합니다.
            # print(text_data)

            document = re.sub('[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ\\s]+', '', document)

            # Set the maximum sequence length
            max_length = 1024

            # Decode the tokenized input
            modified_text = self.truncate_sentence(document, max_length)
            prompt = f"The following text is part of the {company_name} report. Please summarize the main business aspects of this company using bullet points in 5 to 7 sentences or less:\n\n" + modified_text
            result = self.chatGPT(prompt, YOUR_API_KEY)
            result_list = result.split('\n\n')
            if len(result_list) > 0:
                return result_list[-1].strip()
            else:
                return result
        except Exception as ex:
            logger.info(f'unable to summarize the company report. : {ex}')
            return 'unable to summarize the company report.'


    def nlp_article_usa(self):
        '''
        USA 기사 감성분석
        :return:
        '''

        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()

        # Set column names
        columns = ['ticker', 'date_', 'time_', 'headline', 'company_name', 'report_']

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
        self.cursor.executemany("insert into HDBOWN.prediction_pn_us (ticker, date_, time_, headline, company_name, report_, neg, neu, pos, compound) values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)", parsed_and_scored_news.values.tolist())
        self.conn.commit()
        logger.info(f'USA DB SAVE : {len(parsed_and_scored_news)}')


    def get_negative_word(self):
        '''
        (KOR)
        DB에 등록된 부정어 가져오기(현재 사용하지 않음)
        :return:
        '''

        sql = 'select nag_word, lang from nag_word'
        self.negative_word_df = pd.read_sql(sql, self.conn)

    def read_data(self, filename):
        '''
        (KOR)
        영화리뷰 감성분석 모델 활용을 위한 전처리
        :param filename:
        :return:
        '''
        with open(filename, 'r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            # txt 파일의 헤더(id document label)는 제외하기
            data = data[1:]
        return data

    def tokenize(self, doc):
        '''
        (KOR)
        영화리뷰 감성분석 모델 활용을 위한 전처리
        :param doc:
        :return:
        '''
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in self.okt.pos(doc, norm=True, stem=True)]

    def make_docs_from_json(self):
        '''
        (KOR)
        영화리뷰 감성분석 모델 활용을 위한 전처리
        :return:
        '''
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
        '''
        (KOR)
        영화리뷰 감성분석 모델 활용을 위한 전처리
        :param doc:
        :return:
        '''
        return [doc.count(word) for word in self.selected_words]

    def predict_pos_neg(self, review):
        '''
        (KOR)
        영화리뷰 감성분석 모델 활용 감성 분석 실행
        :param review:
        :return:
        '''
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
        '''
        (KOR)
        영화리뷰 감성분석 모델 결과로 정의된 부정어로 긍/부정 후처리(현재 사용하지 않음)
        :param currentPN:
        :param content:
        :return:
        '''
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
        '''
        (KOR)
        주식 종목 정보 가져오기
        :return:
        '''

        stock_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

        # 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
        stock_df.종목코드 = stock_df.종목코드.map('{:06d}'.format)
        # 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
        stock_df = stock_df[['회사명', '종목코드']]
        # 한글로된 컬럼명을 영어로 바꿔준다.
        self.stock_df = stock_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

    def cleanText(self, readData):
        '''
        (KOR)
        텍스트에 포함되어 있는 특수 문자 제거
        :param readData:
        :return:
        '''
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)

        return text

    def exec_predict_article(self):
        '''
        (KOR)
        분석 메인 Function
        '''

        pre_article = ''
        stock_code = ''
        stock_name = ''
        close = 0
        diff = 0
        open = 0
        high = 0
        low = 0
        volume = 0

        for articles in self.total_article:
            for article in articles:
                selection_stock_df = self.stock_df[self.stock_df['name'].str.contains(self.cleanText(article.split(',')[0]))]

                if not selection_stock_df.empty:
                    result_stock_df = selection_stock_df[selection_stock_df['name'] == self.cleanText(article.split(',')[0])]

                    for i, row in result_stock_df.iterrows():
                        stock_code = row['code']
                        stock_name = row['name']

                    # 해당코드가 이미 분석한 결과가 있는지 여부 체크
                    if self.is_exists(stock_code, 'KO'):
                        continue

                    article = article.replace('-', '')

                    ## chatGPT 감성분석 part
                    chatresult = ''

                    if IS_GPT:
                        # ChatGPT result
                        prompt = article + ' 이 문장이 긍정문이야? 부정문이야?'

                        # 파파고 번역 추가
                        prompt = self.trans_papago(prompt)

                        result_gpt_txt = self.chatGPT(prompt).strip()

                        # 긍/부정 결과 판단
                        chatresult = self.get_result_pn(result_gpt_txt)
                    else:
                        chatresult = self.get_result_pn(article)

                    self.stock_data_dict['gpt_pn'].append(chatresult)

                    logger.info(f'openai result : {chatresult}')

                    ## 영화리뷰 모델 활용 감성분석 part
                    result_list = self.predict_pos_neg(article)

                    ## 분석 결과 후처리
                    pn = result_list[0]
                    ratio = result_list[1]

                    stock_info_df = self.get_stock_info_df(stock_code)
                    for i, row in stock_info_df.iterrows():
                        close = (lambda x: 0 if x is None or x == 0 else x)(row['close'])
                        diff = (lambda x: 0 if x is None or x == 0 else x)(row['diff'])
                        open = (lambda x: 0 if x is None or x == 0 else x)(row['open'])
                        high = (lambda x: 0 if x is None or x == 0 else x)(row['high'])
                        low = (lambda x: 0 if x is None or x == 0 else x)(row['low'])
                        volume = (lambda x: 0 if x is None or x == 0 else x)(row['volume'])

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

                    self.get_stock_info_detail_kor(stock_code, stock_name)

    def chatGPT(self, prompt, API_KEY=YOUR_API_KEY):
        '''
        (KOR)
        ChatGPT 감석분석 실행
        :param prompt:
        :param API_KEY:
        :return:
        '''
        time.sleep(self.sleep_time)  # ChatGPT API 호출 타임
        str_decoded = cryptocode.decrypt(API_KEY, "openai")
        # set api key
        openai.api_key = str_decoded

        # Call the chat GPT API
        try:
            completion = openai.Completion.create(
                engine='text-davinci-003'  # 'text-curie-001'  # 'text-babbage-001' #'text-ada-001'
                , prompt=prompt
                , temperature=0.5
                , max_tokens=2096
                , top_p=0.5
                , frequency_penalty=0
                , presence_penalty=0)

            return completion['choices'][0]['text']
        except Exception as ex:
            logger.info('ChatGPT 에러가 발생 했습니다')
            return 'ChatGPT 에러가 발생 했습니다'

    def trans_papago(self, content):
        '''
        (KOR)
        파파고 번역 실행
        :param content:
        :return: 파파고 일 번역 한도 초과시 content 처리 없이 그대로 return
        '''

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
        '''
        (KOR)
        문장이 한글인지 여부 체크
        :param text:
        :return:
        '''
        for char in text:
            if '가' <= char <= '힣':
                return True
        return False

    def is_english(self, text):
        '''
        (KOR)
        문장이 영문인지 여부 체크
        :param text:
        :return:
        '''
        for char in text:
            if 'a' <= char.lower() <= 'z':
                return True
        return False

    def is_exists(self, code, flag):
        '''
        (KOR)
        해당코드가 이미 DB에 존재하고 있는지 여부 체크
        :param code:
        :return:
        '''

        sql = ''
        if flag == 'KO':
            sql = f"select count(*) as cnt from HDBOWN.prediction_pn where code_ = '{code}' and date_ = to_char(to_date('{self.date}', 'YYYYMMDD'), 'YYYY-MM-DD')"
        else:
            sql = f"select count(*) as cnt from HDBOWN.prediction_pn_us where ticker = '{code}' and date_ = to_char(to_date('{self.date}', 'YYYYMMDD'), 'YYYYMMDD')"

        logger.info(f'already exists check sql : {sql}')
        result_df = pd.read_sql(sql, self.conn)
        if result_df.iloc[0, 0] >= 1:
            logger.info(f'code that already exists : {code}')
            return True
        else:
            logger.info(f'run analysis with new code : {code}')
            return False

    def get_exists_report(self, code, flag):
        '''
        코드에 해당하는 레포트 정보가 있으면 DB정보로 저장
        :param code:
        :param flag:
        :return:
        '''
        sql = ''
        if flag == 'KO':
            sql = f"select max(report_) from HDBOWN.prediction_pn where code_ = '{code}' "
        else:
            sql = f"select max(report_) from HDBOWN.prediction_pn_us where ticker = '{code}' "
        result_df = pd.read_sql(sql, self.conn)
        if len(result_df) > 0:
            return result_df.iloc[0, 0]
        else:
            return ''

    def get_result_pn(self, sentence):
        '''
        (KOR)
        감성 분석 결과에 대한 문장으로 긍/부정 판단
        :param sentence:
        :return:
        '''
        # statement word
        s_word_list_kor_type1 = ['긍정', '부정']
        s_word_list_kor_type2 = ['사실', '진술']
        s_word_list_eng_type1 = ['positive', 'negative']
        s_word_list_eng_type2 = ['statement', 'fact']

        chk_statement = False

        chatresult = None

        result_gpt = 0

        # 결과 값이 영문인지 한글인지 여부 체크
        if self.is_korean(sentence):

            if all(word in sentence for word in s_word_list_kor_type1):
                chk_statement = True
            else:
                for word in s_word_list_kor_type2:
                    if word in sentence:
                        chk_statement = True

            if chk_statement:
                result_gpt = 0
            else:
                result_gpt = sentence.find('부정')

        elif self.is_english(sentence):

            if all(word in sentence for word in s_word_list_eng_type1):
                chk_statement = True
            else:
                for word in s_word_list_eng_type2:
                    if word in sentence:
                        chk_statement = True

            if chk_statement:
                result_gpt = 0
            else:
                result_gpt = sentence.find('negative')
        else:
            result_gpt = 0

        if result_gpt == -1:
            chatresult = 'P'
        elif result_gpt >= 1:
            chatresult = 'N'
        else:
            chatresult = 'P'

        return chatresult

    def get_stock_info_df(self, code):
        '''
        (KOR)
        분석 결과 종목에 대한 현재 주가 정보 불러오기
        :param code:
        :return:
        '''
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

    def get_stock_info_detail_kor(self, code, name):
        '''
        (KOR)
        분석 결과 종목에 대한 상세 정보 처리
        :param code:
        :return:
        '''
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
        result_gpt_txt = ''

        C1, C2, C3, C4, C5 = 1, 1, 1, 1, 1

        # 요구수익율 구하기 : BBB-등급 5년 채권 수익율 보다는 높은 수익율 기대치 반영
        try:
            url_rating = 'https://www.kisrating.co.kr/ratingsStatistics/statics_spread.do'
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
            logger.info(f'BBB-등급 5년 채권 수익율 : {srim_revenue_rate}')

        except Exception as ex:
            logger.info(f'BBB-등급 5년 채권 수익율 가져오기 실패!! : {ex}')
            C3 = 20.0

        # 기업 상세 정보
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
        except Exception as ex:
            logger.info(f'기업 상세 자료 가져오기 실패!! : {ex}')

        try:
            # 시세현황
            table = soup.findAll('table')
            read_html = str(table[0])
            dfs = pd.read_html(read_html)
            df = dfs[0]

            np_arr = df.loc[[1], :].values
            list_to_np = np_arr[0].tolist()
            sise_52_price = str(list_to_np[1])

            np_arr = df.loc[[2], :].values
            list_to_np = np_arr[0].tolist()
            sise_revenue_rate = str(list_to_np[1])

            np_arr = df.loc[[3], :].values
            list_to_np = np_arr[0].tolist()
            sise_siga_tot = str(format(int(list_to_np[1]), ','))

            np_arr = df.loc[[4], :].values
            list_to_np = np_arr[0].tolist()
            sise_siga_tot2 = str(format(int(list_to_np[1]), ','))

            np_arr = df.loc[[6], :].values
            list_to_np = np_arr[0].tolist()
            sise_issue_stock_normal = str(list_to_np[1])
            logger.info(f'시세현황 : {sise_issue_stock_normal}')
        except Exception as ex:
            logger.info(f'시세현황 에러!! : {ex}')

        try:
            # 투자의견컨센서스
            table = soup.findAll('table')
            read_html = str(table[7])
            dfs = pd.read_html(read_html)
            df = dfs[0]

            np_arr = df.loc[[0], :].values
            list_to_np = np_arr[0].tolist()
            toja_discision = str(list_to_np[0])
            toja_prop_price = str(format(int(list_to_np[1]), ','))
            toja_eps = str(format(int(list_to_np[2]), ','))
            toja_per = str(list_to_np[3])
            toja_comp = str(int(list_to_np[4]))
            logger.info(f'투자의견컨센서스 : {toja_discision}')
        except Exception as ex:
            logger.info(f'투자의견컨센서스 에러!! : {ex}')

        try:
            # SRIM
            table = soup.findAll('table')
            read_html = str(table[10])
            dfs = pd.read_html(read_html)
            df = dfs[0]
            # logger.info(df)
            # 지배주주 지분(C4) : Annuel 기준 전년도 말 값 (리스트 값 중 3번째 값)
            np_arr = df.loc[[9], :].values
            list_to_np = np_arr[0].tolist()
            C4 = float(self.util_get_array(list_to_np))
            srim_jibea = str(format(C4, ','))

            # ROA(C5) : Annuel 기준 가장 마지막 값으로 함
            np_arr = df.iloc[[16], 1:5].values
            list_to_np = np_arr[0].tolist()
            ROA = float(self.util_get_array(list_to_np))
            srim_roa = ROA

            # ROE(C5) : Annuel 기준 가장 마지막 값으로 함
            np_arr = df.iloc[[17], 1:5].values
            list_to_np = np_arr[0].tolist()
            # logger.info(list_to_np)
            ROE = float(self.util_get_array(list_to_np))
            C5 = ROE
            srim_roe = ROE

            # ROA와 ROE차가 5%이상이면 ROA로 계산함
            RO_Compare = ROE - ROA
            if RO_Compare > 5:
                C5 = ROA
            else:
                C5 = ROE

            # 기업가치 : (C4+(C4*(C5-C3)/C3))*100000000
            company_value = (C4 + (C4 * (ROE - C3) / C3)) * 100000000
            srim_value = str(format(float(company_value), ','))

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

            # 적정가격
            price_hope = company_value / ticker_issue_stock_cnt
            srim_prop_price = str(format(int(price_hope), ','))
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

            logger.info(f'SRIM 계산 : {srim_prop_price}')
        except Exception as ex:
            logger.info(f'SRIM 계산 에러!! : {ex}')

        result_gpt_txt = ''
        result_gpt_txt = self.get_exists_report(code, 'KO')
        if len(result_gpt_txt) == 0:
            if IS_GPT:
                try:
                    # 기업정보Report
                    object1 = soup.find("div", attrs={"class": "um_bssummary"})
                    value1 = object1.find('h3')
                    value2 = object1.find('li')

                    # ChatGPT result
                    prompt = f"Please summarize the following text using bullet points:\n\n 기업명 : {name}\n\n 기업개요 : {value2}\n\n 현재상황 : {value1}"
                    logger.info(f'기업정보Report Original: {prompt}')
                    word_to_check = '동사는'
                    if word_to_check in prompt:
                        prompt = prompt.replace(word_to_check, '')
                    result_gpt_txt = self.chatGPT(prompt).strip()

                    # 문장 구분을 위한 패턴
                    sentence_pattern = re.compile(r'.+?[.?!]')

                    # 문단을 문장으로 분리
                    sentences = sentence_pattern.findall(result_gpt_txt)
                    result_gpt_txt = sentences[0].strip()

                    logger.info(f'기업정보Report GPT: {result_gpt_txt}')
                except Exception as ex:
                    logger.info(f'기업정보Report 에러!! : {ex}')
            else:
                result_gpt_txt = ''
        else:
            logger.info(f'기업정보Report DB: {result_gpt_txt}')

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
        self.stock_data_dict['report'].append(str(result_gpt_txt))

    def save_stock_data_to_mysql(self):
        '''
        (KOR)
        분석결과 Oralce Cloud DB에 저장
        :return:
        '''
        try:
            df = pd.DataFrame(self.stock_data_dict, columns=['key', 'date', 'time', 'code', 'name', 'content', 'pn', 'ratio', 'close', 'diff', 'open', 'high', 'low', 'volume', 'gpt_pn', 'report'],
                              index=self.stock_data_dict['key'])
            # df.to_sql(name='prediction_pn', con=self.engine, if_exists='append', index=False)

            self.cursor.executemany(
                "insert into HDBOWN.prediction_pn (key_, date_, time_, code_, name_, content_, pn_, ratio_, close_, diff_, open_, high_, low_, volume_, gpt_pn_, report_) values (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16)", df.values.tolist())
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
        except Exception as ex:
            logger.info(f'DB저장 에러!! : {ex}')

    def util_get_array(self, array):
        '''
        파이썬 역순으로 순회하여 Nan이 아닌 값 찾기
        :param array:
        :return:
        '''
        return_value = 0
        for value in reversed(array):
            if value is not None and not math.isnan(value):
                return_value = value
                break
        return return_value


## Main #########################################################################################
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



