import requests
from bs4 import BeautifulSoup

import openai
import argparse
import re
import cryptocode

from transformers import AutoTokenizer, AutoModelForCausalLM

YOUR_API_KEY = '85Qs4r6Sa+pJp5zE/t7g7cOOg84nQ9jSw4I9ncRF4EWdt2o8p+wRv4KIx02uUR053gN4*WNddlDiZ242Rf30v/pT3Ag==*eUD3YPnrpi8gCR0in+RUDg==*+TwdK7nlgA8kOnpP9kZ8Jg=='

def truncate_sentence(sentence: str, num_tokens: int) -> str:
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

def summarize_text(text, API_KEY=YOUR_API_KEY):
    str_decoded = cryptocode.decrypt(API_KEY, "openai")
    # set api key
    openai.api_key = str_decoded

    # 문장 분리
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # 문장 수가 10개 이하인 경우 그대로 반환
    # if len(sentences) <= 10:
    #     return text

    # OpenAI 문장 요약 API 요청
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=("The following text is part of a corporate report. Please summarize the main aspects of this company in 5 sentences or less. using bullet points:\n\n" + text),
        temperature=0.3,
        max_tokens=2096,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0
    )

    # 요약 결과 반환
    summary = response.choices[0].text.strip()
    return summary


# NASDAQ ticker를 입력합니다.
ticker = "BIGC"

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

modified_text = truncate_sentence(document, 1024)

# # Load GPT model and tokenizer
# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # Set the maximum sequence length
# max_length = 1024
#
# # Tokenize the text
# input_ids = tokenizer.encode(document, max_length=max_length, truncation=True)
#
# # Decode the tokenized input
# modified_text = tokenizer.decode(input_ids)

print(modified_text)

print(summarize_text(modified_text, YOUR_API_KEY))



