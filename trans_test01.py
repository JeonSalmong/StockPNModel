import requests
import json

# # 번역할 문장
# text = "한화투자증권, 미국주식 주간거래서비스 시작"
#
# # 파파고 API URL
# url = "https://openapi.naver.com/v1/papago/n2mt"
#
# # 파라미터 설정
# data = {
#     "source": "ko",
#     "target": "en",
#     "text": text
# }
#
# # Header 설정
# headers = {
#     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
#     "X-Naver-Client-Id": "KOfQjnVLH_96w5zeZOJN",
#     "X-Naver-Client-Secret": "aVDbTdSfVP"
# }
#
# # POST 요청
# response = requests.post(url, data=data, headers=headers)
#
# # 결과 출력
# result = json.loads(response.text)
# translated_text = result['message']['result']['translatedText']
# print(translated_text)


def is_korean(text):
    for char in text:
        if '가' <= char <= '힣':
            return True
    return False

def is_english(text):
    for char in text:
        if 'a' <= char.lower() <= 'z':
            return True
    return False

def detect_language(text):
    if is_korean(text):
        return 'Korean'
    elif is_english(text):
        return 'English'
    else:
        return 'Unknown'


text1 = "Hello, world!"
text2 = "안녕하세요!"
text3 = "こんにちは"

print(detect_language(text1))
print(detect_language(text2))
print(detect_language(text3))