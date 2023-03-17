import openai
import argparse

import cryptocode

YOUR_API_KEY = 'XZPcwFh3sDNF5ubvhiFkuXkXRuIxxcDXNAtIe3QvcAyYhpk6Ft4tEzSw/SGmBBQSsq7u*5S/2/laTqhpA1cZsbMgzsw==*oldzeHRCsuQ3vwe0AdOfHQ==*dgx9cIxXL6gXUMeJW1LkpA=='


def chatGPT(prompt, API_KEY=YOUR_API_KEY):

    str_decoded = cryptocode.decrypt(API_KEY, "openai")
    # set api key
    openai.api_key = str_decoded

    # Call the chat GPT API
    completion = openai.Completion.create(
        engine='text-davinci-003'  # 'text-curie-001'  # 'text-babbage-001' #'text-ada-001'
        , prompt=prompt
        , temperature=0.5
        , max_tokens=1024
        , top_p=1
        , frequency_penalty=0
        , presence_penalty=0)

    return completion['choices'][0]['text']

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

def get_result_pn(sentence):

    # statement word
    s_word_list_kor_type1 = ['긍정', '부정']
    s_word_list_kor_type2 = ['사실', '진술']
    s_word_list_eng_type1 = ['positive', 'negative']
    s_word_list_eng_type2 = ['statement', 'fact']

    chk_statement = False

    chatresult = None

    result_gpt = 0

    # 결과 값이 영문인지 한글인지 여부 체크
    if is_korean(sentence):

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

    elif is_english(sentence):

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
        chatresult = 'C'

    return chatresult

def main():
    # 지문 입력 란
    # prompt = input("Insert a prompt: ")
    # prompt = 'Hi-Mart, up to 50% discount on spring home appliances Is this a positive sentence? Is it a negative sentence?'
    # prompt = "Setopia, Last Year's Sales of KRW 116.5 Billion...51% from the previous year ↑ Is this a positive sentence? Is it a negative sentence?"
    prompt = "Deb Sisters' stock price fell 6% despite the release of a new product. Is this a positive sentence? Is it a negative sentence"
    result = chatGPT(prompt).strip()
    print(result)

    print(get_result_pn(result))

    # str_encoded = cryptocode.encrypt("api_key", "openai")
    # print(str_encoded)
    # ## And then to decode it:
    # str_decoded = cryptocode.decrypt(str_encoded, "openai")
    # print(str_decoded)

if __name__ == '__main__':
    main()