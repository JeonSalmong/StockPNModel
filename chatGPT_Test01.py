import openai
import argparse

import cryptocode

YOUR_API_KEY = '85Qs4r6Sa+pJp5zE/t7g7cOOg84nQ9jSw4I9ncRF4EWdt2o8p+wRv4KIx02uUR053gN4*WNddlDiZ242Rf30v/pT3Ag==*eUD3YPnrpi8gCR0in+RUDg==*+TwdK7nlgA8kOnpP9kZ8Jg=='


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


def main():
    # 지문 입력 란
    # prompt = input("Insert a prompt: ")
    prompt = '하이브, SM 공개매수 본격화…소액주주 5만2천여명 대상(종합) 이 문장이 긍정문이야? 부정문이야?'
    print(chatGPT(prompt).strip())

    # str_encoded = cryptocode.encrypt("api_key", "openai")
    # print(str_encoded)
    # ## And then to decode it:
    # str_decoded = cryptocode.decrypt(str_encoded, "openai")
    # print(str_decoded)

if __name__ == '__main__':
    main()