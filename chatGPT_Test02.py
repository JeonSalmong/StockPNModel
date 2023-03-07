import openai
import argparse
import re
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


# 문장 요약 함수
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
        prompt=("Please summarize the following text:\n\n" + text),
        temperature=0.3,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # 요약 결과 반환
    summary = response.choices[0].text.strip()
    return summary

def main():
    # 지문 입력 란
    # prompt = input("Insert a prompt: ")
    # prompt = 'Hi-Mart, up to 50% discount on spring home appliances Is this a positive sentence? Is it a negative sentence?'
    # prompt = "Setopia, Last Year's Sales of KRW 116.5 Billion...51% from the previous year ↑ Is this a positive sentence? Is it a negative sentence?"
    prompt = """
    이것은 한글 문단입니다. 문단은 여러 개의 문장으로 이루어져 있습니다. 이 문장은 두 번째 문장입니다.
    """

    # 문장 구분을 위한 패턴
    sentence_pattern = re.compile(r'.+?[.?!]')

    # 문단을 문장으로 분리
    sentences = sentence_pattern.findall(prompt)
    print(sentences[0].strip())

    # 출력
    # for sentence in sentences:
    #     print(sentence.strip())

    # print(summarize_text(prompt))

    # str_encoded = cryptocode.encrypt("api_key", "openai")
    # print(str_encoded)
    # ## And then to decode it:
    # str_decoded = cryptocode.decrypt(str_encoded, "openai")
    # print(str_decoded)

if __name__ == '__main__':
    main()