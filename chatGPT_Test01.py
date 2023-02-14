import openai
import argparse

YOUR_API_KEY = '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'


def chatGPT(prompt, API_KEY=YOUR_API_KEY):
    # set api key
    openai.api_key = API_KEY

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


if __name__ == '__main__':
    main()