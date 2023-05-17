import bardapi
import os
import json

# set your __Secure-1PSID value to key
os.environ['_BARD_API_KEY']="WQhZf3gnVwZeaViB9fzmsfl_hjviLE3O0fWQfkCDACprxXMODVyz_-jygXM5D3XnrwKubQ."

# set your input text
input_text = "LUCID 향후 전망은 어떤지 긍정 또는 부정으로만 답변 해 주세요"

# Send an API request and get a response.
response = bardapi.core.Bard().get_answer(input_text)['content']

# input_text = "다음 문장을 문장 구분 기호를 사용해서 3 문장으로 요약 해 주세요. (요약된 문장만 답변 해 주세요)" + response
# response = bardapi.core.Bard().get_answer(input_text)['content']

print(response)