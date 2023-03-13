import cryptocode

key = 'myapikey'

str_encoded = cryptocode.encrypt(key, "openai")
print(str_encoded)
## And then to decode it:
str_decoded = cryptocode.decrypt(str_encoded, "openai")
print(str_decoded)