import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

model_name = "text-davinci-003"
training_data = "<PATH_TO_TRAINING_DATA>"
model_id = None  # Set this to a model ID if you want to continue training an existing model

response = openai.Model.create(
    model=model_name,
    prompt=None,
    temperature=0.5,
    max_tokens=1024,
    n=1,
    stop=None,
    frequency_penalty=0,
    presence_penalty=0,
    logprobs=10,
    echo=False,
    stop_sequence=None,
    model_id=model_id,
    use_cache=True,
    training_data=training_data,
)

# Do something with the response
print(response)