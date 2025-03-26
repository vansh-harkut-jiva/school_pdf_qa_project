'''
import openai

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


OPENAI_API_KEY = "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"

if check_openai_api_key(OPENAI_API_KEY):
    print("Valid OpenAI API key.")
else:
    print("Invalid OpenAI API key.")'
    '''
'''
import openai

openai.api_key = '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'

def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True

# Check the validity of the API key
api_key_valid = is_api_key_valid()
print("API key is valid:", api_key_valid)'
'''
import openai

# Replace with your OpenAI API key
openai.api_key = "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"

try:
    # Test the embeddings model
    response = openai.Embedding.create(
        input="This is a test input.",
        model="text-embedding-ada-002"
    )
    print("✅ API Key is valid and has access to the embeddings model.")
except openai.error.AuthenticationError:
    print("❌ Invalid API key. Please check your API key and try again.")
except openai.error.PermissionError:
    print("❌ API key does not have access to the embeddings model.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")