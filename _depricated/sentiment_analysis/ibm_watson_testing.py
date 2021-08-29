import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

API_KEY = "kb9SRXf0b6Ra-tqDo7PKEvPWcqjWvWXnrJoMyNoL03p4"
URL = "https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/c6784cc1-e93d-41d9-94b2-48baaac52884"

authenticator = IAMAuthenticator(f'{API_KEY}')
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    authenticator=authenticator
)

tone_analyzer.set_service_url(f'{URL}')

# with open("out.txt", "r") as file:
#     text = file.read()
text = """
Apple expands buybacks by $30 billion, OKs 7-for-1 stock split
Apple, Google agree to settle antitrust lawsuit over hiring deals - filing
Apple's sales boom in communist Vietnam
Apple resets the clock as investors await next big thing
Apple makes final pitch to U.S. jury in Samsung trial
Apple supplier Cirrus to buy British chip maker Wolfson
"""

# tone_analysis = tone_analyzer.tone({'text': text}, content_type='application/json').get_result()


# print(json.dumps(tone_analysis, indent=2))

# with open("returned.json", "w") as file:
#     file.write(json.dumps(tone_analysis, indent=2))