PROMPT = '''Convert the following Hate Speech into Neutral Text.

Hate Speech:
{hate_text}

Neutral Text:
{neutral_text}
'''

PROMPT_INSTRUCT = '''<|startoftext|><|start_header_id|>user<|end_header_id|>

Convert the following Hate Speech into Neutral Text.

Hate Speech:
{hate_text}

Neutral Text:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{neutral_text}
'''