URL_TEMPLATE = {
    "text" : """
Generate a valid wikipedia url from a question. For example:

question: "Who was Napoleon?"
your answer: https://en.wikipedia.org/wiki/Napoleon

question: {user_prompt}
your answer:
""",
    "input_variables" : ['user_prompt']
}

SUMMARY_TEMPLATE = {
    "text" : """
Generate a summary of the following wikipedia article using less then 200 words.

article: {wiki_article}
your summary:   
""",
    "input_variables" : ["wiki_article"]
}