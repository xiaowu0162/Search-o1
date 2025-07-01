import backoff 


OPENAI_REQUEST_TIMEOUT = 60*60*24     # a large timeout to make sure batch requests have enough time to run


@backoff.on_exception(backoff.constant, Exception, interval=5)
def run_chat_completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(backoff.constant, Exception, interval=5)
def run_generate_with_backoff(client, **kwargs):
    return client.completions.create(**kwargs)