class GroqAPIError(Exception):
    '''Base exception for any error returned by the Groq API.'''
    
    def __init__(self, message, status_code=None, response_body=None):
        super().__init__(message)
        self.status_code =status_code
        self.response_body = response_body
        

class GroqClientError(GroqAPIError):
    '''Raised for 4xx responses (bad request, unauthorized, etc.).'''
    


class GroqServerError(GroqAPIError):
    '''Raised for 5xx responses (Groq's servers are having trouble).'''
    
    

class GroqRateLimitError(GroqClientError):
    '''Raised when a 429 response persists after all retries are exhausted.'''
    
