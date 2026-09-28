class TokenTracker:
    '''Accumulates prompt/completion token usage across every call made through a GroqClient, 
    so a runing total can be printed when the program exits.'''
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        
    def record(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens 
    
    
    def summary(self):
        return (
            f'Total calls: {self.calls} |'
            f' prompt tokens: {self.prompt_tokens} |'
            f' completion tokens: {self.completion_tokens} |'
            f' total tokens: {self.total_tokens}'
        )
        
    
    