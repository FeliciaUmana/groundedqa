from groundedqa.prompts import build_system_prompt, build_user_prompt, build_messages

def test_system_prompt_instructs_not_in_data_on_missing_answerss():
    prompt = build_system_prompt()
    assert 'NOT_IN_DATA' in prompt
    assert 'JSON' in prompt
    
def test_user_prompt_orders_instructions_data_then_question():
    prompt = build_user_prompt('How many customers?', 'Row 0: Country=USA')
    instructions_pos = prompt.find('Instructions')
    data_pos = prompt.find('Data:')
    question_pos = prompt.find('Question:')
    assert instructions_pos < data_pos < question_pos
    
def test_build_messages_has_system_then_user_role():
    messages = build_messages('test question', 'some data')
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert 'test question' in messages[1]['content']
    assert 'some data' in messages[1]['content']
    
