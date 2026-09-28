"""
Parses and validates the model's structured JSON output. Never lets a
malformed or non-JSON response crash the program - always returns a clear
InvalidResponseError instead, which callers can catch."""

import json

REQUIRED_FIELDS = {
    'answer': str,
    'source_rows': list,
    'confidence': str,
}
VALID_CONFIDENCE = {'low', 'medium', 'high'}


class InvalidResponseError(Exception):
    '''Raised when the model's response is not valid, well-shaped JSON.'''


def parse_and_validate(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidResponseError(f'Model response was not valid JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise InvalidResponseError('Model response was not an object.')
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            raise InvalidResponseError(f'Missing required field: {field}')
        if not isinstance(data[field], expected_type):
            raise InvalidResponseError(
                f'Field {field} has wrong type: expected {expected_type.__name__}, '
                f'got {type(data[field]).__name__}'
            )

    if data['confidence'] not in VALID_CONFIDENCE:
        confidence_value = data['confidence']
        raise InvalidResponseError(f'Invalid confidence value: {confidence_value!r}')

    return data