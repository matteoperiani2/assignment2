import string
from text_to_num import text2num

def is_number(string: str) -> bool:
    """
    Check whether a string is a number, both written or numeric.

    Args:
    - string (str): The string to be checked.

    Returns:
    - True if the string is a number, False otherwise.
    """
    if string.isdigit():
        return True
    try:
        # Try to convert written number to integer
        text2num(string, 'en', relaxed=True)
        return True
    except ValueError:
        return False
    
def get_readable_question(question, answer, rationale, answer_type, turn=None):
    lines = []
    if turn is not None:          
        lines.append(f'turn: {turn}')
    lines.append(f'Q\t\t{question} || {answer_type}')
    lines.append(f'A\t\t{answer} || {rationale}')

    return '\n'.join(lines)