# utils/helpers.py

def is_hex(value):
    try:
        int(str(value), 16)
        return True
    except (ValueError, TypeError):
        return False
