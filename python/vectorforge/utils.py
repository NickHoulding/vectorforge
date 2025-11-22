

def extract_file_content(content: bytes, file_type: str) -> str:
    """Extract the content of a file"""
    if file_type == 'txt':
        return content.decode('utf-8')
    elif file_type == '.pdf':

        # TODO: Implement PDF extraction here...
        
        return "PDF extraction not yet implemented"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
