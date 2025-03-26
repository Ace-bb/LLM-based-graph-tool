
def dot_to_mermaid(dot_content):
    """
    将dot格式的流程图转换成mermaid格式
    :param dot_content: str, dot格式的内容
    :return: str, mermaid格式的内容
    """
    mermaid_content = "flowchart TD\n"
    lines = dot_content.splitlines()
    for line in lines:
        line = line.strip()
        if '->' in line:
            parts = line.split('->')
            start_node = parts[0].strip()
            end_node = parts[1].strip().strip(';')
            mermaid_content += f"    {start_node} --> {end_node}\n"
        elif '[' in line and ']' in line:
            node = line.split('[')[0].strip()
            mermaid_content += f"    {node}\n"
    return mermaid_content