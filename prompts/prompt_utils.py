def load_messages(model_name, prompt, image=None):
    # Text only messages (Note that VLMs can be used as LLMs without image input)
    if not image:
        templates = {
            # Close sour LLMs
            "claude-3-5-sonnet": [
                {"role": "user", "content": prompt},
            ],
            "gpt-4o": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "gpt-4o-mini": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            # Open source LLMs
            "Llama-3.1-8B": [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant",
                },
                {"role": "user", "content": prompt},
            ],
            "Llama-3.1-70B": [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant",
                },
                {"role": "user", "content": prompt},
            ],
            "Mixtral-8x22B": [{"role": "user", "content": prompt}],
            "Phi-3.5-mini": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "Phi-3.5-MoE": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-7B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-14B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-32B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "Qwen2.5-72B": [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            # Open source VLMs
            "Llama-3.2-11B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Llama-3.2-90B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "llava-v1.6-110b": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-7B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-72B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
    # Messages include both image and text
    else:
        templates = {
            # Close source VLMs
            "claude-3-5-sonnet": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "gpt-4o": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "gpt-4o-mini": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            # Open source VLMs
            "Llama-3.2-11B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Llama-3.2-90B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "llava-v1.6-110b": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ],
            "Qwen2-VL-7B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "Qwen2-VL-72B": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
    return templates.get(model_name)


def load_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_number_of_nodes",
                "description": "Returns the number of nodes in the flowchart.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_number_of_edges",
                "description": "Returns the number of edges in the flowchart.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_successors",
                "description": "Returns the direct successors (outgoing connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_direct_predecessors",
                "description": "Returns the direct predecessors (incoming connections) of the given node by its description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_description": {
                            "type": "string",
                            "description": "The description of the given node.",
                        },
                    },
                    "required": ["node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shortest_path_length",
                "description": "Returns the number of edges in the shortest path between two nodes based on their descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_node_description": {
                            "type": "string",
                            "description": "The description of the start node.",
                        },
                        "end_node_description": {
                            "type": "string",
                            "description": "The description of the end node.",
                        },
                    },
                    "required": ["start_node_description", "end_node_description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_indegree",
                "description": "Returns the maximum indegree (number of incoming edges) for any node.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_outdegree",
                "description": "Returns the maximum outdegree (number of outgoing edges) for any node.",
            },
        },
    ]
    return tools


def load_tools_code():
    return """
def get_number_of_nodes():
    return flowchart.get_number_of_nodes()

def get_number_of_edges():
    return flowchart.get_number_of_edges()

def get_direct_successors(node_description):
    return flowchart.get_direct_successors(node_description)

def get_direct_predecessors(node_description):
    return flowchart.get_direct_predecessors(node_description)

def get_shortest_path_length(node1, node2):
    return flowchart.get_shortest_path_length(node1, node2)

def get_max_indegree():
    return flowchart.get_max_indegree()

def get_max_outdegree():
    return flowchart.get_max_outdegree()
"""
