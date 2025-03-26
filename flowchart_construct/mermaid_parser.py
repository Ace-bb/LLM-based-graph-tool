"""
将Mermaid文件转成JSON格式
"""

import re


class Mermaid2Flowchart:
    def __init__(self, mermaid_code):
        self.mermaid_code = mermaid_code
        self.flowchart_code = (
            "from flowchart import Flowchart\n\nflowchart = Flowchart()\n"
        )
        self.nodes = set()  # Use a set to avoid duplicates
        self.edges = []
        self.flowchart_nodes = []
        self.flowchart_edges = []

    def simplify_mermaid_code(self):
        # Remove content inside parentheses, square brackets, and curly braces
        return re.sub(r"(\[.*?\]|\(.*?\)|\{.*?\})", "", self.mermaid_code)

    def parse_mermaid_code(self):
        # Separate nodes and edges from the mermaid code
        lines = self.mermaid_code.strip().split("\n")

        for line in lines:
            if "-->" in line:  # Edge definition
                # Extract node definitions from the edge lines
                parts = re.split(
                    r"-->|(\|.*\|)", line
                )  # Split by --> and any labels like |Yes|
                for part in parts:
                    if part and not part.startswith(
                        "|"
                    ):  # Avoid labels and empty parts
                        node = part.strip()
                        if node:
                            self.nodes.add(node)
            else:  # Node definition only
                self.nodes.add(line.strip())

        # Simplify the Mermaid code by removing node content
        simplified_code = self.simplify_mermaid_code()

        # Separate nodes and edges from the simplified code
        lines = simplified_code.strip().split("\n")

        for line in lines:
            if "-->" in line:  # Edge definition
                self.edges.append(line.strip())

    def generate_nodes(self):
        # Updated regex to handle different node formats based on the shape conventions:
        # Ellipses for ([ ]), rectangles for [ ], parallelograms for [/ /], and diamonds for { }
        node_pattern = re.compile(
            r"(\w+)\(\[\s?(.*?)\s?\]\)"  # Ellipses: A([Start])
            r"|(\w+)\[\/\s?(.*?)\s?\/\]"  # Parallelograms: A[/Some content/]
            r"|(\w+)\[\s?(.*?)\s?\]"  # Rectangles: A[Some content]
            r"|(\w+)\{\s?(.*?)\s?\}"  # Diamonds: A{Some decision}
        )

        node_lines = []
        for node in self.nodes:
            match = node_pattern.match(node)
            if match:
                content, node_id, shape = None, None, None
                if match.group(1):  # Ellipse (start/end), e.g., A([Start])
                    node_id = match.group(1)
                    content = match.group(2)
                    shape = "ellipse"
                elif match.group(3):  # Parallelogram, e.g., A[/Some content/]
                    node_id = match.group(3)
                    content = match.group(4)
                    shape = "parallelogram"
                elif match.group(5):  # Rectangle, e.g., A[Some content]
                    node_id = match.group(5)
                    content = match.group(6)
                    shape = "box"
                elif match.group(7):  # Diamond (decision), e.g., A{Some decision}
                    node_id = match.group(7)
                    content = match.group(8)
                    shape = "diamond"

                # Clean up the content by removing unnecessary quotes or spaces
                if content:
                    content = content.replace('"', "").strip()
                else:
                    content = ""

                # Add the node line to the list, but don't append it to the code yet
                node_lines.append(
                    f'flowchart.add_node("{node_id}", "{content}", "{shape}")\n'
                )
                self.flowchart_nodes.append({
                    "id": node_id,
                    "content": content,
                    "shape": shape
                })

        # Sort the node lines alphabetically by the node ID
        node_lines.sort()

        # Append the sorted node lines to the flowchart code
        self.flowchart_code += "\n" + "".join(node_lines)

    def generate_edges(self):
        # Regex to match edges, ensuring we capture nodes with various brackets or slashes
        edge_pattern = re.compile(
            r"([A-Za-z0-9_]+)\s*-->\s*(?:\|(.+?)\|)?\s*([A-Za-z0-9_]+)"
        )

        self.flowchart_code += "\n"
        for edge in self.edges:
            # Match the edge with the regex
            match = edge_pattern.match(edge)
            if match:
                from_node = match.group(1).strip()
                to_node = match.group(3).strip()  # The second node
                label = match.group(2) if match.group(2) else ""

                # Ensure we add the edge even if the nodes have extra symbols
                from_node_clean = re.sub(
                    r"(\[.*?\]|\(.*?\)|\{.*?\})", "", from_node
                ).strip()
                to_node_clean = re.sub(
                    r"(\[.*?\]|\(.*?\)|\{.*?\})", "", to_node
                ).strip()
                
                self.flowchart_edges.append({
                    "from": from_node_clean,
                    "to": to_node_clean,
                    "label": label
                })

                # Generate Python code for adding the edge
                if label:
                    label = label.replace('"', "")  # Removing extra quotes from label
                    # Fixing quotes for the label
                    self.flowchart_code += f'flowchart.add_edge("{from_node_clean}", "{to_node_clean}", "{label}")\n'
                else:
                    self.flowchart_code += (
                        f'flowchart.add_edge("{from_node_clean}", "{to_node_clean}")\n'
                    )

    def convert2json(self):
        self.parse_mermaid_code()
        print(self.nodes)
        print()
        print()
        print(self.edges)
        
        self.generate_nodes()  # Ensure nodes are added before edges
        self.generate_edges()
        print(self.flowchart_code)
        print()
        print(self.flowchart_nodes)
        print(self.flowchart_edges)
        
    def convert(self):
        self.parse_mermaid_code()
        self.generate_nodes()  # Ensure nodes are added before edges
        self.generate_edges()
        return self.flowchart_code

if __name__=="__main__":
    mermaid_code = """
flowchart TD\n    A([\"Start\"]) --> B[/\"Moisten the Stain with Water\"/]\n    B --> C[/\"Apply Lemon Juice\"/]\n    C --> D[/\"Apply Salt to Stain\"/]\n    D --> E{\"Is the stain still visible?\"}\n    E -->|Yes| F[/\"Rinse the Stain\"/]\n    F --> G{\"Is the stain nearly gone?\"}\n    G -->|No| F\n    G -->|Yes| H[/\"Dry with Towel\"/]\n    E -->|No| H\n    H --> I[/\"Apply More Lemon Juice\"/]\n    I --> J[/\"Sun-Dry the Fabric\"/]\n    J --> K([\"End\"])
    """
    converter = Mermaid2Flowchart(mermaid_code)
    converter.convert2json()