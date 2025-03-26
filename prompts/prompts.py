def load_vqa_prompt(question):
    return f"Question: {question}\nAnswer:"


def load_textualizer_prompt(output_type):
    if output_type == "mermaid":
        return """Generate the Mermaid code for the provided flowchart.

Here is an example:
```mermaid
flowchart TD
    A(["Start"]) --> B[/"Receive 'arr' and 'n'"/]
    B --> C["Initialize loop index 'i' to 0"]
    C --> D{"Check if arr[i] == i"}
    D -->|"Yes"| E[/"Return index 'i' as fixed point"/]
    E --> F(["End"])
    D -->|"No"| G["Increment 'i'"]
    G --> H{"i < n"}
    H -->|"Yes"| D
    H -->|"No"| I[/"Return -1 as no fixed point found"/]
    I --> F
```"""
    elif output_type == "graphviz":
        return """Generate the Graphviz code for the flowchart in the provided image.

Here is an example:
```dot
digraph G {
    A [label="Start" shape=ellipse];
    B [label="Receive 'arr' and 'n'" shape=parallelogram];
    C [label="Initialize loop index 'i' to 0" shape=box];
    D [label="Check if arr[i] == i" shape=diamond];
    E [label="Return index 'i' as fixed point" shape=parallelogram];
    F [label="End" shape=ellipse];
    G [label="Increment 'i'" shape=box];
    H [label="i < n" shape=diamond];
    I [label="Return -1 as no fixed point found" shape=parallelogram];

    A -> B;
    B -> C;
    C -> D;
    D -> E [label="Yes"];
    E -> F;
    D -> G [label="No"];
    G -> H;
    H -> D [label="Yes"];
    H -> I [label="No"];
    I -> F;
}
```"""
    elif output_type == "plantuml":
        return """Generate the PlantUML code for the provided flowchart.

Here is an example:
```plantuml
@startuml
start
:Receive 'arr' and 'n';
:Initialize loop index 'i' to 0;

while (i < n?) is (Yes)
    if (Check if arr[i] == i?) then (Yes)
        :Return index 'i' as fixed point;
        stop
    else (No)
        :Increment 'i';
    endif
endwhile (No)
:Return -1 as no fixed point found;
stop
@enduml
```"""
    else:
        raise ValueError(f"Unsupported output type: {output_type}")


def load_reasoner_prompt(question, represenation):
    return f"{represenation}\n\nQuestion: {question}\nAnswer:"


def load_evaluation_prompt(question, response, label):
    prompt = f"""Task: Verify if the provided answer is correct based on the given ground truth.
    
You are given a question, an answer and the ground truth. Your task is to determine whether the provided answer matches the ground truth. Output "Correct" if the answer matches groud truth, otherwise output "Incorrect".

Question: {question}

Answer: {response}

Ground Truth: {label}"""

    return prompt
