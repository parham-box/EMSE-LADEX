structural_constraints = (
    'An activity diagram must have exactly one initial node. \n'
    'An activity diagram must have at least one end node. \n'
    'The initial node must have no incoming transitions. \n'
    'End nodes must have no outgoing transitions. \n'
    'Each decision node must have at least two outgoing transitions, each labelled by a guard condition.\n'
    'An activity diagram must be fully connected so that every node is reachable from the initial node.\n'
)

alignment_constraints = (
    'Action and transition labels in the activity diagram must be consistent with and accurately reflect the process description.\n'
    'The sequence of actions and transitions must accurately represent the order of actions and their triggers described in the process description.\n'
    'All possible action flows described in the process description must be represented in the activity diagram. The diagram must not introduce any actions or transitions that are not present in the process description.\n'
    'Concurrency occurs when actions happen simultaneously and is modelled using multiple parallel flows originating from a single action node. The parallel flows may synchronize into a single flow after some steps.\n'
    'Only procedural steps from the process description should be incorporated into the activity diagram. Examples, explanatory text, and commentary should be excluded.\n'
)

example_csv = (
    '1, "start the pc", "start", "", ""\n'
    '2, "are you connected to net?", "condition", "1", ""\n'
    '3, "run: mkdir <NAME>", "entity", "2", "Yes"\n'
    '4, "restart pc", "entity", "2", "No"\n'
    '5, "Open Browser", "entity", "3,4", "Connected to internet"\n'
    '6, "End", "end", "5", ""'
)

generate_prompt_template = (
    'You are an expert at generating activity diagrams in CSV format from user-provided textual process description following the below constraints:'
    '{structural_constraints}'
    '{alignment_constraints}'
    'Ensure nodes that have multiple parents are separated and a new row is created per parent but the same id is used for all rows, if each relationship with the parent requires a different transition label. '
    'The CSV type field can be "entity", "start", "end", "condition". '
    'Return only the final, complete CSV in the format below (without extra commentary or the CSV header): '
    'id, name, type, parent, transition_label\n'
    'Example:\n'
    '{example_csv}'
)

critique_LADEX_LLM_LLM_prompt_template = (
    'You are an expert in critiquing activity diagrams against their corresponding natural-language process descriptions. '
    'Provide detailed feedback identifying any violations of the diagram with respect to the provided constraints below:'
    '{structural_constraints}'
    '{alignment_constraints}'
    'Nodes with duplicate id are allowed when they have the same name. They represent the same node with unique transition labels to each of their parents. '
    'Return your critique of the activity diagram and assign a score from 1 to 10, where 1 indicates entirely incorrect and 10 indicates completely correct, in the format: Final Score: X/10.'
)

critique_LADEX_LLM_NA_prompt_template = (
    'You are an expert in critiquing activity diagrams against their corresponding natural-language process descriptions. '
    'Provide detailed feedback identifying any violations of the diagram with respect to the provided constraints below:'
    '{structural_constraints}'
    'Nodes with duplicate id are allowed when they have the same name. They represent the same node with unique transition labels to each of their parents. '
    'Return your critique of the activity diagram and assign a score from 1 to 10, where 1 indicates entirely incorrect and 10 indicates completely correct, in the format: Final Score: X/10.'
)

critique_LADEX_ALG_LLM_prompt_template = (
    'You are an expert in critiquing activity diagrams against their corresponding natural-language process descriptions. '
    'Provide detailed feedback identifying any violations of the diagram with respect to the provided constraints below:'
    '{alignment_constraints}'
    'Nodes with duplicate id are allowed when they have the same name. They represent the same node with unique transition labels to each of their parents. '
    'Return your critique of the activity diagram and assign a score from 1 to 10, where 1 indicates entirely incorrect and 10 indicates completely correct, in the format: Final Score: X/10.'
)

def refine_prompt_template(history):
    return (
        'You are an expert in refining activity diagrams based on detailed feedback and previously rejected iterations. Analyze the feedback and apply changes only in the areas highlighted by the feedback. The diagrams should follow these constraints:'
        '{structural_constraints}'
        '{alignment_constraints}'
        'History of previously rejected draft diagrams:\n'
        f'{history}\n'
        'Ensure nodes that have multiple parents are separated and a new row is created per parent but the same id is used for all rows, if each relationship with the parent requires a different transition label. '
        'The CSV type field can be "entity", "start", "end", "condition". '
        'Return only the final, complete CSV in the format below (without extra commentary or the CSV header): '
        'id, name, type, parent, transition_label\n'
        'Example:\n'
        '{example_csv}'
    )

def get_generate_prompt():
    config = {
        "structural_constraints": structural_constraints,
        "alignment_constraints": alignment_constraints,
        "example_csv": example_csv
    }
    return generate_prompt_template.format(**config)

def get_LADEX_LLM_LLM_prompt():
    config = {
        "structural_constraints": structural_constraints,
        "alignment_constraints": alignment_constraints,
    }
    return critique_LADEX_LLM_LLM_prompt_template.format(**config)

def get_LADEX_LLM_NA_prompt():
    config = {
        "structural_constraints": structural_constraints,
    }
    return critique_LADEX_LLM_NA_prompt_template.format(**config)

def get_LADEX_ALG_LLM_prompt():
    config = {
        "alignment_constraints": alignment_constraints,
    }
    return critique_LADEX_ALG_LLM_prompt_template.format(**config)

def get_refine_prompt(history):
    config = {
        "structural_constraints": structural_constraints,
        "alignment_constraints": alignment_constraints,
        "example_csv": example_csv
    }
    return refine_prompt_template(history).format(**config)

def get_csv_header():
    header1 = """## FlowChart Generator
# label: %name%
# style: whiteSpace=wrap;html=1;rounded=1;fillColor=#ffffff;strokeColor=#000000;
# namespace: csvimport-
# connect: {"from":"parent", "to":"id", "fromlabel":"transition_label", "invert":true, "style":"endArrow=blockThin;endFill=1;fontSize=11;curved=1;"}
# labels: {"label1" : "%transition_label%"}
# ignore: transition_label,parent,type,col
"""
    header2 = """
# stylename: type
# width: auto
# height: auto
# padding: 20
# nodespacing: 60
# levelspacing: 150
# edgespacing: 60
# layout: verticalflow
# identity: id"""
    header3 = """
id,name,type,parent,transition_label
"""

    return header1 + get_styles() + header2 + header3

def get_styles():
    
    styles = {
        "entity": "rounded=1;whiteSpace=wrap;html=1;absoluteArcSize=1;arcSize=14;strokeWidth=2;align=center;verticalAlign=middle;",
        "start": "strokeWidth=2;html=1;shape=mxgraph.flowchart.start_1;whiteSpace=wrap;align=center;verticalAlign=middle;",
        "end": "ellipse;html=1;shape=endState;fillColor=strokeColor;verticalAlign=middle;fontColor=#FFFFFF;",
        "condition": "strokeWidth=2;html=1;shape=mxgraph.flowchart.decision;whiteSpace=wrap;align=center;verticalAlign=middle;spacingRight=35;spacingLeft=35;"
    }
    
    all_styles = {**styles}
    style_string = ", ".join(f'"{k}":"{v}"' for k, v in all_styles.items())
    
    return "# styles: {" + style_string + "}"