def convert_dot_2_json():
    import json
    import networkx as nx
    from networkx.readwrite import json_graph

    dot_graph = nx.Graph(nx.nx_pydot.read_dot("data/datasets/Flowchart2DotDatasets/dotV1/Dot/儿科/Apgar 评分/0_all.dot"))
    print(nx.node_link_data(dot_graph))
    # print(json_graph.dumps(dot_graph))

if __name__=="__main__":
    # construct_dot_img()
    # construct_dot_img_train_datasets()
    convert_dot_2_json()