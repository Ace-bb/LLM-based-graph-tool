dx_disease_generate_dot_prompt = """你是一个疾病诊断流程图生成专家，下面给你关于一个疾病的“症状”、“病因”以及“诊断方法”相关的内容，你需要根据这些内容生成这个疾病的诊断流程图，并且使用Graphviz的Dot语言来描述生成的流程图。

生成的流程图需要满足以下要求：
1. 生成的流程图以一个症状为根节点，根节点的症状需要从给定的疾病症状中选择。
2. 生成的流程图的节点类型分为“根节点”、“条件节点”和“叶子节点”，其中根节点为起始症状，条件节点为诊断疾病所需要判断的全部条件，可以为是否有某个症状，病因等，叶子节点为诊断方法或疾病。
3. 条件节点使用疑问句描述，叶子节点使用陈述句描述。
4. 条件节点的条件类型可以为二分类，回答是或者否，也可以为多分类，相当于一个多分支的判断，也可以是一个开放性问题，根据回答的内容进行下一步的判断。
5. 节点之间的连接线上需要标注条件，如“是”或者“否”等。
6. 一个条件节点只需要包含一个问题，一个条件节点可以有多个子节点。
7. 使用Graphviz的Dot语言描述生成的流程图，生成的.dot文件内容能够被Graphviz程序正确执行。

参考下面这个Dot流程图示例：
```dot
digraph {{
	0 [label="Are there alarm symptoms?Sudden onset, Incontinence,Saddle anesthesia,Signs of stroke"]
	1 [label="Is the abnormality due to poor initiation of gait, muscular weakness, sensory abnormalities, or inability to stop gait?"]
	2 [label="Obtain spine MRIto exclude abscessor metastatic lesion"]
	4 [label="Obtain immediatehead CT toexclude stroke"]
	5 [label="Is there facial or upperextremity weakness?"]
	6 [label="When you walk, do you waddle like a duck?"]
	7 [label="When you walk, do you feel as if you are walking up steps?"]
	8 [label="Do the knees cross likescissors?"]
	9 [label="having a hard time walking(Gait abnormalities)."]
	12 [label="Have you noticed difficulty keeping your balance?"]
	13 [label="Peripheral neuropathy"]
	14 [label="Do you have a history of multiple sclerosis?"]
	15 [label="Cerebellar disease. When damaged or impaired (by alcohol), the patient cannot make fine motor adjustments to keep moving forward, resulting in truncal imbalance."]
	16 [label="Do you have trouble getting yourself going forward when you try to walk? "]
	17 [label="Waddling pattern: Gluteal or quadriceps weakness"]
	18 [label="Exaggerated foot raise during gait:Damage to the peroneal nerve"]
	19 [label="Spastic paraplegia"]
	20 [label=Parkinsonism]
	21 [label="Were there any complications associated with your birth?"]
	22 [label="Spastic paraplegia:"]
	8 -> 16 [label=No color=black]
	12 -> 15 [label=Yes color=black]
	8 -> 14 [label=Yes color=black]
	12 -> 13 [label=No color=black]
	5 -> 2 [label=No color=black]
	0 -> 1 [label=No color=black]
	9 -> 0 [label="" color=black]
	1 -> 6 [label="Inability to initiate gait" color=black]
	1 -> 7 [label="Inability toinitiate gait" color=black]
	1 -> 12 [label="Sensory abnormalities" color=black]
	1 -> 8 [label="nability to relax the legs; rigid gait (ie, inability to stop gait)" color=black]
	0 -> 5 [label=Yes color=black]
	5 -> 4 [label=Yes color=black]
	6 -> 17 [label=Yes color=black]
	7 -> 18 [label=Yes color=black]
	14 -> 19 [label=Yes color=black]
	16 -> 20 [label=Yes color=black]
	8 -> 21 [label=Yes color=black]
	21 -> 22 [label="Yes, cerebral palsy" color=black]
}}
```

疾病相关内容为：
{content}

现在，请你根据上面这些内容生成一个疾病的诊断流程图，并且使用Graphviz的Dot语言表示，生成的流程图需要满足上面的要求。
Dot语言表示的流程图为：
"""