import graphviz

def create_ingestion_diagram():
    dot = graphviz.Digraph(comment='Ingestion Pipeline Workflow', format='png')
    
    # Best Practice: Use Orthogonal lines for Flowcharts
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', fontname='Arial', fontsize='10', style='filled', fillcolor='white', shape='box')
    dot.attr('edge', fontname='Arial', fontsize='9', color='#333333')
    dot.attr('graph', fontname='Arial', fontsize='12', style='rounded', bgcolor='#FAFAFA')

    # --- Input ---
    dot.node('Zip', 'Zip Archive\n(Source)', shape='note', fillcolor='#FFF2CC', color='#D6B656')

    # --- Stage 1: GPU Worker ---
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1: GPU Worker (Producer)', color='#6C8EBF', fontcolor='#444444')
        c.attr(style='dashed')
        
        c.node('S1_Start', 'Fetch Batch', shape='oval', style='filled', fillcolor='#DAE8FC')
        c.node('S1_Process', 'Sort & Embed\n(RTX 4090)', shape='box', fillcolor='#DAE8FC')
        c.node('S1_Push', 'Push to Q1', shape='box', fillcolor='#DAE8FC')
        
        # Forward flow
        c.edge('S1_Start', 'S1_Process')
        c.edge('S1_Process', 'S1_Push')
        
        # Loop back (Bottom to Bottom)
        # constraint=false prevents it from affecting layout rank
        c.edge('S1_Push', 'S1_Start', label='Next', constraint='false', color='#6C8EBF')

    # --- Queue 1 ---
    dot.node('Q1', 'Queue 1', shape='cylinder', fillcolor='#F5F5F5', width='0.6', fixedsize='true')

    # --- Stage 2: CPU Worker ---
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Stage 2: CPU Worker (Transformer)', color='#B85450', fontcolor='#444444')
        c.attr(style='dashed')

        c.node('S2_Get', 'Get Item', shape='oval', fillcolor='#F8CECC')
        c.node('S2_Build', 'Build Point\n(Metadata)', shape='box', fillcolor='#F8CECC')
        c.node('S2_Push', 'Push to Q2', shape='box', fillcolor='#F8CECC')

        c.edge('S2_Get', 'S2_Build')
        c.edge('S2_Build', 'S2_Push')
        
        # Loop back
        c.edge('S2_Push', 'S2_Get', label='Next', constraint='false', color='#B85450')

    # --- Queue 2 ---
    dot.node('Q2', 'Queue 2', shape='cylinder', fillcolor='#F5F5F5', width='0.6', fixedsize='true')

    # --- Stage 3: I/O Worker ---
    with dot.subgraph(name='cluster_stage3') as c:
        c.attr(label='Stage 3: I/O Worker (Consumer)', color='#82B366', fontcolor='#444444')
        c.attr(style='dashed')

        c.node('S3_Get', 'Get Item', shape='oval', fillcolor='#D5E8D4')
        c.node('S3_Buffer', 'Add to Buffer', shape='box', fillcolor='#D5E8D4')
        
        # Decision Diamond
        c.node('S3_Check', 'Buffer Full?', shape='diamond', fillcolor='#D5E8D4', height='0.6')
        
        c.node('S3_Upsert', 'Batch Upsert\n(Network I/O)', shape='box', fillcolor='#D5E8D4', style='filled,bold')

        # Logic Flow
        c.edge('S3_Get', 'S3_Buffer')
        c.edge('S3_Buffer', 'S3_Check')
        
        # Branch: Yes -> Upsert -> Loop
        c.edge('S3_Check', 'S3_Upsert', label='Yes')
        c.edge('S3_Upsert', 'S3_Get', label='Clear & Repeat', constraint='false', color='#82B366')
        
        # Branch: No -> Loop
        c.edge('S3_Check', 'S3_Get', label='No (Continue)', constraint='false', color='#82B366')

    # --- Output ---
    dot.node('DB', 'Qdrant DB', shape='cylinder', fillcolor='#FFE6CC', color='#D79B00')

    # --- Main Pipeline Connections ---
    dot.edge('Zip', 'S1_Start', style='bold')
    dot.edge('S1_Push', 'Q1', style='bold')
    dot.edge('Q1', 'S2_Get', style='bold')
    dot.edge('S2_Push', 'Q2', style='bold')
    dot.edge('Q2', 'S3_Get', style='bold')
    dot.edge('S3_Upsert', 'DB', style='bold')

    output_path = 'ingestion_optimization_diagram'
    dot.render(output_path, view=False)
    print(f"Diagram generated: {output_path}.png")

if __name__ == '__main__':
    create_ingestion_diagram()
