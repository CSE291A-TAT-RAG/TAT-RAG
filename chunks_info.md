# Chunks info

## Old

    ====== Chunks Token Statistics ======
    Total chunks      : 419696
    Min tokens        : 1
    Max tokens        : 900
    Mean tokens       : 41.88
    Median tokens     : 11.00
    25th percentile   : 4.00
    75th percentile   : 40.00

    Token count histogram (approx):
    [0, 20)      : 263490 chunks ( 62.8%)
    [20, 50)     :  61687 chunks ( 14.7%)
    [50, 150)    :  59835 chunks ( 14.3%)
    [150, 300)   :  27288 chunks (  6.5%)
    [300, 500)   :   6224 chunks (  1.5%)
    [500, +inf)  :   1172 chunks (  0.3%)

    ------ Smallest chunk ------
    Tokens : 1
    Line   : 8
    doc_id       : Monolithic-Power-Systems-Inc_2019
    page         : 1
    section_path : Table of Contents
    text preview : or

    ------ Largest chunk ------
    Tokens : 900
    Line   : 81104
    doc_id       : conagra-brands-inc_2019
    page         : 24
    section_path : ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF
    text preview : Readers of this report should understand 

## Merged 1 (no max limit)

    ====== Chunks Token Statistics ======
    Total chunks      : 144059
    Min tokens        : 1     
    Max tokens        : 900   
    Mean tokens       : 122.01
    Median tokens     : 92.00 
    25th percentile   : 58.00 
    75th percentile   : 160.00

    Token count histogram (approx):
    [0, 20)      :   8346 chunks (  5.8%)
    [20, 50)     :  16603 chunks ( 11.5%)
    [50, 150)    :  79211 chunks ( 55.0%)
    [150, 300)   :  31405 chunks ( 21.8%)
    [300, 500)   :   7277 chunks (  5.1%)
    [500, +inf)  :   1217 chunks (  0.8%)

    ------ Smallest chunk ------
    Tokens : 1
    Line   : 25
    doc_id       : Monolithic-Power-Systems-Inc_2019
    page         : 4
    section_path : Table of Contents
    text preview : 4

    ------ Largest chunk ------
    Tokens : 900
    Line   : 28575
    doc_id       : conagra-brands-inc_2019
    page         : 24
    section_path : ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF
    text preview : Readers of this report should understand that these forward-looking statements are not guarantees of performance or results. Forward-looking statements provide our current expectations and beliefs con ...

## Merged 2 (max limit 350)

    ====== Chunks Token Statistics ======
    Total chunks      : 131751
    Min tokens        : 1     
    Max tokens        : 350   
    Mean tokens       : 126.78
    Median tokens     : 102.00
    25th percentile   : 65.00
    75th percentile   : 171.00

    Token count histogram (approx):
    [0, 20)      :    144 chunks (  0.1%)
    [20, 50)     :  13250 chunks ( 10.1%)
    [50, 150)    :  77340 chunks ( 58.7%)
    [150, 300)   :  35951 chunks ( 27.3%)
    [300, 500)   :   5066 chunks (  3.8%)
    [500, +inf)  :      0 chunks (  0.0%)

    ------ Smallest chunk ------
    Tokens : 1
    Line   : 6854
    doc_id       : advanced-energy_2019
    page         : 53
    section_path : ITEM 7A.           QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK
    text preview : (2.9)%

    ------ Largest chunk ------
    Tokens : 350
    Line   : 408
    doc_id       : Monolithic-Power-Systems-Inc_2019
    page         : 61
    section_path : Table of Contents
    text preview : The fair value measurement involves the analysis of valuation techniques and evaluation of unobservable inputs commonly used by market participants to price similar instruments. Outputs from the valua ...

## some tips

1. rule: same doc_id
