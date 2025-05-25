finance_triplets_prompt_system = """You are a finance expert who is good at reading SEC 10k reports. Your task is to extract triplets including two entities and one relation from the following consecutive chunks from a same SEC 10k report (separated by newlines).
Focus only on factual knowledge that can be directly and concisely answered. For each triplet, you MUST extract it from ONE of the consecutive chunks. But other consecutive chunks should be read through as your reference to extract meaningful, non-naive triplets.

# TASK: EXTRACT TRIPLETS
Crucially, you should extract following types of multi-hop triplets from the text:
- Connected Chain of triplets
- Star-shaped triplets
- Inverted star-shaped triplets

## Connected Chain of triplets:
- First extract triplet of <entity_1, relation, entity_2> from the text. 
- Then, look for the triplets which have entity_2 of the first triplet as entity_1 of the second triplet.
- Continue this process until you have a chain of triplets.
- You are encourage to extract as many connected chains as possible. (less single-hop triplets)
- For each triplet on the chain, you are encouraged to extract from different chunks.
- Skip the this type of triplets if you cannot find any valid triplets that could form a meaningful chain problem.

### Chain Triplet Example 1:
{{"entity_1": "NVIDIA","relation": "acquired","entity_2": "Mellanox Technologies","source_sentence": "On April 27, 2020, we completed the acquisition of Mellanox Technologies Ltd., a supplier of high-performance interconnect solutions for computing, storage and communications applications."}}
{{"entity_1": "Mellanox Technologies","relation": "provides","entity_2": "high-performance interconnect solutions","source_sentence": "On April 27, 2020, we completed the acquisition of Mellanox Technologies Ltd., a supplier of high-performance interconnect solutions for computing, storage and communications applications."}}

### Chain Triplet Example 2:
{{"entity_1": "NVIDIA","relation": "develops","entity_2": "GeForce GPUs","source_sentence": "We operate in two reportable segments: Graphics and Compute & Networking. Our Graphics segment includes GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service, and related infrastructure and services."}}
{{"entity_1": "GeForce GPUs","relation": "designed for","entity_2": "gaming and PCs","source_sentence": "We operate in two reportable segments: Graphics and Compute & Networking. Our Graphics segment includes GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service, and related infrastructure and services."}}

## Star-shaped triplets:
A Star-shape multi-hop pattern is formed when one root entity spawns two or more branches, each through a different relation:
- <entity_1, relation_1, entity_2>
- <entity_1, relation_2, entity_3>
- entity_1 must be identical in every branch triplet.
- relation_1 and relation_2 must be different and satisfy the general relation guidelines.
- Additional downstream triplets may be chained to entity_2, entity_3, etc., but are optional.
- Every branch must come from a sentence (or data from a table) that independently supports its triplet
- Recommend: each triplet should come from different chunks
- Skip this type of triplets if you cannot find any valid triplets that could form a meaningful star-shaped problem.
    
## Star-shaped triplet Example:
{{"entity_1": "Panasonic","relation_1": "supplies lithium-ion cells to","entity_2": "Tesla","source_sentence": "Panasonic Corporation supplies lithium-ion cells to Tesla, Inc."}}
{{"entity_1": "Panasonic","relation_2": "holds","entity_2": "ISO 9001 certification (battery plan)","source_sentence": "Panasonic Corporation holds ISO 9001 certification for its battery production process."}}

## Inverted star-shaped triplets:
A Inverted star-shaped pattern links two otherwise unrelated entities through a shared attribute:
- <entity_1, relation_1, entity_2>
- <entity_3, relation_2, entity_2>
- entity_2 must be identical in both triplets.
- relation_1 and relation_2 might be different (not necessarily) and satisfy the general relation guidelines.
- The two relations should describe different perspectives on the same attribute.
- Skip this type of triplets if you cannot find any valid triplets that could form a meaningful inverted star-shaped problem.

## Inverted star-shaped triplet Example:
{{"entity_1": "Ford",   "relation": "manufactures F-150 trucks in", "entity_2": "Dearborn, Michigan plant", "source_sentence": "We manufacture F-150 trucks in our Dearborn, Michigan plant."}}
{{"entity_3": "SK On", "relation": "supplies batteries to", "entity_2": "Dearborn, Michigan plant", "source_sentence": "SK On will supply batteries to Ford's Dearborn, Michigan plant under a long-term agreement."}}

## General Requirements You Should ALWAYS Follow for relations and entities:
1. Each returned triplets MUST contain the following fields:
    - entity_1 (str): The first entity
    - relation (str): The relationship between entity_1 and entity_2
    - entity_2 (str): The second entity
    - source_sentence (str): The exact sentence from the text that support this triplet.
      - Note: the source_sentence should be extracted from one of the consecutive chunks that you extracted this triplet from.
      - Note: the source_sentence should keep the original format of the text, including the original punctuation, numbers, newlines, etc (except for the table).
      - Note: if you extract a triplet from a table, your source_sentence should strictly recent the most relevant row+column+specific data, DO NOT RETURN THE WHOLE TABLE. DO NOT RETURN OTHER INFORMATION IN THE SOURCE_SENTENCE.

2. Critical Instructions for Relations:
    - Relations should be GENERALIZED and REUSABLE across different companies in the same industry
    - Relations should use common financial and business terminology
    - Relations should be concise and specific (ideally 2-4 words)
    - Relations should describe actions, states, or relationships in a standard way
    - AVOID company-specific language in the relation itself
    - Your relation should not contain any time-sensitive or any other specific information (e.g. location, other company names etc.). 
        For example: "sale in 2023" is not a good relation, it should be "sale".

    Examples of GOOD relations:
    - "manufactures", "acquired", "operates in", "sells to", "partnered with", "increased revenue by"
    - "appointed as", "headquartered in", "founded in", "competes with"

    Examples of BAD relations (too specific):
    - "software technology helps accelerate deployment of AI solutions optimized for"
    - "created groundbreaking advancements in semiconductor architecture that powers"
    - "revolutionary cloud computing platform introduced in 2023 helps customers"

3. Critical Instructions for Entities:
    - Avoid vague entities like "the company", "the report", or "the government". Specify clearly: "NVIDIA", "NVIDIA's product", "NVIDIA's service", "NVIDIA's business", etc.
    - Your entities should not contain any general information which is available in every 10k report, it should be specific to the company/company's product/service/business/etc.
    - But it is okay/welcome to include the specific information (e.g. year, location, etc.) in the entity_2 as the answer.
    - For the table, you may include the both column, row and description information as part of entity

Your goal is to extract {number_of_triplets_per_run} triplets. You must reach this goal.
For each triplet, you must return the following fields:
- entity_1
- relation
- entity_2
- source_sentence

If no valid triplets can be extracted, return an empty array: []
"""
finance_triplets_prompt_user = """
Text Chunks:
{chunk_text}

Metadata:
{metadata}

Number of triplets to extract: 
{number_of_triplets_per_run}

Triplets:
"""

finance_onehop_query_prompt_system = """
You are an expert in finance and financial analysis. Generate a finance-related question and answer based on the information below:
- Triplet Information:
    - Entity 1
    - Relation
    - Entity 2
- One answer chunk from the SEC 10k report that contains the more details.

Instructions:
Requirements for question generation:
1. Create a clear and specific finance question that requires the information in the triplet to answer.
2. The question should be natural, use proper financial terminology, and be something a financial analyst might ask.
3. If the triplet cannot form a meaningful financial question (or the question is too general/naive), you MUST return an empty string for question and answer.
4. If the fact in entity 1 or entity 2 is slightly different from the fact in the context, you should use the fact in the context as the answer of question.
- For example, if the entity 2 is "$3,818.1 million in 2024" in the triplet, but the context mentions "3,818.1 million" in 2023, you should use the fact in the context to generate question.

Requirements for answer generation:
1. You answer should only have few words (not sentence). It will ONLY contain the core information. 
2. Make sure that the answer can always be found in the triplet.
3. If the fact in entity_1 or entity_2 (the entity you selected as the answer) is slightly different from the fact in the context, you should use the fact in the context as the answer.
- For example, if the entity_2 (the entity you selected as the answer) is "$3,818.1 million in 2024" in the triplet, but the context mentions "3,818.1 million" in 2023, you should use the fact in the context to generate answer.

Example:
Entity 1: "smith a o"
Relation: "consolidated sales"
Entity 2: "$3,818.1 million in 2024"

Context:
In this section, we discuss the results of our operations for 2024 compared with 2023. We discuss our cash flows and current financial condition under \"Liquidity and Capital Resources.\" For a discussion related to 2023 compared with 2022, please refer to Item 7 of Part II, \"Managements Discussion and Analysis of Financial Condition and Results of Operations\" in our Annual Report on Form 10-K for the Year Ended December 31, 2023, which was filed with the United States Securities and Exchange Commission (SEC) on February 13, 2024, and is available on the SEC's website at www.sec.gov.\n| | | | | | | | | | | | | | | | | | | | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | | | Years Ended December 31, | | (dollars in millions) | 2024 | | 2023 | | 2022 | | Net sales | $ | 3,818.1 | | | $ | 3,852.8 | | | $ | 3,753.9 | | | Cost of products sold | 2,362.0 | | | 2,368.0 | | | 2,424.3 | | | Gross profit | 1,456.1 | | | 1,484.8 | | | 1,329.6 | | | Gross profit margin % | 38.1 | % | | 38.5 | % | | 35.4 | % | | Selling, general and administrative expenses | 739.3 | | | 727.4 | | | 670.9 | | | Restructuring and impairment expenses | 17.6 | | | 18.8 | | | | | | Interest expense | 6.7 | | | 12.0 | | | 9.4 | | | Other (income) expense-net | (8.5) | | | (6.9) | | | 425.6 | | | Earnings before provision for income taxes | 701.0 | | | 733.5 | | | 223.7 | | | Provision for (benefit from) income taxes | 167.4 | | | 176.9 | | | (12.0) | | | Net Earnings | $ | 533.6 | | | $ | 556.6 | | | $ | 235.7 | |\nOur sales in 2024 were $3,818.1 million, a decrease of $34.7 million compared to 2023 sales of $3,852.8 million. Our decrease in net sales was primarily driven by lower water heater volumes in North America, lower sales in China, and unfavorable currency translation of approximately $18 million due to the depreciation of foreign currencies compared to the U.S. dollar, which more than offset our higher boiler sales and pricing actions. Our 2024 and 2023 acquisitions of water treatment companies in North America added approximately $18 million of incremental net sales in 2024.\n\nOur 2024 gross profit margin of 38.1 percent decreased compared to 38.5 percent in 2023. The lower gross profit margin in 2024 compared to 2023 was primarily due to higher production costs and operational inefficiencies associated with volume volatility, which outpaced our pricing actions.\n\nSelling, general, and administrative (SG&A) expenses were $739.3 million in 2024, or $11.9 million higher than in 2023. The increase in SG&A expenses in 2024 compared to the prior year was primarily due to higher employee costs from increased wages and higher selling and advertising expenses to support our strategic initiatives.\n\nWe recognized $17.6 million of restructuring and impairment expenses during the year ended December 31, 2024. Of these expenses, $6.3 million was related to our water treatment business in the North America segment and was a result of a profitability improvement strategy that prioritizes improving our cost structure and emphasizes our more profitable channels. In the Rest of World segment, restructuring included severance costs in China of $11.3 million and was related to the right sizing of that business for current market conditions. Restructuring and impairment expenses in 2023 were $18.8 million, of which $15.7 million was recorded in the Rest of World segment and $3.1 million was recorded in Corporate Expense and related primarily to the sale of our business in Turkey.\n\nInterest expense was $6.7 million in 2024, compared to $12.0 million in 2023. The decrease in interest expense in 2024 compared to last year was primarily due to lower average debt levels.\n\nOther (income) expense, net was $8.5 million of income in 2024 compared to income of $6.9 million in 2023. The increase in other income was driven by lower foreign currency translation losses compared to last year, partially offset by lower interest income from lower average cash balances.\n\nOur effective income tax rate in 2024 was lower compared to 2023. The change in the effective income tax rate in 2024 compared to the prior year was primarily due to the restructuring and impairment expense recorded in 2023 with no associated tax benefit. We estimate that our annual effective income tax rate for the full year of 2025 will be approximately 24 to 24.5 percent.\n\nWe are providing non-U.S. Generally Accepted Accounting Principles (GAAP) measures (adjusted earnings, adjusted earnings per share (EPS), total segment earnings, adjusted segment earnings, and adjusted corporate expense) that exclude the impact of restructuring and impairment expenses and pension settlement income. Reconciliations from GAAP measures to non-GAAP measures are provided in the Non-GAAP Measures section below. We believe that the measures of adjusted earnings, adjusted EPS, total segment earnings, adjusted segment earnings, and adjusted corporate expense provide useful information to investors about our performance and allow management and our investors to better understand our performance between periods without regard to items that we do not consider to be a component of our core operating performance or recurring in nature.

Metadata:
"metadata": {{
    "cik": "91142",
    "company": "SMITH A O CORP",
    "filing_type": "10-K",
    "filing_date": "2025-02-11",
    "period_of_report": "2024-12-31",
    "domain": "finance"
}}
question: "What were SMITH A O CORP's consolidated sales for the year ended December 31, 2024?",
answer: "$3,818.1 million"

Your responses should ONLY include:
- question
- answer
"""

finance_onehop_query_prompt_user = """
# Triplet Information:
Entity 1: {entity_1}
Relation: {relation}
Entity 2: {entity_2}

# Context:
{chunk_text}

# Metadata:
{metadata}

Your response:
"""

finance_multihop_query_prompt_system = """
You are an expert in finance and financial analysis. Generate a complex and meaningful finance-related question and answer based on the provided multi-hop triplet type.

Generation Instructions:
Requirements for question generation:
1. Create a clear and specific finance question that requires the information in the triplet to answer.
2. The question should be natural, use proper financial terminology, and be something a financial analyst might ask.
3. If the triplet cannot form a meaningful financial question (or the question is too general/naive), you MUST return an empty string for question and answer.
4. If the fact in entity 1 or entity 2 is slightly different from the fact in the context, you should use the fact in the context as the answer of question.
- For example, if the entity 2 is "$3,818.1 million in 2024" in the triplet, but the context mentions "3,818.1 million" in 2023, you should use the fact in the context to generate question.

Requirements for answer generation:
1. You answer should only have few words (not sentence). It will ONLY contain the core information. 
2. Make sure that the answer can always be found in the triplet.
3. If the fact in entity_1 or entity_2 (the entity you selected as the answer) is slightly different from the fact in the context, you should use the fact in the context as the answer.
- For example, if the entity_2 (the entity you selected as the answer) is "$3,818.1 million in 2024" in the triplet, but the context mentions "3,818.1 million" in 2023, you should use the fact in the context to generate answer.

Instructions for multi-hop triplets:
1. You must generate the multi-hop triplets based on the provided multi-hop type.
2. If the given triplets cannot form a meaningful multi-hop question (or the question is too general/naive), you MUST return an empty string for question and answer.

Types of multi-hop triplets:
- Chain Triplet
- Star Triplet
- Inverted Star Triplet

Short description of each type:
- Chain Triplet: A chain triplet is a sequence of triplets where the second entity of one triplet is the same as the first entity of the next triplet.
- Star Triplet: A star triplet is a sequence of triplets where the multiple triplets share same starting entity_1
- Inverted Star Triplet: A inverted star triplet is a sequence of triplets where the multiple triplets share same ending entity_2

Examples:
- Chain Triplet:
{{"entity_1": "NVIDIA","relation": "develops","entity_2": "GeForce GPUs","source_sentence": "We operate in two reportable segments: Graphics and Compute & Networking. Our Graphics segment includes GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service, and related infrastructure and services."}}
{{"entity_1": "GeForce GPUs","relation": "designed for","entity_2": "gaming and PCs","source_sentence": "We operate in two reportable segments: Graphics and Compute & Networking. Our Graphics segment includes GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service, and related infrastructure and services."}}

This forms a chain of triplets: NVIDIA -> develops -> GeForce GPUs -> designed for -> gaming and PCs
which can eventually form a multi-hop query like: "What is the name of chip did NVIDIA develop for gaming and PCs?"

Answer: "GeForce GPUs"

- Star Triplet:
{{"entity_1": "Panasonic","relation_1": "supplies lithium-ion cells to","entity_2": "Tesla","source_sentence": "Panasonic Corporation supplies lithium-ion cells to Tesla, Inc."}}
{{"entity_1": "Panasonic","relation_2": "holds","entity_2": "ISO 9001 certification (battery plan)","source_sentence": "Panasonic Corporation holds ISO 9001 certification for its battery production process."}}

These two triplets can eventually form a star-shaped question like: "Which Tesla battery supplier is also ISO 9001-certified?"
Answer: "Panasonic"

- Inverted Star Triplet:
{{"entity_1": "Ford",   "relation": "manufactures F-150 trucks in", "entity_2": "Dearborn, Michigan plant", "source_sentence": "We manufacture F-150 trucks in our Dearborn, Michigan plant."}}
{{"entity_3": "SK On", "relation": "supplies batteries to", "entity_2": "Dearborn, Michigan plant", "source_sentence": "SK On will supply batteries to Ford's Dearborn, Michigan plant under a long-term agreement."}}

This will form a question like: "Which company supplies batteries to F-150 truck manufacturing plant in Dearborn, Michigan?"
Answer: "SK On"

Your responses should ONLY include:
- question
- answer
"""

finance_multihop_query_prompt_user = """
# Triplets Information:
{triplets}

# Contexts:
{chunk_text}

# Metadata:
{metadata}

# Multi-hop Type:
{multi_hop_type}

Your responses:
"""


econ_triplets_prompt_system = """
You are an economic analyst skilled at interpreting OECD Economic Surveys. Your task is to extract structured triplets consisting of **{{"entity_1", "relation", "entity_2"}}** from provided consecutive text chunks from a single OECD Economic Survey. Each triplet must **be supported explicitly by one specific chunk**, but other chunks can be referenced to form insightful, multi-hop triplets. You should include the **source chunk ID** and **source sentence** as the metadata of the triplets.

# TASK: EXTRACT STRUCTURED MULTI-HOP TRIPLETS

Extract triplets fitting these multi-hop categories:

- Connected Chain
- Star
- Inverted Star

### 1. Connected Chain Triplets:

- Extract an initial triplet: <entity_1, relation, entity_2>.
- Then identify subsequent triplets where entity_2 of the previous triplet becomes entity_1 of the next.
- Ideally, different subsequent triplets should be sourced from different chunks.
- Extract as many meaningful chains as possible.
- Skip if no valid connected chain is available.

**Example:**

- {{"entity_1": "Luxembourg", "relation": "implemented", "entity_2": "free public transport"}}
- {{"entity_1": "free public transport", "relation": "aims to reduce", "entity_2": "carbon emissions"}}

### 2. Star Triplets:

- One root entity branching into multiple distinct relationships.
- Each branch must independently derive from a unique chunk.
- Skip if no meaningful star relationship is possible.

**Example:**

- {{"entity_1": "Luxembourg", "relation": "invests in", "entity_2": "renewable energy"}}
- {{"entity_1": "Luxembourg", "relation": "develops", "entity_2": "sustainable transport infrastructure"}}

### 3. Inverted Star Triplets:

- Two distinct entities connected through a shared attribute (entity_2).
- Relations may differ and offer varied perspectives on the attribute.
- Skip if no valid inverted star relationship is possible.

**Example:**

- {{"entity_1": "Luxembourg", "relation": "faces challenges in", "entity_2": "housing affordability"}}
- {{"entity_1": "OECD recommendations", "relation": "address", "entity_2": "housing affordability"}}

## REQUIRED STRUCTURE:

Each extracted triplet must include:

- entity_1 (str)
- relation (str)
- entity_2 (str)
- answer_chunk_id (str)
  - The chunk ID is at the very beginning of each text chunk, such as "Chunk ID: economics_0e32d909-en_chunk_9". 
  - You should copy the chunk ID where the triplet is extracted from as the "answer_chunk_id".
- source_sentence (str)
  - Extracted exactly from the supporting chunk, COPY WORD BY WORD.
  - If sourced from a table, strictly include relevant row, column, and specific data only.

## CRITICAL INSTRUCTIONS:

### Relations:

- Generalized and reusable across similar economic and policy contexts.
- Concise and specific (2-4 words preferred).
- Use standard economic and policy terminology.
- Avoid specific dates or overly detailed references in the relations.

**Good Examples:**

- "implemented", "faces challenges in", "invests in", "promotes"

**Bad Examples:**

- "introduced free transport in 2020", "planned reforms announced in 2023"

### Entities:

- Clearly specify entities (avoid general terms like "the country" or "the government").
- Maintain consistent terminology when referring to similar concepts, such as using "Luxembourg" all the time instead of using "Luxembourg government" sometimes.
- Include specific, detailed information relevant to economic policies, recommendations, or outcomes.
- For table-derived entities, clearly indicate row, column, and description.

## Goal:

Try to extract 15 to 20 triplets. If no valid connected triplets can be extracted, return an empty array: []"""

econ_triplets_prompt_user = """Text Chunk:
{chunk_text}

Metadata:
- Country Surveyed: {country_name}
- Survey Year: {survey_year}"""

econ_onehop_query_prompt_system = '''
Create an economics-related natural question-answer pair using a relation triplet (entity_1, relation, entity_2) based on the text context and the file metadata where the triplet was extracted.

# Requirements

- The question and answer should be entirely based on the given text context; that is, one can only generate the correct answer from the information available in the context.
- Always use "{file_country}" instead of "{file_country} government," "government," or "country" to make the query more specific.
- You should use entity_1 or entity_2 as the answer to the question and construct the question using the other entity and relation with appropriate context information.
- Aim to formulate questions that appear natural and are likely to be asked by a human.
- Avoid generating questions that are overly general or vague, where multiple ground truth chunks could answer the question or it would be hard to retrieve the ground truth chunk given the question. You MUST return an EMPTY string for question and answer in this case.

# Examples

Example 1:
Triplet:
{{"entity_1": "inflation", "relation": "is", "entity_2": "2.9% in 2023"}}

Text Context:
Table 1. Real GDP growth will gradually pick up \nAnnual growth rates, $\\%$ , unless specified \n|   | 2023 | 2024 | 2025 | 2026 |\n| --- | --- | --- | --- | --- |\n| Real GDP | -0.7 | 1.0 | 2.1 | 2.3 |\n| Unemployment rate (% labour force) | 5.2 | 5.7 | 5.9 | 5.8 |\n| Inflation (harmonised index of consumer prices) | 2.9 | 2.3 | 2.1 | 1.9 |\n| Government budget balance (% of GDP) | -0.8 | 1.0 | -0.0 | -0.1 |\n| Public debt, Maastricht definition (% GDP) | 24.9 | 22.9 | 22.9 | 23.5 |Source: OECD, Economic Outlook No. 116.

Metadata:
- File Type: OECD Economic Surveys
- Country Surveyed: Luxembourg
- Survey Year: 2023

Output:
{{"question": "What is the inflation of Luxembourg in 2023?", "answer": "2.9%"}}

Example of Vague Triplet (Should Return Empty):
Triplet:
{{"entity_1": "luxembourg", "relation": "should maintain", "entity_2": "prudent fiscal policy"}}

Text Context:
Despite several major shocks and the recent slowdown, Luxembourg has grown vigorously over recent decades. Living standards are among the highest in the OECD. The stable institutional framework, responsive regulation, and a relatively favorable tax regime have attracted foreign investment and foreign workers, especially in finance and related business services. Yet, the growth model based on rapid labor force expansion has reached its limits. Productivity has stagnated over the past 15 years, congestion has increased, and housing has become less affordable for many residents. Policies fostering the transition to a more sustainable growth model based on skills and innovation need to be prioritized while ensuring the sustainability of the pension system and addressing climate change. Recovery is now underway, but fiscal policy should remain prudent. Economic activity is set to gradually pick up as inflation remains low and financial conditions ease. As the recovery takes hold, fiscal policy should remain prudent to prepare for rising expenditure pressures, including on pensions, public investment related to the green and digital transitions, and defense.

Metadata:
- File Type: OECD Economic Surveys
- Country Surveyed: Luxembourg
- Survey Year: 2023

Output:
{{"question": "", "answer": ""}}

# Output Format

Respond in JSON format with "question" and "answer" fields encapsulating the formulated question and its answer.

# Notes

Ensure questions are specific to the context provided, emphasizing precision and clarity in wording. If no singular answer emerges due to generality, opt for returning an empty dictionary to indicate an unsuitably specific query.'''

econ_onehop_query_prompt_user = '''Triplet:
{triplet}

Text Context:
{chunk_text}

Metadata:
- File Type: {file_type}
- Country Surveyed: {file_country}
- Survey Year: {file_year}

Output:
'''

econ_multihop_query_prompt_system = '''
You are a benchmark designer creating **multi-hop retrieval questions** based on three types of multi-hop triplets.

### Input
• Triplet 1  = ({{head1}}, {{rel1}}, {{tail1}})    ← extracted from Chunk 1
• Triplet 2  = ({{head2}}, {{rel2}}, {{tail2}})    ← extracted from Chunk 2
• Chunk 1: {{chunk1}}
• Chunk 2: {{chunk2}}

### Multi-hop Triplets DEFINITIONS
1. Chain Triplets
- Gurantee: {{tail1}} == {{head2}}
- Define A = {{head1}}, B = {{tail1}} / {{head2}}, C = {{tail2}}

2. Star-shaped Triplets
- Gurantee: {{head1}} == {{head2}}
- Define A = {{tail1}}, B = {{head1}} / {{head2}}, C = {{tail2}}

3. Inverted-star-shaped Triplets
- Gurantee: {{tail1}} == {{tail2}}
- Define A = {{head1}}, B = {{tail1}} / {{tail2}}, C = {{head2}}

### GOAL
Write ONE natural-language **multi-hop** question that *requires* evidence from both chunks and answer it succinctly (no full sentences, only essential information).

### ALGORITHM
1. Decide whether the final answer will be **A** or **C**.
   - Pick **A** if you can phrase the question so the solver must:
     - hop-1: use (C, rel2) to identify B,
     - hop-2: use (B, rel1) to reach A.

   - Pick **C** if you can phrase the question so the solver must:
     - hop-1: use (A, rel1) to identify B,
     - hop-2: use (B, rel2) to reach C.

2. Write a fluent, specific, and natural question that:
   - References the **pivot B** indirectly (via the opposite hop as above).
   - Omits the answer itself.
   - Cannot be answered from a single chunk.
   - Includes detailed and specific context from the source text chunks. DO NOT just use "according to OECD Economic Survey".
      - BAD example: "What is the primary export sector of the country that faces risk from global supply chain disruptions?" (Too vague; could refer to any country)
      - GOOD example: "What is the primary export sector of the country that faces risk from global supply chain disruptions in Q3 2021?" (Specific to the context and time frame)

3. Return the answer based on A or C. Ensure the answer precisely matches the facts provided in the context.

### EXAMPLE
{{"entity_1": "forward-looking fuel-tax trajectory", "relation_1": "would reduce", "entity_2": "reliance on combustion-engine cars"}}
{{"entity_1": "reliance on combustion-engine cars", "relation_2": "drives", "entity_2": "transport-sector emissions"}}

*question*: Which forward-looking tax trajectory is proposed to cut the main driver of transport-sector emissions?
*answer*: forward-looking fuel-tax trajectory

### QUALITY CHECKS
- **Pivot-rarity**: B must be distinctive (≥ 2 meaningful words, not generic terms like “measures”, “it”, “the company”). If B is too generic, output **empty strings for the question and answer**.
- **Negative-distractor safety**: Ask could a system answer your question after retrieving only *one* chunk? If yes, output **empty strings for the question and answer**.

### OUTPUT
Respond in JSON format with question and answer only as shown below:
{{
  "question": "...", 
  "answer": "..."
}}'''

econ_multihop_query_prompt_user = '''
# Triplets Information:
{triplets}

# Contexts:
{chunk_text}

# Metadata:
{metadata}

# Multi-hop Type:
{multi_hop_type}

Your responses:'''


policy_triplets_prompt_system = '''
You are a policy analyst skilled in evaluating Consolidated Annual Performance and Evaluation Reports (CAPERs). Your task is to extract structured triplets consisting of **{{"entity_1", "relation", "entity_2"}}** from provided consecutive text chunks from a CAPER report. Each triplet must **be supported explicitly by one specific chunk**, but other chunks can be referenced to form insightful, multi-hop triplets. You should include the **source chunk ID** and **source sentence** as the metadata of the triplets.

### TYPES OF MULTI-HOP TRIPLETS TO EXTRACT

### 1. Connected Chain Triplets:

- Extract an initial triplet: <entity_1, relation, entity_2>.
- Then identify subsequent triplets where entity_2 of the previous triplet becomes entity_1 of the next.
- Ideally, different subsequent triplets should be sourced from different chunks.
- Extract as many meaningful chains as possible.
- Skip if no valid connected chain is available.

**Example:**

{{"entity_1": "HOME funds", "relation": "support", "entity_2": "affordable housing construction"}}
{{"entity_1": "affordable housing construction", "relation": "targets", "entity_2": "low and moderate income households"}}

### 2. **Star-shaped Triplets**

- One root entity branching into multiple distinct relationships.
- Each branch must independently derive from a unique chunk.
- Skip if no meaningful star-shaped relationship is possible.

**Example:**

{{"entity_1": "HOME funds", "relation": "benefitted", "entity_2": "Black or African American families"}}
{{"entity_1": "HOME funds", "relation": "used to support", "entity_2": "renter-occupied households"}}

### 3. **Inverted-star-shaped  Triplets**

- Two **distinct entities** connected through a **shared attribute** (entity_2).
- Relations may differ and offer varied perspectives on the attribute.
- Skip if no valid inverted-star shaped triplets exists.

**Example:**

{{"entity_1": "URA of Pittsburgh", "relation": "administers", "entity_2": "HOME funds"}}
{{"entity_1": "City of Pittsburgh", "relation": "allocates", "entity_2": "HOME funds"}}

**Example:**

{{"entity_1": "CDBG funds", "relation": "support", "entity_2": "public service activities"}}
{{"entity_1": "public service activities", "relation": "benefit", "entity_2": "low-income residents"}}
{{"entity_1": "low-income residents", "relation": "qualify for", "entity_2": "CDBG funds"}}

## REQUIRED OUTPUT FORMAT

Each extracted triplet must include:

- entity_1 (str)
- relation (str)
- entity_2 (str)
- answer_chunk_id (str)
  - The chunk ID is at the very beginning of each text chunk, such as "Chunk ID: policy_0e32d909-en_chunk_9". 
  - You should copy the chunk ID where the triplet is extracted from as the "answer_chunk_id".
- source_sentence (str)
  - Extracted exactly from the supporting chunk, COPY WORD BY WORD.
  - If sourced from a table, strictly include relevant row, column, and specific data only.

## RELATION & ENTITY GUIDELINES

### RELATIONS

- Relations should be **specific**, factual, and concise (2-4 words). Ensure relations are **reusable across CAPERs** and not report-specific

**Good Examples:** "provides", "targets", "allocates funds to", "administers", "assists with housing", "benefits"

### ENTITIES

- Maintain consistent terminology when referring to similar concepts
- Use **specific project types** and **target groups** (e.g., "emergency shelter beds", "low-income renters", "youth services")
- Always name the **city**, **fund**, or **program** as an explicit entity (e.g., "City of Pittsburgh", "CDBG Program", "HOPWA funds"). Avoid vague nouns like "the plan", "it", or "the program"
- For table-derived entities, clearly indicate row, column, and description.

**Good Examples:** "HOPWA program", "low-income households", "rental unit rehabilitation"

## GOAL

Extract **15 to 25 triplets**, covering a mix of categories above. Skip any category if no valid data appears. Always include exact source details.'''

policy_triplets_prompt_user = '''Text Chunk:
{chunk_text}

Metadata:
- Grantee: {grantee_name}
- Report Year: {year}'''

policy_onehop_query_prompt_system = '''Create a **natural and fact-based question-answer pair** using a **relation triplet** (`entity_1`, `relation`, `entity_2`) based on the **given text context** and **CAPER metadata**.

### Requirements:

- The **question and answer must be directly answerable** using only the **provided text context** — no outside inference or general knowledge is allowed.
- Construct the **question using one entity and the relation**, and use the **other entity as the answer**.
- Questions should sound natural and be specific enough to uniquely match the chunk. DO NOT only use the file metadata, use the specific chunk context instead.
- Avoid general, vague, or overlapping questions that other chunks across the report could answer.
- If the triplet is **too vague or general**, return empty question and answer: {{"question": "", "answer": ""}}

### Example 1:

**Triplet**:

{{"entity_1": "City of Pittsburgh", "relation": "allocated", "entity_2": "$8.3 million in CDBG funds"}}

**Text Context**:

In 2023, the City of Pittsburgh allocated a total of $8.3 million in Community Development Block Grant (CDBG) funds to a range of public service, housing, and infrastructure programs that primarily benefit low- and moderate-income residents.

**Metadata**:

- File Type: CAPER
- Grantee City: Pittsburgh
- Report Year: 2023

**Output**:

{{
  "question": "How much in CDBG funds did Pittsburgh allocate in 2023?",
  "answer": "$8.3 million"
}}

### Example 2:

**Triplet**:

{{"entity_1": "emergency shelter services", "relation": "benefited", "entity_2": "456 homeless individuals"}}

**Text Context**:

ESG funds supported emergency shelter services for homeless individuals across Pittsburgh. In total, 456 individuals were served through various outreach, case management, and shelter programs.

**Metadata**:

- File Type: CAPER
- Grantee City: Pittsburgh
- Report Year: 2023

**Output**:

{{
  "question": "How many individuals benefited from emergency shelter services in Pittsburgh?",
  "answer": "456 homeless individuals"
}}

### Example of Vague Triplet (Should Return Empty):

**Triplet**:

{{"entity_1": "City of Pittsburgh", "relation": "supported", "entity_2": "community programs"}}

**Text Context**:

In 2023, the City of Pittsburgh supported various community programs aimed at improving local infrastructure and public services.

**Metadata**:

- File Type: CAPER
- Grantee City: Pittsburgh
- Report Year: 2023

**Output**:

{{"question": "", "answer": ""}}

### Output Format:

Respond in the following JSON format:

{{
  "question": "...",
  "answer": "..."
}}

Return an **empty question and answer** (`{{"question": "", "answer": ""}}`) if no precise and factual Q&A pair can be generated.'''

policy_onehop_query_prompt_user = '''
Triplet:
{triplet}

Text Context:
{chunk_text}

Metadata:
- File Type: {file_type}
- Grantee City: {file_grantee}
- Report Year: {file_year}

Output:'''

policy_multihop_query_prompt_system = '''
You are a benchmark designer creating **multi-hop retrieval questions** based on three types of multi-hop triplets.

### Input
• Triplet 1  = ({{head1}}, {{rel1}}, {{tail1}})    ← extracted from Chunk 1
• Triplet 2  = ({{head2}}, {{rel2}}, {{tail2}})    ← extracted from Chunk 2
• Chunk 1: {{chunk1}}
• Chunk 2: {{chunk2}}

### Multi-hop Triplets DEFINITIONS
1. Chain Triplets
- Gurantee: {{tail1}} == {{head2}}
- Define A = {{head1}}, B = {{tail1}} / {{head2}}, C = {{tail2}}

2. Star-shaped Triplets
- Gurantee: {{head1}} == {{head2}}
- Define A = {{tail1}}, B = {{head1}} / {{head2}}, C = {{tail2}}

3. Inverted-star-shaped Triplets
- Gurantee: {{tail1}} == {{tail2}}
- Define A = {{head1}}, B = {{tail1}} / {{tail2}}, C = {{head2}}

### GOAL
Write ONE natural-language **multi-hop** question that *requires* evidence from both chunks and answer it succinctly (no full sentences, only essential information).

### ALGORITHM
1. Decide whether the final answer will be **A** or **C**.
   - Pick **A** if you can phrase the question so the solver must:
     - hop-1: use (C, rel2) to identify B,
     - hop-2: use (B, rel1) to reach A.

   - Pick **C** if you can phrase the question so the solver must:
     - hop-1: use (A, rel1) to identify B,
     - hop-2: use (B, rel2) to reach C.

2. Write a fluent, specific, and natural question that:
   - References the **pivot B** indirectly (via the opposite hop as above).
   - Omits the answer itself.
   - Cannot be answered from a single chunk.
   - Includes detailed and specific context from the source text chunks. DO NOT only use the file metadata, use the specific chunk context instead.
      - BAD example: "What is the primary export sector of the country that faces risk from global supply chain disruptions?" (Too vague; could refer to any country)
      - GOOD example: "What is the primary export sector of the country that faces risk from global supply chain disruptions in Q3 2021?" (Specific to the context and time frame)

3. Return the answer based on A or C. Ensure the answer precisely matches the facts provided in the context.

### EXAMPLE
{{"entity_1": "forward-looking fuel-tax trajectory", "relation_1": "would reduce", "entity_2": "reliance on combustion-engine cars"}}
{{"entity_1": "reliance on combustion-engine cars", "relation_2": "drives", "entity_2": "transport-sector emissions"}}

*question*: Which forward-looking tax trajectory is proposed to cut the main driver of transport-sector emissions?
*answer*: forward-looking fuel-tax trajectory

### QUALITY CHECKS
- **Pivot-rarity**: B must be distinctive (≥ 2 meaningful words, not generic terms like “measures”, “it”, “the company”). If B is too generic, output **empty strings for the question and answer**.
- **Negative-distractor safety**: Ask could a system answer your question after retrieving only *one* chunk? If yes, output **empty strings for the question and answer**.

### OUTPUT
Respond in JSON format with question and answer only as shown below:
{{
  "question": "...", 
  "answer": "..."
}}'''

policy_multihop_query_prompt_user = '''
# Triplets Information:
{triplets}

# Contexts:
{chunk_text}

# Metadata:
{metadata}

# Multi-hop Type:
{multi_hop_type}

Your responses:'''


finance_relation_similarity_prompt = '''
Instruct: Represent this financial knowledge graph relation term for semantic similarity matching with other relations.
Query: 
'''

econ_relation_similarity_prompt = '''
Instruct: Represent this economic knowledge graph relation term for semantic similarity matching with other relations.
Query: 
'''

policy_relation_similarity_prompt = '''
Instruct: Represent this governance knowledge graph relation term for semantic similarity matching with other relations.
Query: 
'''


grammar_perturbation_prompt_system = "You are a copy editor. Rephrase the following so only grammar, punctuation, and word order change; do not replace content words with synonyms. Return only the rephrased query, no other text."
grammar_perturbation_prompt_user = "Query: {query}\nRephrased query:"

irrelevant_info_prompt_system = """
You are an assistant that rewrites user queries for adversarial testing. 
Take the query below and insert 1 pieces of information that are relevant to the domain but irrelevant to answering the question itself

More requirements:
- Do not change the meaning of the query
- Do not repeat words
- Keep the original wording and punctuation of the question.
- Return only the perturbed query.
"""
irrelevant_info_prompt_user = "Query: {query}\nRewritten query:"

rag_prompt_system = """
You are a {domain} expert. You are given a {domain} question and one or multiple contexts.
Your task is to answer the question strictly based on the these contexts.
You should think step by step and answer the question in a detailed and comprehensive way. Please return the detailed reasoning process in the cot_answer part.

Requirements:
- Your answer is short and concise, do not return any other text in the answer part.
  - Example #1: "What is the United States' GDP in 2024?"
  - Good: "$31.1 trillion"
  - Bad: "According to the context, as my knowledge, the answer is $31.1 trillion"
  - Example #2: "Who is the president of the United States from 2021 to 2025?"
  - Good: "Joe Biden"
  - Bad: "The president of the United States from 2021 to 2025 is Joe Biden, according to my knowledge"
- If the question is not related to the context, strictly return "no such info" for answer part. Do not return any other text in such case.

Here are some examples of how to answer based on the given context:

Example 1:
Question: What was Apple's revenue in Q2 2023?
Context: [Doc] Apple Inc. reported financial results for its fiscal 2023 second quarter ended April 1, 2023. The Company posted quarterly revenue of $94.8 billion, down 2.5 percent year over year.

cot_answer: The question asks about Apple's revenue in Q2 2023. According to the context, Apple reported quarterly revenue of $94.8 billion for its fiscal 2023 second quarter ended April 1, 2023. This represents a decrease of 2.5 percent year over year.
answer: $94.8 billion

Example 2:
Question: What is Luxembourg's approach to public transport?
Context: [Doc] On March 1, 2020, Luxembourg became the first country to make all public transport free, including buses, trains, and trams. This policy aims to reduce traffic congestion and carbon emissions while promoting sustainable mobility solutions across the country.

cot_answer: The question asks about Luxembourg's approach to public transport. According to the context, Luxembourg made all public transport free on March 1, 2020, becoming the first country to do so. This includes buses, trains, and trams. The goal of this policy is to reduce traffic congestion and carbon emissions while promoting sustainable mobility solutions.
answer: Free public transport for all

Example 3:
Question: How many homeless individuals received emergency shelter services in Pittsburgh?
Context: [Doc] The City of Pittsburgh allocated CDBG funds to various community programs including affordable housing initiatives. The HOME program supported the construction of 45 new housing units for low-income families.

cot_answer: The question asks about the number of homeless individuals who received emergency shelter services in Pittsburgh. After reviewing the context carefully, I don't see any information about emergency shelter services for homeless individuals or any numbers related to this. The context only mentions CDBG funds for community programs and the HOME program supporting 45 new housing units for low-income families. There is no specific information about homeless emergency shelter services.
answer: no such info

Example 4:
Question: What were Smith A O Corp's consolidated sales for the year ended December 31, 2024?
Context: [Doc] In this section, we discuss the results of our operations for 2024 compared with 2023. Our sales in 2024 were $3,818.1 million, a decrease of $34.7 million compared to 2023 sales of $3,852.8 million. Our decrease in net sales was primarily driven by lower water heater volumes in North America, lower sales in China, and unfavorable currency translation of approximately $18 million due to the depreciation of foreign currencies compared to the U.S. dollar, which more than offset our higher boiler sales and pricing actions.

cot_answer: The question asks about Smith A O Corp's consolidated sales for the year ended December 31, 2024. According to the context, the sales in 2024 were $3,818.1 million, which was a decrease of $34.7 million compared to 2023 sales of $3,852.8 million. The context explains that this decrease was primarily due to lower water heater volumes in North America, lower sales in China, and unfavorable currency translation of approximately $18 million.
answer: $3,818.1 million

Output Format:
- cot_answer: detailed reasoning process
- answer: concise answer to the question
"""

rag_prompt_user = """
Question: {question}
Context: {context}
Output:
"""